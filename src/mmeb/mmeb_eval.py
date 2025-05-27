import logging
import torch
from tqdm import tqdm

from fuse_clip.fuse_clip_utils import get_cos_sims
from mmeb.mmeb_dataset import EvalDataset
from mmeb.mmeb_utils import (mmeb_eval_collator, DataArguments, MMEB_EVAL_SUBSETS, MMEB_EVAL_SHORT_SUBSETS)

from datasets import load_dataset
from torch.utils.data import DataLoader

from open_clip import build_zero_shot_classifier, OPENAI_IMAGENET_TEMPLATES



@torch.no_grad()
def mmeb_eval(
        model: torch.nn.Module, image_processor: callable, short: bool, tokenizer: callable,
        batch_size: int, workers: int,
        txt_modifier_qry: callable, txt_modifier_tgt: callable, ensemble_prompt: bool = False,
        return_extended: bool = False,
        subsets: list[str] = None,
) -> dict:
    # eval on MMEB eval dataset

    data_args = DataArguments()
    data_args.subset_name = MMEB_EVAL_SUBSETS if not short else MMEB_EVAL_SHORT_SUBSETS
    if subsets is not None:
        data_args.subset_name = subsets

    res_dict = {}
    for i_subset, subset in enumerate(data_args.subset_name):
        logging.info(f"evaluating on {subset} ({i_subset+1}/{len(data_args.subset_name)})")

        qry_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
            image_processor=image_processor,
            txt_modifier=txt_modifier_qry
        )
        tgt_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
            image_processor=image_processor,
            txt_modifier=txt_modifier_tgt
        )

        qry_loader = DataLoader(
            qry_dataset,
            batch_size=batch_size,
            collate_fn=mmeb_eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
        )
        tgt_loader = DataLoader(
            tgt_dataset,
            batch_size=batch_size,
            collate_fn=mmeb_eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
        )

        # collect all query embeddings
        qry_embeds_dct = {}
        for el in tqdm(qry_loader, desc="encode queries"):
            qry_imgs = el["imgs"].cuda()
            qry_img_paths = el["img_paths"]
            qry_txts = el["txts"]
            qry_txt_ids = tokenizer(qry_txts).cuda()
            embed = model(image=qry_imgs, text=qry_txt_ids, force_fused=True).detach().cpu()
            qry_embeds_dct.update(
                {(qry_img_paths[i], qry_txts[i]): embed[i] for i in range(len(qry_txts))}
            )

        # collect all target embeddings
        tgt_embeds_dct = {}
        for el in tqdm(tgt_loader, desc="encode targets"):
            tgt_imgs = el["imgs"].cuda()
            tgt_img_paths = el["img_paths"]
            tgt_txts = el["txts"]
            if not ensemble_prompt or tgt_imgs.numel() > 0:
                tgt_txt_ids = tokenizer(tgt_txts).cuda()
                embed = model(image=tgt_imgs, text=tgt_txt_ids, force_fused=True).detach().cpu()
            else:
                assert tgt_imgs.numel() == 0
                embed = build_zero_shot_classifier(
                    model=model, tokenizer=tokenizer, classnames=tgt_txts, templates=OPENAI_IMAGENET_TEMPLATES,
                    device="cuda",
                ).T.detach().cpu()

            for i in range(len(tgt_txts)):
                k = (tgt_img_paths[i], tgt_txts[i])
                tgt_embeds_dct[k] = embed[i]

        # print(f"[qry] `{qry_txts[0]}` [tgt] `{tgt_txts[0]}`")  # Example output

        print_cos_sims = False
        if print_cos_sims:
            qry_tens = torch.stack(list(qry_embeds_dct.values()))
            tgt_tens = torch.stack(list(tgt_embeds_dct.values()))
            cos_sim_qry = get_cos_sims(qry_tens)
            cos_sim_tgt = get_cos_sims(tgt_tens)

        loaded = False
        while not loaded:
            try:
                eval_data = load_dataset(
                    data_args.dataset_name,
                    subset,
                    split=data_args.dataset_split,
                )
                loaded = True
            except Exception as e:
                print(f"failed to load dataset {data_args.dataset_name} {subset}, retrying..")
                print(e)
                continue

        n_correct = 0
        cos_sim = 0.
        preds = {}
        for row in tqdm(eval_data, desc="evaluate"):
            qry_txt = txt_modifier_qry(row["qry_text"], subset=subset)
            qry_embeds = qry_embeds_dct[
                (row["qry_img_path"], qry_txt)
            ]

            tgt_embeds = []
            for i in range(len(row["tgt_text"])):
                k = (row["tgt_img_path"][i], txt_modifier_tgt(row["tgt_text"][i], subset=subset))
                tgt_embeds.append(tgt_embeds_dct[k])
            tgt_embeds = torch.stack(tgt_embeds)

            logits = qry_embeds.unsqueeze(0) @ tgt_embeds.T
            pred = logits.argmax().item()
            qry_txt_cln = qry_txt.replace('\"', '')
            preds.update(  # for plotting
                {f"{row['qry_img_path']}__{qry_txt_cln}":
                     f"{pred}__{row['tgt_img_path'][pred]}__{txt_modifier_tgt(row['tgt_text'][pred], subset=subset)}"}
            )
            if pred == 0:
                n_correct += 1
            cos_sim += logits.squeeze()[0].item()

        n = len(eval_data)
        acc = n_correct / n
        cos_sim /= n
        print(f"Score {subset}: {n_correct} / {n} = {acc*100 :.2f}%")
        if print_cos_sims:
            modality_gap = qry_tens.mean(dim=0) - tgt_tens.mean(dim=0)
            modality_gap = modality_gap.norm().item()
            print(f"[cos sim] qry: {cos_sim_qry:.4f}, tgt: {cos_sim_tgt:.4f}, qry-tgt: {cos_sim:.4f}"
                  f"[gap] {modality_gap:.4f}")

        if return_extended:
            res_dict[subset] = {
                "n_correct": n_correct,
                "n": n,
                "acc": acc,
                "cos_sim_qry": round(cos_sim_qry, 4),
                "cos_sim_tgt": round(cos_sim_tgt, 4),
                "cos_sim_qry_tgt": round(cos_sim, 4),
                "modality_gap": round(modality_gap, 4),
                "preds": preds,
            }
        else:
            res_dict[subset] = acc

    return res_dict
