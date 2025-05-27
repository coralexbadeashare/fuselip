import logging
import os
import textwrap

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm

from fuse_clip.fuse_clip_utils import set_seed, decode_tokens, load_model
from model_naming import MODEL_ID_TO_SHORT_NAME, _MODEL_SHORT_NAME_TO_ID
from config import PROJECT_PATH, CC3M_VAL_CSV_PATH
from fuse_data.augmented_pretrain_data import AugmentedCsvDataset, _CROP_LOCATIONS
from open_clip_train.logger import setup_logging



def eval_all_augs(model, image_processor, txt_tokenizer, vlm2vec_processor=None):
    acc_aug_dict = {}
    for aug in ["randomcrop", "rotate", "colorjitterv2", "flip", "colorize_grayscale"]:
        acc_cur = eval_augs(model, image_processor, txt_tokenizer, aug, vlm2vec_processor=vlm2vec_processor)
        print(f"acc {aug}: {acc_cur*100:.2f}")
        acc_aug_dict[aug] = acc_cur
    acc_aug_dict["cc3m-aug"] = np.mean(list(acc_aug_dict.values())).item()
    return acc_aug_dict

@torch.inference_mode()
def eval_augs(model, image_processor, txt_tokenizer, aug, do_plot=False, return_dict=False, unpad_image=False, vlm2vec_processor=None):
    """
    Evaluates model on augmented images.
    
    Args:
        model: The model to evaluate
        image_processor: Image preprocessing function
        txt_tokenizer: Text tokenizer
        aug: Augmentation type to evaluate
        do_plot: Whether to generate visualization plots
        return_dict: Whether to return detailed results dictionary
        unpad_image: Whether to remove padding from images
        vlm2vec_processor: VLM2Vec processor tuple (processor, collator) if using VLM2Vec model
    """
    # Use identity function for image processor if using VLM2Vec
    if vlm2vec_processor is not None:
        from vlm2vec_new.eval import batch_to_device
        processor, collator = vlm2vec_processor
        image_processor = None
        txt_tokenizer = None

    ds = AugmentedCsvDataset(
        input_filename=CC3M_VAL_CSV_PATH,
        preprocess=image_processor,
        img_key="path",
        caption_key="caption",
        unimodal_prob=0.0,
        sep=",",
        tokenizer=txt_tokenizer,
        augmentations=[aug],
        eval_mode=True,
        multiaug=True,
        unpad_image=unpad_image,
        do_tokenize=not bool(vlm2vec_processor),
        )

    # get subset
    # with pre-saved idcs (they are for cc3m-val)
    idcs_file = os.path.join(PROJECT_PATH, "src/fuse_eval/eval_augmented_idcs.txt")
    with open(idcs_file, "r") as f:
        idcs = eval(f.read())

    assert len(set(idcs)) == len(idcs)
    logging.info(f"len idcs: {len(idcs)}")
    idcs = idcs[:1000]
    ds.images = [ds.images[i] for i in idcs]
    ds.captions = [ds.captions[i] for i in idcs]

    logging.info(f"len ds: {len(ds)}")

    n_samples_plot = 30
    fig = None

    # eval
    preds = []
    for i_sample in tqdm(range(len(ds))):
        txts_left, images_left, txts_right, images_right = ds[i_sample]
        image_left = images_left[0]
        txt_left = txts_left[0]
        
        if vlm2vec_processor is not None:
            # VLM2Vec processing
            # Format texts for VLM2Vec
            txt_left_str = f"<|image_1|>\nSelect the portion of the image that follows the language expressions: {txt_left}"
            txts_right_str = [f"<|image_1|>\nRepresent the given image" for _ in images_right]
            # images to PIL
            image_left = transforms.ToPILImage()(image_left)
            images_right = [transforms.ToPILImage()(img) for img in images_right]
            
            # Prepare inputs
            inputs_left = collator([[txt_left_str, image_left]])
            inputs_right = collator(list(zip(txts_right_str, images_right)))
            inputs_left = batch_to_device(inputs_left, device="cuda")
            inputs_right = batch_to_device(inputs_right, device="cuda")
            
            # Get embeddings
            left_features = model(qry=inputs_left)["qry_reps"]
            right_features = model(tgt=inputs_right)["tgt_reps"]
            sims = torch.cosine_similarity(left_features, right_features, dim=-1)
            pred = sims.argmax().item()
        else:
            # Standard processing for non-VLM2Vec models
            images_right = torch.stack(images_right)
            txts_right = torch.stack(txts_right)

            embed_left = model(
                image=image_left.unsqueeze(0).cuda(), text=txt_left.unsqueeze(0).cuda(), force_fused=True
            )
            embeds_right = model(image=images_right.cuda(), text=txts_right.cuda(), force_fused=True)
            sims = torch.cosine_similarity(embed_left, embeds_right, dim=-1)
            pred = sims.argmax().item()
        
        preds.append(pred)

        # plotting
        if  do_plot and (i_sample < n_samples_plot) and not vlm2vec_processor:
            if fig is None:
                n_cols = images_right.shape[0]+1
                fig, axs = plt.subplots(n_samples_plot, n_cols, figsize=(1.8*n_cols, 2.5 * n_samples_plot))
            txts_left = [f"{txt_tokenizer.decode(el[el != 0][1:-1].tolist())}" for el in txts_left]
            txts_left = [el.replace("crop to ", "").strip() for el in txts_left]
            if aug == "randomcrop":
                # sort by crop location, they are in random order
                crop_locations = _CROP_LOCATIONS.copy()
                sorted_idcs = [txts_left.index(el) for el in crop_locations]
            else:
                sorted_idcs = range(len(txts_left))
            txts_left = [txts_left[i] for i in sorted_idcs]
            images_right = images_right[sorted_idcs]
            pred_sorted = sorted_idcs.index(pred)
            sims_sorted = sims[sorted_idcs]
            # plot
            axs_cur = axs[i_sample]
            axs_cur[0].imshow(image_left.detach().cpu().permute(1, 2, 0))
            axs_cur[0].set_title(f"[{i_sample}] {textwrap.fill(decode_tokens(txt_tokenizer, txt_left), 30)}", fontsize=8)
            for i, (txt, img) in enumerate(zip(txts_left, images_right)):
                axs_cur[i + 1].imshow(img.detach().cpu().permute(1, 2, 0))
                title = f"{txt}\nsim: {sims_sorted[i]:.3f}"
                if i == pred_sorted:
                    # put frame around predicted image
                    title = "X " + title
                title = textwrap.fill(title, 30)
                axs_cur[i + 1].set_title(title, fontsize=8)
            [ax.axis("off") for ax in axs_cur]
            # put frame around predicted image
            frame_ax = pred_sorted + 1
            axs_cur[frame_ax].axis("on")
            axs_cur[frame_ax].set_xticks([])
            axs_cur[frame_ax].set_yticks([])
            for spine in axs_cur[frame_ax].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('red')
        elif (i_sample == n_samples_plot) and do_plot:
            fig.suptitle(model_name)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


    preds = np.array(preds)
    n_correct = (preds == 0).sum()
    acc = n_correct / len(preds)
    if return_dict:
        return {
            "n_correct": n_correct.item(),
            "n": len(preds),
            "acc": acc.item(),
            "preds": preds.tolist(),
        }
    else:
        return acc.item()


if __name__ == '__main__':
    setup_logging(log_file=None, level="INFO")
    # set seeds
    seed = 0
    set_seed(seed)

    model_names = [
        "tiger-lab-vlm2vec-full",
    ]
    model_ids = [_MODEL_SHORT_NAME_TO_ID.get(model_name, model_name) for model_name in model_names]

    aug = "randomcrop"
    # aug = "colorize_grayscale"
    # aug = "colorjitterv2"
    # aug = "rotate"
    # aug = "flip"
    # aug = "all"

    do_plot = False


    for model_id in model_ids:
        set_seed(seed)
        model_name = MODEL_ID_TO_SHORT_NAME.get(model_id, model_id)
        logging.info(f"evaluating model {model_id} ({model_name})")

        model, image_processor, txt_tokenizer = load_model(model_id, "cuda")
        
        # Check if using VLM2Vec model
        vlm2vec_processor = None
        if "vlm2vec" in model_name:
            vlm2vec_processor = image_processor  # For VLM2Vec, image_processor is (processor, collator)
            image_processor = None
            logging.info("Using VLM2Vec model")

        if aug == "all":
            # Evaluate all augmentations
            res_dict = eval_all_augs(
                model=model,
                image_processor=image_processor,
                txt_tokenizer=txt_tokenizer,
                vlm2vec_processor=vlm2vec_processor
            )
            logging.info(f"accs: {res_dict}")
            logging.info(f"mean acc: {res_dict['cc3m-aug']*100:.2f}")
        else:
            res_dict = eval_augs(
                model=model,
                image_processor=image_processor,
                txt_tokenizer=txt_tokenizer,
                aug=aug,
                return_dict=True,
                unpad_image=False,
                do_plot=do_plot,
                vlm2vec_processor=vlm2vec_processor
            )
            logging.info(f"acc: {res_dict['acc']*100:.2f}")


    logging.info("done")