import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from fuse_clip.fuse_clip_utils import compute_cos_sim_matrix, decode_tokens, set_seed, load_model
from fuse_data.visual_genome import VisualGenome
from model_naming import MODEL_ID_TO_SHORT_NAME
from open_clip_train.logger import setup_logging

_VG_CROP_IOU_THRESH = 0.3

@torch.inference_mode()
def eval_vg_crop(model, image_processor, txt_tokenizer, vlm2vec_processor=None, 
                 do_plot=False, plot_hist=False, return_dict=False):
    """
    Evaluates visual grounding on cropped images.

    Args:
        model: The model to evaluate
        image_processor: Image preprocessing function
        txt_tokenizer: Text tokenizer
        vlm2vec_processor: VLM2Vec processor tuple (processor, collator) if using VLM2Vec model
        do_plot: Whether to generate visualization plots
        plot_hist: Whether to plot histograms of similarities
        return_dict: Whether to return detailed results dictionary
    """
    if vlm2vec_processor is not None:
        from vlm2vec_new.eval import batch_to_device
        processor, collator = vlm2vec_processor
        image_processor = lambda x: x
        txt_tokenizer = None

    ds = VisualGenome(
        txt_tokenizer=txt_tokenizer,
        image_processor=image_processor,
        mode="crop",
        do_tokenize=not bool(vlm2vec_processor),
        is_train=False,
        iou_thresh=_VG_CROP_IOU_THRESH,
        prune_small=True
    )

    same_image_idcs = []
    cur = []
    img_idx = ds.regions[0]["image_id"]  # starts with 1
    for idx, region in enumerate(ds.regions):
        if region["image_id"] == img_idx:
            cur.append(idx)
        else:
            same_image_idcs.append(cur)
            cur = [idx]
            img_idx = region["image_id"]

    avg_regions_per_image = np.mean([len(el) for el in same_image_idcs])
    logging.info(f"avg regions per image: {avg_regions_per_image:.2f}")

    n_samples_plot = 50
    if do_plot:
        fig, axs = plt.subplots(n_samples_plot, 10, figsize=(18, 2.5 * n_samples_plot))
        n_plotted = 0

    n_correct = 0
    n_total = 0
    all_preds = []
    all_labels = []

    for same_img_idcs_cur in tqdm(same_image_idcs[:100]):
        if vlm2vec_processor is not None:
            # VLM2Vec processing with explicit batch processing
            left_features = []
            right_features = []
            batch_size = 16  # Same batch size as original
            
            for same_img_idcs_batch in [same_img_idcs_cur[i:i + batch_size] for i in range(0, len(same_img_idcs_cur), batch_size)]:
                els = [ds[idx] for idx in same_img_idcs_batch]
                txts_left, images_left, txts_right, images_right = zip(*els)
                txts_left = [
                    f"<|image_1|>\nSelect the portion of the image that follows the language expressions: {txt}" for txt in txts_left
                ]
                assert sum([len(txt) for txt in txts_right]) == 0
                txts_right = [f"<|image_1|>\nRepresent the given cropped image of the object" for _ in txts_right]

                inputs_left = collator(list(zip(txts_left, images_left)))
                inputs_right = collator(list(zip(txts_right, images_right)))
                inputs_left = batch_to_device(inputs_left, device="cuda")
                inputs_right = batch_to_device(inputs_right, device="cuda")

                # Immediately move results to CPU to free GPU memory
                left_features_batch = model(qry=inputs_left)["qry_reps"].cpu()
                right_features_batch = model(tgt=inputs_right)["tgt_reps"].cpu()
                left_features.append(left_features_batch)
                right_features.append(right_features_batch)
                
            left_features = torch.cat(left_features)
            right_features = torch.cat(right_features)
            
        else:
            # Standard processing
            els = [ds[idx] for idx in same_img_idcs_cur]
            txts_left, images_left, txts_right, images_right = zip(*els)
            txts_left = torch.stack(txts_left).cuda()
            images_left = torch.stack(images_left).cuda()
            txts_right = torch.stack(txts_right).cuda()
            images_right = torch.stack(images_right).cuda()
            
            with torch.inference_mode():
                out = model(
                    image=(images_left, images_right),
                    text=(txts_left, txts_right)
                )
                left_features, right_features = out["image_features"], out["text_features"]

        # Compute similarities on CPU
        sims = compute_cos_sim_matrix(left_features, right_features)
        preds = sims.argmax(dim=-1).detach().cpu()
        labels = torch.arange(len(preds))
        
        if not vlm2vec_processor:
            all_preds.extend((preds + n_total).tolist())
            all_labels.extend((labels + n_total).tolist())
            
        n_correct += (preds == labels).sum().item()
        n_total += len(preds)

        # plotting
        if do_plot and n_plotted < n_samples_plot:
            import textwrap
            argsort = sims.argsort(dim=-1, descending=True)
            sims_sorted = sims.gather(dim=-1, index=argsort)
            # plot 3 from each batch
            for i_sample in range(3):
                i_total = same_img_idcs_cur[i_sample]
                if n_plotted >= n_samples_plot:
                    break
                ax_row = axs[n_plotted]
                ax_row[0].imshow(images_left[i_sample].detach().cpu().permute(1, 2, 0))
                ax_row[0].set_title(
                    f"[{i_total}] " + textwrap.fill(decode_tokens(txt_tokenizer, txts_left[i_sample]), 30), fontsize=8
                )
                ax_row[1].imshow(images_right[i_sample].detach().cpu().permute(1, 2, 0))
                ax_row[1].set_title(f"[target] {sims[i_sample, i_sample]:.4f}")
                # get the right images with highest cos-sim
                images_right_sorted = images_right[argsort[i_sample], ...]
                for i_other in range(2, 10):
                    ax_row[i_other].imshow(images_right_sorted[i_other - 2].detach().cpu().permute(1, 2, 0))
                    ax_row[i_other].set_title(f"{sims_sorted[i_sample, i_other - 2]:.4f}")
                [ax.axis("off") for ax in ax_row]
                n_plotted += 1
            fig.suptitle(f"{MODEL_ID_TO_SHORT_NAME.get(model_id)}\n")
        if do_plot and n_plotted >= n_samples_plot:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.show()
            # alternatively save the plot
            fig.savefig(os.path.join(RES_PATH, "plots", f"vg-crop-{model_name.replace('/', '-')}.png"))
            do_plot = False

        if plot_hist:
            import textwrap
            fig, axs = plt.subplots(10, 2, figsize=(6, 20))
            for i_sample in range(10):
                ax_row = axs[i_sample]
                ax_row[0].imshow(images_left[i_sample].detach().cpu().permute(1, 2, 0))
                ax_row[0].axis("off")
                ax_row[0].set_title(
                    textwrap.fill(decode_tokens(txt_tokenizer, txts_left[i_sample]), 30), fontsize=8
                )
                ax_row[1].hist(sims[i_sample].detach().cpu().numpy(), bins=20)
                ax_row[1].axvline(sims[i_sample, i_sample].item(), color="red")
                ax_row[1].set_xlim(0, 0.6)
                ax_row[1].set_title(f"similarities")
            fig.suptitle(f"{MODEL_ID_TO_SHORT_NAME.get(model_id)}\n")
            plt.tight_layout()
            plt.show()
            plot_hist = False


    acc = n_correct / n_total
    if return_dict:
        result = {
            "n_correct": n_correct,
            "n": n_total,
            "acc": acc,
        }
        if not vlm2vec_processor:
            result.update({
                "preds": all_preds,
                "labels": all_labels
            })
        return result
    else:
        return acc



if __name__ == '__main__':
    import os
    import json
    from config import RES_PATH
    from model_naming import _MODEL_SHORT_NAME_TO_ID

    setup_logging(log_file=None, level="INFO")
    # set seeds
    seed = 0
    set_seed(seed)

    model_names = [
        "tiger-lab-vlm2vec-full"
    ]
    model_ids = [_MODEL_SHORT_NAME_TO_ID.get(model_name, model_name) for model_name in model_names]

    for model_id in model_ids:
        model_name = MODEL_ID_TO_SHORT_NAME.get(model_id, model_id)
        logging.info(f"evaluating model {model_id} ({model_name})")
        model, image_processor, txt_tokenizer = load_model(model_id, "cuda")

        if "vlm2vec" not in model_name:
            res_dict = eval_vg_crop(
                model=model, image_processor=image_processor, txt_tokenizer=txt_tokenizer,
                do_plot=True,
                plot_hist=False, return_dict=True
            )
        else:
            res_dict = eval_vg_crop(
                model=model, image_processor=None, txt_tokenizer=None,
                vlm2vec_processor=image_processor,
                do_plot=False, plot_hist=False, return_dict=True
            )
            
        logging.info(f"acc: {res_dict['acc']*100:.2f}")

        # save results
        model_name_clean = model_name.replace("/", "-").replace(" ", "-")
        res_file = os.path.join(RES_PATH, model_name_clean, f"preds_vg-crop.json")
        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        with open(res_file, "w") as f:
            json.dump(res_dict, f, indent=2)


    logging.info("done")