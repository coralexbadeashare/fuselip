import logging

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from fuse_clip.fuse_clip_utils import set_seed, load_model, decode_tokens
from fuse_data.oi_data import OpenImages
from model_naming import MODEL_ID_TO_SHORT_NAME
from open_clip_train.logger import setup_logging
from vlm2vec_new.eval import batch_to_device



@torch.inference_mode()
def eval_openimages(model, image_processor, txt_tokenizer, mask_target, mode="crop",
                    do_plot=False, return_dict=False, vlm2vec_processor=None):
    """
    Evaluates model on OpenImages dataset.
    
    Args:
        model: The model to evaluate
        image_processor: Image preprocessing function
        txt_tokenizer: Text tokenizer
        mask_target: Whether to mask target
        mode: Evaluation mode - "crop", "pos", or "vqa"
        do_plot: Whether to generate visualization plots
        return_dict: Whether to return detailed results dictionary
        vlm2vec_processor: VLM2Vec processor tuple (processor, collator) if using VLM2Vec model
    """
    # Use identity function for image processor if using VLM2Vec
    if vlm2vec_processor is not None:
        processor, collator = vlm2vec_processor
        image_processor = lambda x: x
        txt_tokenizer = None

    ds = OpenImages(
        txt_tokenizer=txt_tokenizer, image_processor=image_processor, mask_target=mask_target,
        mode=mode, do_tokenize=not bool(vlm2vec_processor)
    )

    n_samples_plot = 50
    if do_plot:
        n_cols = {"crop": 11, "pos": 5, "vqa": 5}[mode]
        fig, axs = plt.subplots(n_samples_plot, n_cols, figsize=(1.5 * n_cols, 2.5 * n_samples_plot))
        n_plotted = 0

    preds = []
    n_correct = 0
    n_total = 0
    for idx in tqdm(np.random.permutation(range(len(ds)))):
        query_text, ref_img, retrieval_text, retrieval_imgs = ds[idx]
        
        # Process differently based on model type
        if vlm2vec_processor is not None:
            # VLM2Vec processing
            # Format texts for VLM2Vec
            query_text = f"<|image_1|>\nSelect the portion of the image that follows the language expressions: {query_text}"
            retrieval_text = [f"<|image_1|>\nRepresent the given cropped image of the object" for _ in retrieval_text]
            
            # Prepare query inputs
            inputs_qry = collator([[query_text, ref_img]])
            inputs_tgt = collator(list(zip(retrieval_text, retrieval_imgs)))
            inputs_qry = batch_to_device(inputs_qry, device="cuda")
            inputs_tgt = batch_to_device(inputs_tgt, device="cuda")
            
            # Get embeddings
            query_feats = model(qry=inputs_qry)["qry_reps"]
            retrieval_feats = model(tgt=inputs_tgt)["tgt_reps"]
        else:
            # Standard processing for non-VLM2Vec models
            # prepare input
            ref_img = ref_img.unsqueeze(0).cuda()
            retrieval_imgs = torch.stack(retrieval_imgs).cuda() if retrieval_imgs[0] is not None else retrieval_imgs
            query_text = query_text.unsqueeze(0).cuda()
            retrieval_text = torch.stack(retrieval_text).cuda()  # is empty, unless vqa
            
            # get features
            query_feats = model.encode_multimodal(image=ref_img, text=query_text)
            retrieval_feats = model.encode_multimodal(image=retrieval_imgs, text=retrieval_text)
        
        # compute similarity
        sim = F.cosine_similarity(query_feats, retrieval_feats, dim=-1)
        
        # get pred
        pred = sim.argmax().item()
        preds.append({
            "idx": int(idx),
            "pred": pred,
            "label": ds.data[idx].get("label"),
            "query_text": query_text if vlm2vec_processor else decode_tokens(txt_tokenizer, query_text),
            "query_img": ds.data[idx]["query_img"],
            "retrieval_samples": ds.data[idx]["retrieval_samples"] if mode != "vqa" else None,
            "retrieval_txts": ds.data[idx]["retrieval_txts"] if mode == "vqa" else None,
            "sims": sim.tolist(),
        })
        is_correct = pred == 0
        n_correct += is_correct
        n_total += 1

        # plotting - similar for both model types
        if do_plot and n_plotted < n_samples_plot:
            import textwrap
            ax_row = axs[n_plotted]
            
            # Show the reference image
            if vlm2vec_processor:
                ax_row[0].imshow(ref_img)
            else:
                ax_row[0].imshow(ref_img.squeeze().detach().cpu().permute(1, 2, 0))
                
            # Set title
            query_text_display = query_text if vlm2vec_processor else decode_tokens(txt_tokenizer, query_text)
            ax_row[0].set_title(
                f"[{n_plotted}] " + textwrap.fill(query_text_display, 30), fontsize=9
            )
            
            # Show retrieval images
            for i_crop in range(1, len(retrieval_imgs) + 1):
                if retrieval_imgs[i_crop - 1] is not None:
                    if vlm2vec_processor:
                        ax_row[i_crop].imshow(retrieval_imgs[i_crop - 1])
                    else:
                        ax_row[i_crop].imshow(retrieval_imgs[i_crop - 1].detach().cpu().permute(1, 2, 0))
                axtitle = f"{sim[i_crop - 1]:.4f}"
                if mode == "vqa":
                    retrieval_text_display = retrieval_text[i_crop - 1] if vlm2vec_processor else decode_tokens(txt_tokenizer, retrieval_text[i_crop - 1])
                    axtitle += f"\n\"{retrieval_text_display}\""
                ax_row[i_crop].set_title(axtitle, fontsize=9)
            
            [ax.axis("off") for ax in ax_row]
            
            # put frame around predicted image
            frame_ax = pred + 1
            ax_row[frame_ax].axis("on")
            ax_row[frame_ax].set_xticks([])
            ax_row[frame_ax].set_yticks([])
            for spine in ax_row[frame_ax].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('red')
            n_plotted += 1
            fig.suptitle(f"{MODEL_ID_TO_SHORT_NAME.get(model_id)}\n")
            
        if do_plot and n_plotted >= n_samples_plot:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # save plot
            plt.savefig(
                os.path.join(RES_PATH, "plots", f"oi-crop-{mode}-{model_name.replace('/', '-')}.png")
            )
            do_plot = False

    acc = n_correct / n_total
    if return_dict:
        return {
            "n_correct": n_correct,
            "n": n_total,
            "acc": acc,
            "preds": preds,
        }
    else:
        return acc


if __name__ == '__main__':
    import os
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

    ###
    mode = "pos"  # Can be "crop" or "pos"
    do_plot = True
    ###

    logging.info(f"MODE: {mode}")

    for model_id in model_ids:
        set_seed(seed)  # reset seed, so we can use same order of samples
        model_name = MODEL_ID_TO_SHORT_NAME.get(model_id, model_id)
        logging.info(f"evaluating model {model_id} ({model_name})")

        model, image_processor, txt_tokenizer = load_model(model_id, "cuda")
        
        # Check if using VLM2Vec model
        vlm2vec_processor = None
        if "vlm2vec" in model_name:
            vlm2vec_processor = image_processor # For VLM2Vec, image_processor is (processor, collator)
            image_processor = None
            logging.info("Using VLM2Vec model")

        res_dict = eval_openimages(
            model=model, 
            image_processor=image_processor,
            txt_tokenizer=txt_tokenizer,
            mode=mode,
            mask_target=False,
            return_dict=True,
            do_plot=do_plot,
            vlm2vec_processor=vlm2vec_processor
        )

        logging.info(f"accuracy: {res_dict['acc']*100:.2f}")