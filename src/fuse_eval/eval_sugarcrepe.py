# adapted from https://github.com/RAIVNLab/sugar-crepe/blob/main/main_eval.py
import json
import logging
import os

import torch
from PIL import Image
from tqdm import tqdm

from config import PROJECT_PATH, RES_SUGARCREPE_PATH
from fuse_clip.fuse_clip_utils import set_seed, load_model
from model_naming import MODEL_ID_TO_SHORT_NAME


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, tokenizer, transform, device):
    pos_text = tokenizer(pos_text).to(device)
    pos_text_embedding = model.encode_text(pos_text, normalize=True)
    neg_text = tokenizer(neg_text).to(device)
    neg_text_embedding = model.encode_text(neg_text, normalize=True)
    image_embedding = model.encode_image(transform(image).unsqueeze(dim=0).to(device), normalize=True)
    pos_score = pos_text_embedding @ image_embedding.t()
    neg_score = neg_text_embedding @ image_embedding.t()
    return 1 if pos_score.item() > neg_score.item() else 0


def evaluate_sc(image_root, dataset, model, tokenizer, transform, device):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path)
            correct = text_retrieval(data['caption'], data['negative_caption'], image, model, tokenizer, transform, device)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    set_seed(0)

    model_ids = [
        el for el in MODEL_ID_TO_SHORT_NAME.keys() #if el not in json.load(open(RES_SUGARCREPE_PATH, "r")).keys()
    ]
    save_to_disk = True

    print(f"evaluating {len(model_ids)} models: {model_ids}\n({[MODEL_ID_TO_SHORT_NAME.get(el, el) for el in model_ids]})")

    data_root = os.path.join(PROJECT_PATH, 'data/sugarcrepe')
    coco_image_root = '/mnt/datasets/coco/val2017'
    data_dict = {
        'replace_obj': f'{data_root}/replace_obj.json',
        'replace_att': f'{data_root}/replace_att.json',
        'replace_rel': f'{data_root}/replace_rel.json',
        'swap_obj': f'{data_root}/swap_obj.json',
        'swap_att': f'{data_root}/swap_att.json',
        'add_obj': f'{data_root}/add_obj.json',
        'add_att': f'{data_root}/add_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    for model_id in model_ids:
        print(f"evaluating model {model_id} ({MODEL_ID_TO_SHORT_NAME.get(model_id)})")
        try:
            model, image_processor, txt_tokenizer = load_model(model_id, "cuda")
        except:
            print(f"failed to load model {model_id}")
            continue

        # eval sugarcrepe
        res_dict_sc = evaluate_sc(
            image_root=coco_image_root, dataset=dataset, model=model, tokenizer=txt_tokenizer,
            transform=image_processor, device="cuda"
            )
        print(res_dict_sc)


        if save_to_disk:
            # check if json exists
            res_file_sc = RES_SUGARCREPE_PATH
            old_res_dict = {}
            if os.path.exists(res_file_sc):
                with open(res_file_sc, "r") as f:
                    old_res_dict = json.load(f)
            # update subsets that were evaluated
            old_res_dict.setdefault(model_id, {}).update(res_dict_sc)
            with open(res_file_sc, "w") as f:
                json.dump(old_res_dict, f, indent=2)
            print(f"sc results saved to {res_file_sc}")
        print("done\n")