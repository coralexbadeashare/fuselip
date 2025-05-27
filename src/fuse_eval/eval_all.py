import logging
import os
import json
import traceback
from argparse import Namespace

from config import RES_MMEB_PATH
from fuse_clip.fuse_clip_utils import set_seed, load_model
from fuse_eval.eval_augmented_data import eval_all_augs
from fuse_eval.eval_oi import eval_openimages
from fuse_eval.eval_vg_crop import eval_vg_crop
from mmeb.mmeb_eval import mmeb_eval
from mmeb.mmeb_utils import TxtModifier, PREFIXES_RIGHT

from model_naming import _MODEL_SHORT_NAME_TO_ID
from open_clip_train.data import get_imagenet
from open_clip_train.zero_shot import zero_shot_eval


def eval_imagenet(model, image_processor, txt_tokenizer):
    args = Namespace(
        imagenet_val="/mnt/datasets/imagenet/val/",
        zeroshot_frequency=1,
        epochs=1,
        distributed=False,
        precision="amp",
        device="cuda",
        batch_size=bs,
        workers=num_workers,
    )
    data = get_imagenet(args, preprocess_fns=(None, image_processor), split="val")
    results = zero_shot_eval(
        model=model, data={"imagenet-val": data}, epoch=0, args=args, tokenizer=txt_tokenizer
    )
    results = {
        "imagenet-top1": results['imagenet-zeroshot-val-top1'],
        "imagenet-top5": results['imagenet-zeroshot-val-top5'],
    }
    return results




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    seed = 0
    set_seed(seed)

    model_names = [
        "chs20/FuseLIP-S-CC3M-MM",
        "chs20/FuseLIP-B-CC3M-MM",
        "chs20/FuseLIP-S-CC12M-MM",
        "chs20/FuseLIP-B-CC12M-MM",
    ]

    # take all model_ids from MODEL_ID_TO_SHORT_NAME, if they are not in mmeb-res.json
    # model_ids = [
    #     el for el in MODEL_ID_TO_SHORT_NAME.keys() if el not in json.load(open(RES_MMEB_PATH, "r")).keys()
    # ]
    # model_names = [MODEL_ID_TO_SHORT_NAME[model_id] for model_id in model_ids]

    print(f"evaluating {len(model_names)} models: {model_names}")

    device = "cuda"
    num_workers = 16
    bs = 512
    short = False
    save_to_disk = True
    tasks = None  # = all
    empty_prompt_left = True
    empty_prompt_right = True
    ensemble_prompt = False
    prompt_prefix_right = PREFIXES_RIGHT if not ensemble_prompt else {}

    txt_modifier_left = TxtModifier(model_name="titok", is_train=False, empty_prompt=empty_prompt_left)
    txt_modifier_right = TxtModifier(
        model_name="titok", is_train=False, empty_prompt=empty_prompt_right, prompt_prefixes=prompt_prefix_right
        )

    os.makedirs(os.path.dirname(RES_MMEB_PATH), exist_ok=True)

    failed_models = []
    for model_name in model_names:
        model_id = _MODEL_SHORT_NAME_TO_ID.get(model_name)
        print(f"evaluating model {model_name} ({model_id})")
        try:
            model, image_processor, txt_tokenizer = load_model(model_id, "cuda")
        except Exception as e:
            print(f"failed to load model {model_id}")
            traceback.print_exc()
            failed_models.append(model_id)
            # raise
            continue

        res_dict = {}

        if tasks is None or "imagenet" in tasks:
            print(f"evaluate ImageNet")
            res_dict_imagenet = eval_imagenet(
                model=model, image_processor=image_processor, txt_tokenizer=txt_tokenizer
            )
            res_dict.update(res_dict_imagenet)
            print(f"ImageNet: {res_dict_imagenet}\n")

        if tasks is None or "cc3m-aug" in tasks:
            print(f"evaluate cc3m-aug")
            acc_augs_dict = eval_all_augs(
                model=model, image_processor=image_processor, txt_tokenizer=txt_tokenizer
            )
            res_dict.update(acc_augs_dict)
            print(f"cc3m-aug: {acc_augs_dict}\n")

        if tasks is None or "vg-crop" in tasks:
            print(f"evaluate vg-crop")
            acc_vgcrop = eval_vg_crop(
                model=model, image_processor=image_processor, txt_tokenizer=txt_tokenizer, return_dict=False
            )
            res_dict.update({"vg-crop": acc_vgcrop})
            print(f"vg-crop: {acc_vgcrop}\n")

        if tasks is None or "oi-crop" in tasks:
            print(f"evaluate oi-crop")
            acc_oicrop = eval_openimages(
                model=model, image_processor=image_processor, txt_tokenizer=txt_tokenizer, mask_target=False
            )
            res_dict.update({"oi-crop": acc_oicrop})
            print(f"oi-crop: {acc_oicrop}\n")

        if tasks is None or "oi-pos" in tasks:
            print(f"evaluate oi-pos")
            acc_oipos = eval_openimages(
                model=model, image_processor=image_processor, txt_tokenizer=txt_tokenizer, mask_target=False,
                mode="pos"
            )
            res_dict.update({"oi-pos": acc_oipos})
            print(f"oi-pos: {acc_oipos}\n")

        if tasks is None or "mmeb" in tasks:
            print(f"evaluate on MMEB")
            res_dict_mmeb = mmeb_eval(
                model=model, image_processor=image_processor, short=short, tokenizer=txt_tokenizer,
                batch_size=bs, workers=num_workers, subsets=None,
                txt_modifier_qry=txt_modifier_left, txt_modifier_tgt=txt_modifier_right,
                ensemble_prompt=ensemble_prompt
            )
            res_dict.update(res_dict_mmeb)


        print(res_dict)

        if save_to_disk:
            # check if mmeb-res.json exists
            res_file = RES_MMEB_PATH
            old_res_dict = {}
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    old_res_dict = json.load(f)
            # update subsets that were evaluated
            k = model_id if not ensemble_prompt else f"{model_id}-ENSEMBLE"
            old_res_dict.setdefault(k, {}).update(res_dict)
            with open(res_file, "w") as f:
                json.dump(old_res_dict, f, indent=2)

            print(f"results saved to {res_file}\n")

    print(f"failed models: {failed_models}")
    print("done\n")