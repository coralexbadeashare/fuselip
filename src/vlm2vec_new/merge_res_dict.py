import os
import json




if __name__ == '__main__':

    model_name = "tiger-lab-vlm2vec-full"

    my_res_dict_path = "/mnt/cschlarmann37/project_fuse-clip/results/mmeb-res.json"
    vlm2vec_res_dir = f"/mnt/cschlarmann37/project_fuse-clip/results-vlm2vec/{model_name}/"

    my_res_dict = json.load(open(my_res_dict_path, "r"))
    my_res_dict[model_name] = {}

    for el in os.listdir(vlm2vec_res_dir):
        if el.endswith(".json"):
            ds_name = el.split(".")[0].replace("_score", "")
            vlm2vec_res_dict = json.load(open(os.path.join(vlm2vec_res_dir, el), "r"))
            score = vlm2vec_res_dict["acc"]
            my_res_dict[model_name][ds_name] = score

    print(my_res_dict)
    with open(my_res_dict_path, "w") as f:
        json.dump(my_res_dict, f, indent=2)



