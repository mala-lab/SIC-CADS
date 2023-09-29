import clip
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
import torch
import json

if __name__ == '__main__':
    manual_cat_prompts = []
    cat_names = []
    for c in LVIS_CATEGORIES:
        name = str(c["name"])
        cat_names.append(name)
        manual_cat_prompts.append("There is a " + name)
    clip_model, pre = clip.load("ViT-B/32", device="cpu")
    text = clip.tokenize(manual_cat_prompts)
    text_features = clip_model.encode_text(text)
    print(text_features.shape)
    torch.save(text_features, "resources/lvis_there_is_a_cls_vitb32.pt")


    cat_info = json.load(open("datasets/coco/zero-shot/instances_train2017_seen_2_oriorder_cat_info.json"))
    manual_cat_prompts = []
    cat_names = []
    for info in cat_info:
        name = info["name"]
        cat_names.append(name)
        manual_cat_prompts.append("There is a " + name)
    print(manual_cat_prompts)
    clip_model, pre = clip.load("ViT-B/32", device="cpu")
    text = clip.tokenize(manual_cat_prompts)
    text_features = clip_model.encode_text(text)
    print(text_features.shape)
    torch.save(text_features, "resources/coco_there_is_a_cls_vitb32.pt")