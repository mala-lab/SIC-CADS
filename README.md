# Simple Image-level Classification Improves Open-vocabulary Object Detection *([arXiv 2312.10439](http://arxiv.org/abs/2312.10439))* 
## Installation
Our environment: Unbuntu18.04, cuda10.2, python3.8, pytorch1.12.0, detectron2.0.6

```bash
conda create -n cads python=3.8
conda activate cads
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e . 
cd .. 
git clone https://github.com/openai/CLIP.git
cd CLIP 
pip install -e .
cd .. 
pip install -r requirements.txt
```

## Dataset preparation
Please follow [prepare datasets](datasets/README.md)

## Pretrained Models
Download ImageNet-21K pretrained ResNet-50 from [MIIL](https://github.com/Alibaba-MIIL/ImageNet21K)

convert the pretrained models into d2 style:
```
python tools/convert-thirdparty-pretrained-model-to-d2.py --path models/resnet50_miil_21k.pth
``` 
Download the pretrained [BoxSup](https://dl.fbaipublicfiles.com/detic/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth)\ [Detic](https://dl.fbaipublicfiles.com/detic/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth) for OV-LVIS 
and [Detic](https://dl.fbaipublicfiles.com/detic/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth) for OV-COCO

## Train our MLR module
Train on LVIS-Base
```
python train_net.py --resume --dist-url auto --num-gpus 4 \
    --config configs/ovmlr_lvis_base.yaml OUTPUT_DIR training_dir/ovmlr_lvis_base 
``` 

Train on LVIS-Base + ImageNet-21K
```
python train_net.py --resume --dist-url auto --num-gpus 4 \
    --config configs/ovmlr_lvis_base_in21k.yaml OUTPUT_DIR training_dir/ovmlr_lvis_base_in21k 
``` 

Train on COCO-Base + COCO Caption
```
python train_net.py --resume --dist-url auto --num-gpus 4 \
    --config configs/ovmlr_coco_base_caption.yaml OUTPUT_DIR training_dir/ovmlr_coco_base_caption 
``` 

## CADS for BoxSup and Detic
Merge the weight of MLR model and pretrained OVOD models on OV-LVIS
```
# for Boxsup
python tools/merge_weights.py --weight_path_mlr training_dir/ovmlr_lvis_base/model_final.pth \
    --weight_path_ovod models/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth \
    --output_path models/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x_mlr.pth 

# for Detic
python tools/merge_weights.py --weight_path_mlr training_dir/ovmlr_lvis_base_in21k/model_final.pth  \
    --weight_path_ovod models/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth \
    --output_path models/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size_mlr.pth 
``` 

Evaluation on OV-LVIS
```
python train_net.py --resume --eval-only --dist-url auto --num-gpus 4 \
    --config configs/boxsup_cads_ovlvis.yaml OUTPUT_DIR training_dir/boxsup_lvis_base_cads \
    MODEL.WEIGHTS models/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x_mlr.pth 

python train_net.py --resume --eval-only --dist-url auto --num-gpus 4 \
    --config configs/boxsup_cads_ovlvis.yaml OUTPUT_DIR training_dir/detic_lvis_cads \
    MODEL.WEIGHTS models/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size_mlr.pth 
``` 

Merge the weight of MLR model and pretrained OVOD models on OV-COCO
```
python tools/merge_weights.py --weight_path_mlr training_dir/ovmlr_coco_base_caption/model_final.pth \
    --weight_path_ovod models/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth \
    --output_path models/Detic_OVCOCO_CLIP_R50_1x_max-size_caption_mlr.pth 
``` 
Evaluation on OV-COCO
```
python train_net.py --resume --eval-only --dist-url auto --num-gpus 4 \
    --config configs/detic_cads_ovcoco.yaml OUTPUT_DIR training_dir/detic_ovcoco_cads \
    MODEL.WEIGHTS models/Detic_OVCOCO_CLIP_R50_1x_max-size_caption_mlr.pth 
``` 

## Acknowledgment

Our code is mainly based on [Detic](https://github.com/facebookresearch/Detic) and [Detectron2](https://github.com/facebookresearch/detectron2).

## Citation
```bibtex
@inproceedings{fang2023simple,
  title={Simple Image-level Classification Improves Open-vocabulary Object Detection},
  author={Fang, Ruohuan and Pang, Guansong and Bai, Xiao},
  booktitle={The 38th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
