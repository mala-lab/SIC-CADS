import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.layers import move_device_like
from torch.cuda.amp import autocast
import clip
import json


logger = logging.getLogger(__name__)
@META_ARCH_REGISTRY.register()
class OvMlr(nn.Module):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        *,
        with_image_labels = False,
        backbone: Backbone,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        fp16 = False,
        
        enable_mlr: bool = True,
        mlr_embedding_path: str = "", 
        enable_image_kd: bool = True,
        use_clip_inference: bool = True, 
        normed_score: bool = True,
        
        num_classes: int,
        mlr_loss_weight: float,
        img_kd_loss_weight: float,
        normed_embedding: bool,
        logit_scale: float, 
        clip_version: str, 
        global_dim: int,
        dataset: str, 
        clip_img_size: int,
        ignore_novel: bool,
        mlr_multi_scale: bool, 
        mlr_levels: int,
        cross_dataset_eval: bool, 
        online_train_mlr: bool,
        lambda_novel: float,
        lambda_base: float, 
        inference_resolution: int, 
        **kwargs
    ):
        """
        """
        super().__init__()
        self.backbone = backbone
        self.input_format = input_format
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.fp16 = fp16
        self.with_image_labels = with_image_labels
        
        self.num_classes = num_classes
        if dataset == "LVIS":
            self.cat_freqs = []
            self.cat_names = []
            self.novel_cats = []
            self.base_cats = []
            for c in LVIS_CATEGORIES:
                self.cat_freqs.append(c['frequency'])
                name = str(c["name"]).replace("_", " ")
                self.cat_names.append(name)
                if c['frequency'] == "r":
                    self.novel_cats.append(c["id"]-1)
                else:
                    self.base_cats.append(c["id"]-1)
            self.base_cat_map = torch.zeros([self.num_classes]).long().cpu()
            self.base_cat_map[self.base_cats] = torch.arange(0, len(self.base_cats), 1).long().cpu()
        elif dataset == "COCO":
            self.novel_cats = []
            self.base_cats = []
            self.ignore_cats = [9, 10, 11, 12, 32, 34, 35, 38, 40, 52, 58, 60, 67, 77, 78]
            cat_info = json.load(open("datasets/coco/zero-shot/instances_train2017_seen_2_oriorder_cat_info.json"))
            for id, info in enumerate(cat_info):
                if info["image_count"] == 0:
                    self.novel_cats.append(id)
                else:
                    self.base_cats.append(id)
            self.base_cat_map = torch.zeros([self.num_classes]).long().cpu()
            self.base_cat_map[self.base_cats] = torch.arange(0, len(self.base_cats), 1).long().cpu()
            
        self.enable_mlr = enable_mlr
        self.ignore_novel = ignore_novel
        self.multi_scale = mlr_multi_scale
        if self.multi_scale:
            self.mlr_levels = mlr_levels
            self.patches = []
            self.num_patches_levels = []
            self.num_pathes = 0
            self.gmp_levels = []
            for level in range(self.mlr_levels):
                self.num_patches_levels.append((level + 1)**2)
                self.num_pathes = self.num_pathes + (level + 1)**2
                for i in range(level + 1):
                    for j in range(level + 1):
                        patch = [i/(level + 2), (i + 2)/(level + 2), j/(level + 2), (j + 2)/(level + 2)]
                        self.patches.append(patch)
                setattr(self, "gmp_level_"+str(level+1), torch.nn.AdaptiveMaxPool2d((level + 1, level + 1)))
                self.gmp_levels.append(getattr(self, "gmp_level_"+str(level+1)))
        else:
            self.gmp = torch.nn.AdaptiveMaxPool2d(1)
            
        self.enable_image_kd = enable_image_kd
        self.cross_dataset_eval = cross_dataset_eval
        self.mlr_loss_weight = mlr_loss_weight
        self.kd_loss_weight = img_kd_loss_weight
        pixel_mean_clip = [0.48145466, 0.4578275, 0.40821073]
        pixel_std_clip = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("pixel_mean_clip", torch.tensor(pixel_mean_clip).view(-1, 1, 1).cpu(), False)
        self.register_buffer("pixel_std_clip", torch.tensor(pixel_std_clip).view(-1, 1, 1).cpu(), False)
        self.clip_resize = Resize([clip_img_size, clip_img_size])
        self.anno_type = "box"
        self.normed_embedding = normed_embedding

        logger.info("Using CLIP Classifiers for MLR.")
        mlr_embeddings = torch.load(mlr_embedding_path, map_location='cpu')

        logger.info("mlr embedding file: " + mlr_embedding_path + ", shape:"+str(mlr_embeddings.shape))
        if self.normed_embedding:
            mlr_embeddings = F.normalize(mlr_embeddings.float(), p=2, dim=1)
        self.logit_scale = logit_scale
        embedding_dim = mlr_embeddings.shape[1]
        
        self.mlr_lin = torch.nn.Linear(global_dim, embedding_dim)
        self.mlr_lin2 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.text_embedding = torch.nn.Linear(embedding_dim, self.num_classes, bias=False)
        self.normed_score = normed_score
        self.use_clip_inference = use_clip_inference
        self.online_train_mlr = online_train_mlr
        self.lambda_novel = lambda_novel
        self.lambda_base = lambda_base
        
        with torch.no_grad():
            self.text_embedding.weight.copy_(mlr_embeddings)
        for param in self.text_embedding.parameters():
            param.requires_grad = False
        
        self.image_embedding = torch.nn.Linear(embedding_dim, embedding_dim)
        clip_model, pre_clip = clip.load(clip_version, device="cpu")
        logger.info("using clip " + clip_version)
        self.clip_model = clip_model.visual
        self.pre_clip = pre_clip
        self.clip_dtype = clip_model.dtype
        self.inference_resolution = inference_resolution
        self.inference_resize = Resize([inference_resolution, inference_resolution])
        for param in self.clip_model.parameters():
            param.requires_grad = False
            

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        if cfg.MLR.ONLINE_TRAIN:
            backbone = None
        else:
            backbone = build_backbone(cfg)
            
        ret.update({
            "backbone"                  : backbone,
            'with_image_labels'         : cfg.WITH_IMAGE_LABELS,
            "pixel_mean"                : cfg.MODEL.PIXEL_MEAN,
            "pixel_std"                 : cfg.MODEL.PIXEL_STD,
            "input_format"              : cfg.INPUT.FORMAT,
            'fp16'                      : cfg.FP16,
            "enable_mlr"                : cfg.MLR.ENABLE_MLR,
            "mlr_embedding_path"        : cfg.MLR.EMBEDDING_PATH, 
            "enable_image_kd"           : cfg.MLR.ENABLE_IMAGE_KD, 
            "num_classes"               : cfg.MLR.NUM_CLASSES,  
            "mlr_loss_weight"           : cfg.MLR.MLR_LOSS_WEIGHT, 
            "img_kd_loss_weight"        : cfg.MLR.IMG_KD_LOSS_WEIGHT, 
            "normed_embedding"          : cfg.MLR.NORMED_EMBEDDING,  
            "logit_scale"               : cfg.MLR.LOGIT_SCALE,  
            "clip_version"              : cfg.MLR.CLIP_VERSION, 
            "global_dim"                : cfg.MLR.GLOBAL_DIM, 
            "dataset"                   : cfg.MLR.DATASET, 
            "clip_img_size"             : cfg.MLR.CLIP_IMG_SIZE, 
            "ignore_novel"              : cfg.MLR.IGNORE_NOVEL, 
            "mlr_multi_scale"           : cfg.MLR.MULTI_SCALE, 
            "mlr_levels"                : cfg.MLR.MULTI_SCALE_LEVELS,
            "cross_dataset_eval"        : cfg.MLR.CROSS_DATASET_EVAL, 
            "online_train_mlr"          : cfg.MLR.ONLINE_TRAIN, 
            "use_clip_inference"        : cfg.MLR.CLIP_IE_INFERENCE, 
            "normed_score"              : cfg.MLR.NORMED_SCORE, 
            "lambda_novel"              : cfg.MLR.LAMBDA_NOVEL,
            "lambda_base"               : cfg.MLR.LAMBDA_BASE, 
            "inference_resolution"      : cfg.MLR.INFERENCE_RESOLUTION, 
        })

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
    
    
    def pred_mlr(self, batched_inputs, features = None):
        
        if not self.online_train_mlr:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            # print(images.tensor.shape)
            
        img_features, text_features = self.extract_global_feature(features)
        
        if self.use_clip_inference:
            imgs_clip = self.preprocess_image_clip(batched_inputs)
            clip_img_feature = self.clip_model(imgs_clip)
            clip_img_feature = F.normalize(clip_img_feature, p=2, dim=-1)
            
            if self.multi_scale:
                img_mlr_scores = self.get_multi_level_scores(clip_img_feature)
            else:
                img_mlr_scores = self.text_embedding(clip_img_feature)
        else:
            if not self.multi_scale:
                img_mlr_scores = self.text_embedding(img_features)
            else:
                img_mlr_scores = self.get_multi_level_scores(img_features)
            
        if not self.multi_scale:
            text_mlr_scores = self.text_embedding(text_features)
        else:
            text_mlr_scores = self.get_multi_level_scores(text_features)
        
        if self.normed_score:
            img_mlr_scores = (img_mlr_scores - img_mlr_scores.mean())/img_mlr_scores.std()
            img_mlr_scores = img_mlr_scores.sigmoid()
            
            text_mlr_scores = (text_mlr_scores - text_mlr_scores.mean())/text_mlr_scores.std()
            text_mlr_scores = text_mlr_scores.sigmoid()
                
        if not self.cross_dataset_eval:
            l_b = self.lambda_base
            pred_mlr_scores = (((text_mlr_scores)**(l_b)) * ((img_mlr_scores)**(1 - l_b)))
            l_n = self.lambda_novel
            pred_mlr_scores_novel =  (((text_mlr_scores)**(1 - l_n)) * ((img_mlr_scores)**(l_n)))
            pred_mlr_scores[:, self.novel_cats] = pred_mlr_scores_novel[:, self.novel_cats]
        else:
            lamda = 0.2
            pred_mlr_scores = (((text_mlr_scores)**(1-lamda)) * ((img_mlr_scores)**(lamda)))
        
        return pred_mlr_scores
    
    def get_multi_level_scores(self, multi_level_embeddings):
        
        mlr_scores = self.text_embedding(multi_level_embeddings)
        mlr_scores = mlr_scores.reshape([-1, self.num_pathes, self.num_classes])
        mlr_scores = mlr_scores.split(self.num_patches_levels, dim = 1)
        mlr_scores_levels = []
        for level, mlr_scores_per_level in enumerate(mlr_scores) :
            mlr_scores_per_level = mlr_scores_per_level.max(dim=1, keepdim=True)[0]
            mlr_scores_levels.append(mlr_scores_per_level)
            
        mlr_scores = torch.cat(mlr_scores_levels, dim = 1)
        mlr_scores = mlr_scores.mean(dim = 1)
        
        return mlr_scores
    
    def inference_mlr(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert not self.training
        assert self.enable_mlr
        
        img_id = [batched_input['image_id'] for batched_input in batched_inputs]
        pred_ml_scores = self.pred_mlr(batched_inputs)
        results = []
        for i in range(pred_ml_scores.shape[0]):
            top10_scores, top10_idxes = torch.topk(pred_ml_scores[i], k=10, dim=0)
            top20_scores, top20_idxes = torch.topk(pred_ml_scores[i], k=20, dim=0)
            result = {}
            result["top10_scores"] = top10_scores.cpu().numpy()
            result["top20_scores"] = top20_scores.cpu().numpy()
            result["top10_idxes"] = top10_idxes.cpu().numpy()
            result["top20_idxes"] = top20_idxes.cpu().numpy()
            result["logits"] = pred_ml_scores[i].cpu()
            result["img_id"] = img_id[i]
            results.append(result)
            
        return results

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None
        
        images = self.preprocess_image(batched_inputs)
        
        features = self.backbone(images.tensor)
        pred_mlr_scores = self.pred_mlr(features, batched_inputs)

        return pred_mlr_scores
        
    def extract_global_feature(self, feature_maps):
        features = []
        if not self.multi_scale:
            for name, feature in feature_maps.items():
                feature = self.gmp(feature)
                features.append(feature)
            features = torch.cat(features, dim = 1).flatten(start_dim=1)
        else:
            for level in range(self.mlr_levels):
                features_per_level = []
                for name, feature in feature_maps.items():
                    feature = self.gmp_levels[level](feature).flatten(start_dim=2).permute([0,2,1])
                    features_per_level.append(feature)
                features_per_level = torch.cat(features_per_level, dim = 2)
                features.append(features_per_level)
            features = torch.cat(features, dim = 1)
        
        global_features = self.mlr_lin(features)
        
        text_features = self.mlr_lin2(global_features)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        img_features = self.image_embedding(global_features)
        img_features = F.normalize(img_features, p=2, dim=-1)
        
        return img_features, text_features


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        
        if self.fp16:
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)
            
        losses = {}
        img_features, text_features = self.extract_global_feature(features)
        if not self.multi_scale:
            pred_ml_scores = self.logit_scale * self.text_embedding(text_features)
        else:
            pred_ml_scores = self.logit_scale * self.get_multi_level_scores(text_features)
        
        mlr_loss = self.get_rank_loss(pred_ml_scores, batched_inputs)
        losses["mlr_loss"] = self.mlr_loss_weight * mlr_loss
        
        if self.enable_image_kd:
            imgs_clip = self.preprocess_image_clip(batched_inputs)
            with autocast():
                global_embeddings = self.clip_model(imgs_clip)
            del imgs_clip
            global_embeddings = F.normalize(global_embeddings, p=2, dim=-1)
            
            img_features = img_features.reshape([-1, img_features.shape[-1]])
            img_kd_loss = torch.abs(img_features - global_embeddings).sum()/(global_embeddings.shape[0])
            losses["img_kd_loss"] = self.kd_loss_weight * img_kd_loss
            
        return losses
    
    def get_mlr_losses(self, batched_inputs, features):
        
        losses = {}
        img_features, text_features = self.extract_global_feature(features)
        if not self.multi_scale:
            pred_ml_scores = self.logit_scale * self.text_embedding(text_features)
        else:
            pred_ml_scores = self.logit_scale * self.get_multi_level_scores(text_features)
        
        mlr_loss = self.get_rank_loss(pred_ml_scores, batched_inputs)
        losses["mlr_loss"] = self.mlr_loss_weight * mlr_loss
        
        if self.enable_image_kd:
            imgs_clip = self.preprocess_image_clip(batched_inputs)
            with autocast():
                global_embeddings = self.clip_model(imgs_clip)
            del imgs_clip
            global_embeddings = F.normalize(global_embeddings, p=2, dim=-1)
            
            img_features = img_features.reshape([-1, img_features.shape[-1]])
            img_kd_loss = torch.abs(img_features - global_embeddings).sum()/(global_embeddings.shape[0])
            losses["img_kd_loss"] = self.kd_loss_weight * img_kd_loss
            
        return losses

    def preprocess_image_clip(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        if "clip_image" in batched_inputs[0]:
            images = [self._move_to_current_device(x["clip_image"]) for x in batched_inputs]
        else:
            images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
            
        if self.multi_scale:
            images_multi_scale = []
            for img in images:
                _, h, w = img.shape
                for patch in self.patches:
                    x1, x2, y1, y2 = patch
                    img_patch = img[:, int(x1 * h):int(x2 * h), int(y1 * w):int(y2 * w)]
                    images_multi_scale.append(img_patch)
            images = images_multi_scale
                
        images = [self.clip_resize(x).to(self.clip_dtype) for x in images]
        images = [(x - 0.0)/255.0 for x in images]
        assert (self.input_format == "BGR") or (self.input_format == "RGB")
        if self.input_format == "BGR":
            images = [x[[2,1,0],:,:] for x in images]
        
        images = [(x - self.pixel_mean_clip) / self.pixel_std_clip for x in images]
        images = [x.unsqueeze(0) for x in images]
        images = torch.cat(images, dim = 0)

        return images
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        if not self.training:
            scale = self.inference_resolution / min(images[0].shape[1], images[0].shape[2])
            images = [F.interpolate(x.unsqueeze(dim = 0), scale_factor=scale, recompute_scale_factor=True).squeeze(dim = 0) for x in images]
            # images = [self.inference_resize(x) for x in images]
        
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images
    

    def get_rank_loss(self, pred_ml_scores, batched_inputs):
        
        gt_classes_imgs = []
        if self.with_image_labels:
            anno_type = batched_inputs[0]['ann_type']
        else:
            anno_type = "box"
        
        if anno_type == "box":
            for x in batched_inputs:
                gt_instance = x["instances"]
                gt_classes = gt_instance.gt_classes.cpu().numpy()
                gt_classes = np.unique(gt_classes)
                if self.ignore_novel:
                    gt_classes = self.base_cat_map[gt_classes]
                gt_classes_imgs.append(gt_classes)
        else:
            for x in batched_inputs:
                pos_cat_ids = x['pos_category_ids']
                gt_classes_imgs.append(pos_cat_ids)
        
        rank_loss = torch.tensor(0.0).to(self.device)
        for ml_scores_per_img, gt_classes_per_img in zip(pred_ml_scores, gt_classes_imgs):
            
            num_gt_classes = len(gt_classes_per_img)
            if num_gt_classes == 0:
                continue
            if anno_type == "box" and self.ignore_novel:
                ml_scores_per_img = ml_scores_per_img[self.base_cats]
            len_ml_scores = ml_scores_per_img.shape[0]
            
            ml_scores_per_img_neg = ml_scores_per_img.expand(num_gt_classes, len_ml_scores)
            ml_scores_per_img_pos = ml_scores_per_img[gt_classes_per_img]
            ml_scores_per_img_pos = ml_scores_per_img_pos.expand(len_ml_scores, num_gt_classes)
            ml_scores_per_img_pos = ml_scores_per_img_pos.permute(1, 0)
            neg_minus_pos = ml_scores_per_img_neg - ml_scores_per_img_pos + 1
            neg_minus_pos = neg_minus_pos.clamp(min = 0)
            rank_loss_per_img = neg_minus_pos.mean()
            rank_loss = rank_loss + rank_loss_per_img
            
        if len(gt_classes_imgs) != 0:
            rank_loss = rank_loss/len(gt_classes_imgs)
        
        return rank_loss
    
