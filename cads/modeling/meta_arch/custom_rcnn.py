# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor
import json
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.utils.comm as comm

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb, get_fed_loss_cls_weights
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES

from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds
from .ov_mlr import OvMlr
import clip
from centernet.config import add_centernet_config
from detectron2.config import get_cfg
from cads.config import add_detic_config

logger = logging.getLogger(__name__)
@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        *,
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        
        enable_mlr: bool = True,
        ovmlr_model: nn.Module = None,
        
        num_classes: int = 80,
        online_train_mlr: bool = False,
        **kwargs
    ):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            # self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False
                
        self.num_classes = num_classes
        self.enable_mlr = enable_mlr
        if self.enable_mlr:
            self.ovmlr_model = ovmlr_model
            self.online_train_mlr = online_train_mlr

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
            
            "enable_mlr": cfg.MLR.ENABLE_MLR,
            "num_classes" : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
        if ret["enable_mlr"]:
            cfg_mlr = get_cfg()
            add_detic_config(cfg_mlr)
            add_centernet_config(cfg_mlr)
            cfg_mlr.merge_from_file(cfg.MLR.CONFIG_PATH)
            ret["ovmlr_model"] = OvMlr(cfg_mlr)
            ret["online_train_mlr"] = cfg.MLR.ONLINE_TRAIN
        
        return ret
    
    def inference_mlr(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert not self.training
        assert self.enable_mlr
        
        img_id = [batched_input['image_id'] for batched_input in batched_inputs]
        
        if self.input_format == "BGR":
            batched_inputs_mlr = copy.deepcopy(batched_inputs)
            for batched_input in batched_inputs_mlr:
                batched_input["image"] = batched_input["image"][[2,1,0], :, :]
        if not self.online_train_mlr:
            pred_ml_scores = self.ovmlr_model.pred_mlr(batched_inputs_mlr)
        else:
            images = self.preprocess_image(batched_inputs)
            if self.fp16:
                with autocast():
                    features = self.backbone(images.tensor.half())
                features = {k: v.float() for k, v in features.items()}
            else:
                features = self.backbone(images.tensor)
            
            pred_ml_scores = self.ovmlr_model.pred_mlr(batched_inputs_mlr, features)
            
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
        # features = self.backbone(images.tensor)
        
        if self.fp16: 
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)
        
        if self.enable_mlr:
            if not self.online_train_mlr:
                batched_inputs_mlr = copy.deepcopy(batched_inputs)
                if self.input_format == "BGR":
                    for batched_input in batched_inputs_mlr:
                        batched_input["image"] = batched_input["image"][[2,1,0], :, :]
                pred_mlr_scores = self.ovmlr_model.pred_mlr(batched_inputs_mlr)
            else:
                images = self.preprocess_image(batched_inputs)
                pred_mlr_scores = self.ovmlr_model.pred_mlr(batched_inputs, features)
        else:
            pred_mlr_scores = None
        
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, pred_mlr_scores=pred_mlr_scores)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results
        

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if self.with_image_labels:
            for inst, x in zip(gt_instances, batched_inputs):
                inst._ann_type = x['ann_type']
                inst._pos_category_ids = x['pos_category_ids']
            ann_types = [x['ann_type'] for x in batched_inputs]
            assert len(set(ann_types)) == 1
            ann_type = ann_types[0]
            if ann_type in ['prop', 'proptag']:
                for t in gt_instances:
                    t.gt_classes *= 0
        
        if self.fp16: # TODO (zhouxy): improve
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)
            
        losses = {}
        if self.online_train_mlr:
            losses_mlr = self.ovmlr_model.get_mlr_losses(batched_inputs, features)

        cls_features, cls_inds, caption_features = None, None, None

        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()
        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))
        
        if self.dynamic_classifier and ann_type != 'caption':
            cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
            ind_with_bg = cls_inds[0].tolist() + [-1]
            cls_features = self.roi_heads.box_predictor[
                0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

        classifier_info = cls_features, cls_inds, caption_features
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)

        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances)
        else:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                ann_type=ann_type, classifier_info=classifier_info)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        if self.with_image_labels:
            if ann_type in ['box', 'prop', 'proptag']:
                losses.update(proposal_losses)
            else: # ignore proposal loss for non-bbox data
                losses.update({k: v * 0 for k, v in proposal_losses.items()})
        else:
            losses.update(proposal_losses)
        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
                
        if self.online_train_mlr:
            losses.update(losses_mlr)
        
        if self.return_proposal:
            return proposals, losses
        else:
            return losses


    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32, 
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
                if has_caption_feature else None # (NB) x (D + 1)
        return caption_features


    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C, 
            weight=freq_weight)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)
        return inds, cls_id_map
    
