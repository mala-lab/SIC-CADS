import logging
import weakref
import datetime
import time

import numpy as np
from contextlib import ExitStack, contextmanager
import torch
from detectron2.evaluation.evaluator import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm
from torch import nn
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
from tqdm import tqdm
from lvis import LVIS
import json
from pycocotools.coco import COCO

def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).to(predictions.device)
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]
        # mask = label.abs() == 1
        if (label > 0).sum() == 0:
            empty_class += 1
            ap[idx_cls] = -1.0
            continue
        # binary_label = torch.clamp(label[mask], min=0, max=1)
        sorted_pred, sort_idx = prediction.sort(descending=True)
        sorted_label = label[sort_idx]
        tmp = (sorted_label == 1).double()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).double().cumsum(0)
        num_pos = label.sum().double()
        rec = tp/num_pos
        prec = tp/(tp+fp)
        ap_cls = (tmp*prec).sum()/num_pos
        ap[idx_cls].copy_(ap_cls)
    return 100.0 * ap

def compute_F1(predictions, labels, mode_F1, k_val):
    idx = predictions.topk(dim=1, k=k_val)[1]
    predictions.fill_(0)
    predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val).to(predictions.device))
    mask = predictions == 1
    TP = (labels[mask] == 1).sum().float()
    tpfp = mask.sum().float()
    tpfn = (labels == 1).sum().float()
    p = TP / tpfp
    r = TP/tpfn
    f1 = 2*p*r/(p+r)
    
def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def compute_ap(preds, targs):
    
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = 100.0 * average_precision(scores, targets)
    return ap

def mlr_evaluation(
    model, data_loader, dataset="LVIS"
):

    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    results_per_rank = []
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
            
            start_compute_time = time.perf_counter()
                            
            outputs = model.inference_mlr(inputs)
            results_per_rank = results_per_rank + outputs
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
            
    results_gathered = comm.gather(results_per_rank)
    results = []
    if comm.is_main_process():
        for result in results_gathered:
            results = results + result
            
    if dataset == "LVIS":
        lvis_mlr_eval(results)
    elif dataset == "COCO":
        coco_mlr_eval(results)

        
    return results

def lvis_mlr_eval(results):
    
    lvis_api = LVIS("datasets/lvis/lvis_v1_val.json")
    num_classes = 1203
    rare_cats, common_cats, freq_cats = [], [], []
    
    for c in LVIS_CATEGORIES:
        if c['frequency'] == "r":
            rare_cats.append(c["id"]-1)
        elif c['frequency'] == "c":
            common_cats.append(c["id"]-1)
        elif c['frequency'] == "f":
            freq_cats.append(c["id"]-1)
    
    if comm.is_main_process():
        logger = logging.getLogger(__name__)
        tp_top10, tpfp_top10, tpfn_top10 = 0, 0, 0 
        tp_top20, tpfp_top20, tpfn_top20 = 0, 0, 0
        
        tp_top10_r, tpfp_top10_r, tpfn_top10_r = 0, 0, 0 
        tp_top20_r, tpfp_top20_r, tpfn_top20_r = 0, 0, 0
        for result in  tqdm(results, ncols=90):
            img_id = int(result["img_id"])
            annos = lvis_api.img_ann_map[img_id]
            gt_classes = []
            gt_classes_r = []
            for anno in annos:
                if anno['category_id'] - 1 not in gt_classes:
                    gt_classes.append(anno['category_id'] - 1)
                    if anno['category_id'] - 1 in rare_cats:
                        gt_classes_r.append(anno['category_id'] - 1)
            
            tpfn_top10 = tpfn_top10 + len(gt_classes)
            tpfp_top10 = tpfp_top10 + 10
            
            tpfn_top10_r = tpfn_top10_r + len(gt_classes_r)
            for cat in result['top10_idxes']:
                if cat in rare_cats:
                    tpfp_top10_r = tpfp_top10_r + 1
                    if cat in gt_classes:
                        tp_top10_r = tp_top10_r + 1
                if cat in gt_classes:
                    tp_top10 = tp_top10 + 1
                    
            tpfn_top20 = tpfn_top20 + len(gt_classes)
            tpfp_top20 = tpfp_top20 + 20
            tpfn_top20_r = tpfn_top20_r + len(gt_classes_r)
            for cat in result['top20_idxes']:
                if cat in rare_cats:
                    tpfp_top20_r = tpfp_top20_r + 1
                    if cat in gt_classes:
                        tp_top20_r = tp_top20_r + 1
                if cat in gt_classes:
                    tp_top20 = tp_top20 + 1
        
        
        logger.info("num of tp_top10, tpfp_top10, tpfn_top10, %d, %d, %d"%(tp_top10, tpfp_top10, tpfn_top10))
        logger.info("num of tp_top20, tpfp_top20, tpfn_top20, %d, %d, %d"%(tp_top20, tpfp_top20, tpfn_top20))
        p_top10 = float(tp_top10)/float(tpfp_top10-tpfp_top10_r) 
        r_top10 = float(tp_top10)/float(tpfn_top10) 
        p_top20 = float(tp_top20)/float(tpfp_top20-tpfp_top20_r) 
        r_top20 = float(tp_top20)/float(tpfn_top20) 
        
        logger.info("num of tp_top10_r, tpfp_top10_r, tpfn_top10_r, %d, %d, %d"%(tp_top10_r, tpfp_top10_r, tpfn_top10_r))
        logger.info("num of tp_top20_r, tpfp_top20_r, tpfn_top20_r, %d, %d, %d"%(tp_top20_r, tpfp_top20_r, tpfn_top20_r))
        p_top10_r = float(tp_top10_r)/float(tpfp_top10_r) 
        r_top10_r = float(tp_top10_r)/float(tpfn_top10_r) 
        p_top20_r = float(tp_top20_r)/float(tpfp_top20_r) 
        r_top20_r = float(tp_top20_r)/float(tpfn_top20_r) 
        
        logger.info("p_top10: %.3f, r_top10: %.3f, p_top20: %.3f, r_top20: %.3f"%(p_top10, r_top10, p_top20, r_top20))
        logger.info("p_top10_r: %.5f, r_top10_r: %.3f, p_top20_r: %.5f, r_top20_r: %.3f"%(p_top10_r, r_top10_r, p_top20_r, r_top20_r))

    if comm.is_main_process():
        logger = logging.getLogger(__name__)
        gt_labels_imgs = []
        pred_logits_imgs = []
        for result in tqdm(results, ncols=90):
            img_id = int(result["img_id"])
            annos = lvis_api.img_ann_map[img_id]
            gt_classes = []
            for anno in annos:
                if anno['category_id'] - 1 not in gt_classes:
                    gt_classes.append(anno['category_id'] - 1)
            
            gt_labels_per_img = torch.zeros(1, num_classes).int().cpu()
            gt_labels_per_img[:, gt_classes] = 1
            gt_labels_imgs.append(gt_labels_per_img)
            pred_logits_per_img = result["logits"].float().reshape([1, -1])
            pred_logits_imgs.append(pred_logits_per_img)
        pred_logits_imgs = torch.cat(pred_logits_imgs, dim = 0)
        gt_labels_imgs = torch.cat(gt_labels_imgs, dim = 0)
        ap_classes = compute_AP(pred_logits_imgs, gt_labels_imgs)
        ap_rare_classes = ap_classes[rare_cats]
        ap_common_classes = ap_classes[common_cats]
        ap_freq_classes = ap_classes[freq_cats]
        ap_rare_classes = ap_rare_classes[ap_rare_classes >= 0]
        ap_common_classes = ap_common_classes[ap_common_classes >= 0]
        ap_freq_classes = ap_freq_classes[ap_freq_classes >= 0]
        ap_classes = ap_classes[ap_classes >= 0]
        logger.info("ap_r: %.2f, ap_c: %.2f, ap_f: %.2f, ap: %.2f"%(ap_rare_classes.mean(), ap_common_classes.mean(), ap_freq_classes.mean(), ap_classes.mean()))


def coco_mlr_eval(results):
    
    if comm.is_main_process():
        coco_api = COCO("datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json")
        num_classes = 80
        novel_cats, base_cats = [], []
        cat_ids = sorted(coco_api.getCatIds())
        id_map = {v: i for i, v in enumerate(cat_ids)}
        
        cat_info = json.load(open("datasets/coco/zero-shot/instances_train2017_seen_2_oriorder_cat_info.json"))
        for id, info in enumerate(cat_info):
            if info["image_count"] == 0:
                novel_cats.append(id)
            else:
                base_cats.append(id)
                
        logger = logging.getLogger(__name__)
        tp_top10, tpfp_top10, tpfn_top10 = 0, 0, 0 
        tp_top20, tpfp_top20, tpfn_top20 = 0, 0, 0
        
        tp_top10_n, tpfp_top10_n, tpfn_top10_n = 0, 0, 0 
        tp_top20_n, tpfp_top20_n, tpfn_top20_n = 0, 0, 0
        
        
        for result in  tqdm(results, ncols=90):
            img_id = int(result["img_id"])
            annos = coco_api.imgToAnns[img_id]
            
            gt_classes = []
            gt_classes_n = []
            for anno in annos:
                cat_id = id_map[anno['category_id']] 
                if cat_id not in gt_classes:
                    gt_classes.append(cat_id)
                    if cat_id in novel_cats:
                        gt_classes_n.append(cat_id)
                        
            
            
            tpfn_top10 = tpfn_top10 + len(gt_classes)
            tpfp_top10 = tpfp_top10 + 10
            
            tpfn_top10_n = tpfn_top10_n + len(gt_classes_n)
            for cat in result['top10_idxes']:
                if cat in novel_cats:
                    tpfp_top10_n = tpfp_top10_n + 1
                    if cat in gt_classes:
                        tp_top10_n = tp_top10_n + 1
                if cat in gt_classes:
                    tp_top10 = tp_top10 + 1
                    
            tpfn_top20 = tpfn_top20 + len(gt_classes)
            tpfp_top20 = tpfp_top20 + 20
            tpfn_top20_n = tpfn_top20_n + len(gt_classes_n)
            for cat in result['top20_idxes']:
                if cat in novel_cats:
                    tpfp_top20_n = tpfp_top20_n + 1
                    if cat in gt_classes:
                        tp_top20_n = tp_top20_n + 1
                if cat in gt_classes:
                    tp_top20 = tp_top20 + 1
                    
        
        logger.info("num of tp_top10, tpfp_top10, tpfn_top10, %d, %d, %d"%(tp_top10, tpfp_top10, tpfn_top10))
        logger.info("num of tp_top20, tpfp_top20, tpfn_top20, %d, %d, %d"%(tp_top20, tpfp_top20, tpfn_top20))
        p_top10 = float(tp_top10 - tp_top10_n)/float(tpfp_top10-tpfp_top10_n) 
        r_top10 = float(tp_top10 - tp_top10_n)/float(tpfn_top10-tpfn_top10_n) 
        p_top20 = float(tp_top20 - tp_top20_n)/float(tpfp_top20-tpfp_top20_n) 
        r_top20 = float(tp_top20 - tp_top20_n)/float(tpfn_top20 - tpfn_top20_n) 
        
        logger.info("num of tp_top10_n, tpfp_top10_n, tpfn_top10_n, %d, %d, %d"%(tp_top10_n, tpfp_top10_n, tpfn_top10_n))
        logger.info("num of tp_top20_n, tpfp_top20_n, tpfn_top20_n, %d, %d, %d"%(tp_top20_n, tpfp_top20_n, tpfn_top20_n))
        p_top10_n = float(tp_top10_n)/float(tpfp_top10_n) 
        r_top10_n = float(tp_top10_n)/float(tpfn_top10_n) 
        p_top20_n = float(tp_top20_n)/float(tpfp_top20_n) 
        r_top20_n = float(tp_top20_n)/float(tpfn_top20_n) 
        
        logger.info("p_top10: %.3f, r_top10: %.3f, p_top20: %.3f, r_top20: %.3f"%(p_top10, r_top10, p_top20, r_top20))
        logger.info("p_top10_n: %.5f, r_top10_n: %.3f, p_top20_n: %.5f, r_top20_n: %.3f"%(p_top10_n, r_top10_n, p_top20_n, r_top20_n))


        gt_labels_imgs = []
        pred_logits_imgs = []
        for result in tqdm(results, ncols=90):
            img_id = int(result["img_id"])
            annos = coco_api.imgToAnns[img_id]
            gt_classes = []
            for anno in annos:
                cat_id = id_map[anno['category_id']] 
                if cat_id not in gt_classes:
                    gt_classes.append(cat_id)
            
            gt_labels_per_img = torch.zeros(1, num_classes).int().cpu()
            gt_labels_per_img[:, gt_classes] = 1
            gt_labels_imgs.append(gt_labels_per_img)
            pred_logits_per_img = result["logits"].float().reshape([1, -1])
            pred_logits_imgs.append(pred_logits_per_img)
        pred_logits_imgs = torch.cat(pred_logits_imgs, dim = 0)
        gt_labels_imgs = torch.cat(gt_labels_imgs, dim = 0)
        ap_classes = compute_AP(pred_logits_imgs, gt_labels_imgs)
        ap_base_classes = ap_classes[base_cats]
        ap_novel_classes = ap_classes[novel_cats]
        ap_base_classes = ap_base_classes[ap_base_classes >= 0]
        ap_novel_classes = ap_novel_classes[ap_novel_classes >= 0]
        ap_classes = ap_classes[ap_classes >= 0]
        
        logger.info("ap_b: %.2f, ap_n: %.2f, ap: %.2f"%(ap_base_classes.mean(), ap_novel_classes.mean(), ap_classes.mean()))