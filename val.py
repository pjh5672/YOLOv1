import json

import cv2
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval

from dataloader import to_image
from utils import transform_xcycwh_to_x1y1x2y2, transform_x1y1x2y2_to_x1y1wh, filter_confidence, run_NMS, visualize_prediction



@torch.no_grad()
def validate(cocoGt, dataloader, model, mAP_file_path, conf_threshold, nms_threshold, class_list, color_list, device):
    with open(mAP_file_path, mode="r") as f:
       mAP_json = json.load(f)
    imageToid = mAP_json["imageToid"]
    cocoPred = []

    model.eval()
    for i, minibatch in enumerate(dataloader):
        filenames, images, labels, ori_img_sizes = minibatch
        predictions = model(images.to(device))
        predictions[..., 5:] *= predictions[..., [0]]

        for j in range(len(filenames)):
            prediction = predictions[j].cpu().numpy()
            prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=model.input_size)
            prediction = filter_confidence(prediction=prediction, conf_threshold=conf_threshold)
            prediction = run_NMS(prediction=prediction, iou_threshold=nms_threshold)
            
            if (i==0) and (j == 0):
                check_image = to_image(images[j])
                check_pred = prediction.copy()
            
            if len(prediction) > 0:
                filename = filenames[j]
                cls_id = prediction[:, [0]]
                conf = prediction[:, [-1]]
                box_x1y1wh = transform_x1y1x2y2_to_x1y1wh(boxes=prediction[:, 1:5])
                img_id = np.array((imageToid[filename],) * len(cls_id))[:, np.newaxis]
                cocoPred.append(np.concatenate((img_id, box_x1y1wh, conf, cls_id), axis=1))

        if i == 0:
            check_result = visualize_prediction(image=check_image, prediction=check_pred, class_list=class_list, color_list=color_list)
            cv2.imwrite('./asset/test-predict.png', check_result)

    if len(cocoPred) > 0:
        cocoDt = cocoGt.loadRes(np.concatenate(cocoPred, axis=0))
        cocoEval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType="bbox")
        cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        cocoEval.evaluate()
        cocoEval.accumulate()
        eval_stats = summarize_performance(cocoEval=cocoEval)
        print(eval_stats)


def summarize_performance(cocoEval):
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = cocoEval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            s = cocoEval.eval['precision']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            s = cocoEval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        return mean_s
        
    stats = np.zeros((12,))
    stats[0] = _summarize(1)
    stats[1] = _summarize(1, iouThr=.5, maxDets=cocoEval.params.maxDets[2])
    stats[2] = _summarize(1, iouThr=.75, maxDets=cocoEval.params.maxDets[2])
    stats[3] = _summarize(1, areaRng='small', maxDets=cocoEval.params.maxDets[2])
    stats[4] = _summarize(1, areaRng='medium', maxDets=cocoEval.params.maxDets[2])
    stats[5] = _summarize(1, areaRng='large', maxDets=cocoEval.params.maxDets[2])
    stats[6] = _summarize(0, maxDets=cocoEval.params.maxDets[0])
    stats[7] = _summarize(0, maxDets=cocoEval.params.maxDets[1])
    stats[8] = _summarize(0, maxDets=cocoEval.params.maxDets[2])
    stats[9] = _summarize(0, areaRng='small', maxDets=cocoEval.params.maxDets[2])
    stats[10] = _summarize(0, areaRng='medium', maxDets=cocoEval.params.maxDets[2])
    stats[11] = _summarize(0, areaRng='large', maxDets=cocoEval.params.maxDets[2])
    return stats



if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    from pycocotools.coco import COCO
    from dataloader import Dataset, BasicTransform
    from model import YoloModel
    from utils import generate_random_color

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 224
    num_classes = 1
    batch_size = 8
    device = torch.device('cuda:0')
    checkpoint = torch.load("./model_toy.pt")

    transformer = BasicTransform(input_size=input_size)
    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_dataset.load_transformer(transformer=transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=None)
    model = YoloModel(input_size=input_size, num_classes=num_classes, device=device, num_boxes=2).to(device)
    model.load_state_dict(checkpoint, strict=True)

    color_list = generate_random_color(num_classes)
    class_list = val_dataset.class_list
    mAP_file_path = val_dataset.mAP_file_path
    cocoGt = COCO(annotation_file=mAP_file_path)

    validate(cocoGt=cocoGt, dataloader=val_loader, model=model, 
             mAP_file_path=mAP_file_path, conf_threshold=0.4, nms_threshold=0.5, 
             class_list=class_list, color_list=color_list, device=device)