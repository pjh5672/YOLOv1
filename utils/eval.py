import json
from collections import defaultdict

import numpy as np



class Evaluator():
    ########################################################################################
    #  This work is refered by: https://github.com/rafaelpadilla/Object-Detection-Metrics  #
    ########################################################################################
    def __init__(self, annotation_file):
        self.maxDets = 100
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        
        with open(annotation_file, "r") as f:
            val_anno = json.load(f)
        self.groundtruths = self.split_areaRng(data=val_anno["annotations"])
        self.areaToann = {}
        for areaLbl in self.areaRngLbl:
            groundtruths = self.groundtruths[areaLbl]
            clsToann = self.build_clsToann(groundtruths=groundtruths)
            clsToimgToann = self.build_clsToimgToann(clsToann=clsToann)
            self.areaToann[areaLbl] = clsToimgToann


    def __call__(self, predictions):
        predictions = self.transform_prediction_eval_format(data=predictions)
        predictions = self.split_areaRng(data=predictions)

        mAP_dict = {}
        for areaLbl in self.areaRngLbl:
            classes_AP = []
            clsToimgToann = self.areaToann[areaLbl]
            preds = predictions[areaLbl]
            for c in sorted(clsToimgToann.keys()):
                res = self.calculate_AP(groundtruths=clsToimgToann, predictions=preds, class_id=c)
                classes_AP.append(res)
            mAP_dict[areaLbl] = self.calculate_mAP(classAP_data=classes_AP)
        
        eval_text = '\n'
        for areaLbl in self.areaRngLbl:
            if areaLbl == 'all':
                eval_text += self.summarize(mAP_dict[areaLbl]['mAP_5095'], iouThr=None, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
                eval_text += self.summarize(mAP_dict[areaLbl]['mAP_50'], iouThr=0.50, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
                eval_text += self.summarize(mAP_dict[areaLbl]['mAP_75'], iouThr=0.75, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
            else:
                eval_text += self.summarize(mAP_dict[areaLbl]['mAP_5095'], iouThr=None, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
                eval_text += self.summarize(mAP_dict[areaLbl]['mAP_50'], iouThr=0.50, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
        return mAP_dict, eval_text


    def summarize(self, mAP, iouThr=None, areaLbl='all', maxDets=100):
        iStr = '\t - {:<16} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision'
        typeStr = '(AP)'
        iouStr = '{:0.2f}:{:0.2f}'.format(self.iouThrs[0], self.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)
        return iStr.format(titleStr, typeStr, iouStr, areaLbl, maxDets, mAP)


    def calculate_mAP(self, classAP_data):
        AP_50_per_class, PR_50_pts_per_class = {}, {}
        num_true_per_class, num_positive_per_class, num_TP_50_per_class, num_FP_50_per_class = {}, {}, {}, {}
        mAP_50, mAP_75, mAP_5095 = 0, 0, 0
        valid_num_classes = 0 + 1E-8

        for res in classAP_data:
            if res["total_positive"] > 0:
                valid_num_classes += 1
                AP_50_per_class[res["class"]] = res["AP_50"]
                PR_50_pts_per_class[res["class"]] = {"mprec": res["prec_50"], "mrec": res["rec_50"]}
                num_true_per_class[res["class"]] = res["total_true"]
                num_positive_per_class[res["class"]] = res["total_positive"]
                num_TP_50_per_class[res["class"]] = res["total_TP_50"]
                num_FP_50_per_class[res["class"]] = res["total_FP_50"]
                mAP_50 += res["AP_50"]
                mAP_75 += res["AP_75"]
                mAP_5095 += res["AP_5095"]
                
        mAP_50 /= valid_num_classes
        mAP_75 /= valid_num_classes
        mAP_5095 /= valid_num_classes

        res = {"AP_50_PER_CLASS": AP_50_per_class,
               "PR_50_PTS_PER_CLASS": PR_50_pts_per_class,
               "NUM_TRUE_PER_CLASS": num_true_per_class,
               "NUM_POSITIVE_PER_CLASS": num_positive_per_class,
               "NUM_TP_50_PER_CLASS": num_TP_50_per_class,
               "NUM_FP_50_PER_CLASS": num_FP_50_per_class,
               "mAP_50": round(mAP_50, 4),
               "mAP_75": round(mAP_75, 4),
               "mAP_5095": round(mAP_5095, 4)}
        return res
        

    def calculate_AP(self, groundtruths, predictions, class_id):
        imgToann = groundtruths[class_id]
        pred_for_cls_id = [pred for pred in predictions if pred["category_id"] == class_id]
        pred_for_cls_id = sorted(pred_for_cls_id, key=lambda x:x["score"], reverse=True)

        num_true = sum([len(imgToann[img_id]) for img_id in imgToann])
        num_positive = len(pred_for_cls_id)
        TP = np.zeros(shape=(len(self.iouThrs), num_positive))
        FP = np.zeros(shape=(len(self.iouThrs), num_positive))

        if num_positive == 0:
            res = {
                "class": class_id,
                "prec_50": 0,
                "rec_50": 0,
                "total_true": num_true,
                "total_positive": num_positive,
                "total_TP_50": int(np.sum(TP[0])),
                "total_FP_50": int(np.sum(FP[0])),
                "AP_50": 0,
                "AP_75": 0,
                "AP_5095": 0
            }
            return res

        imgToflag = {}
        for img_id in imgToann:
            imgToflag[img_id] = np.zeros(shape=(len(self.iouThrs), len(imgToann[img_id])))

        for i in range(len(pred_for_cls_id)):
            pred = pred_for_cls_id[i]
            ann = imgToann[pred["image_id"]]

            iou_max = 0
            for j in range(len(ann)):
                iou = self.get_IoU(pred["bbox"], ann[j]["bbox"])
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

            for k in range(len(self.iouThrs)):
                if iou_max >= self.iouThrs[k]:
                    if imgToflag[pred["image_id"]][k, jmax] == 0:
                        imgToflag[pred["image_id"]][k, jmax] = 1
                        TP[k, i] = 1
                    else:
                        FP[k, i] = 1
                else:
                    FP[k, i] = 1

        acc_FP = np.cumsum(FP, axis=1)
        acc_TP = np.cumsum(TP, axis=1)
        rec = acc_TP / (num_true + 1e-10)
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        APs = []
        for i in range(len(self.iouThrs)):
            ap, mprec, mrec = self.EveryPointInterpolation(rec[i], prec[i])
            if i == 0:
                mprec_50 = mprec
                mrec_50 = mrec
            APs.append(ap)
            
        res = {
                "class": class_id,
                "prec_50": [round(item, 4) for item in mprec_50],
                "rec_50": [round(item, 4) for item in mrec_50],
                "total_true": num_true,
                "total_positive": num_positive,
                "total_TP_50": int(np.sum(TP[0])),
                "total_FP_50": int(np.sum(FP[0])),
                "AP_50": round(APs[0], 4),
                "AP_75": round(APs[5], 4),
                "AP_5095": round(sum(APs) / len(APs), 4)
            }
        return res


    def ElevenPointInterpolatedAP(self, rec, prec):
        mrec = [e for e in rec]
        mpre = [e for e in prec]
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)

        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11

        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)

        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return ap, rhoInterp, recallValues


    def EveryPointInterpolation(self, rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1+i] != mrec[i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1]


    def transform_prediction_eval_format(self, data):
        ann = []
        for i in range(data.shape[0]):
            ann += [{
                "image_id": int(data[i, 0]),
                "bbox" : [round(item, 2) for item in data[i, 1:5].tolist()],
                "area": round((data[i, 3] * data[i, 4]), 2),
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann


    def is_intersect(self, boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        if boxA[2] < boxB[0]:
            return False  # boxA is left boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        return True


    def get_intersection(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return (xB - xA + 1) * (yB - yA + 1)


    def get_union(self, boxA, boxB):
        area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return area_A + area_B


    def transform_x1y1wh_to_x1y1x2y2(self, box):
        x1 = round(box[0], 2)
        y1 = round(box[1], 2)
        x2 = round(box[0] + box[2], 2)
        y2 = round(box[1] + box[3], 2)
        return [x1, y1, x2, y2]


    def get_IoU(self, boxA, boxB):
        # x1y1wh -> x1y1x2y2
        boxA = self.transform_x1y1wh_to_x1y1x2y2(boxA)
        boxB = self.transform_x1y1wh_to_x1y1x2y2(boxB)
        if self.is_intersect(boxA, boxB) is False:
            return 0
        inter = self.get_intersection(boxA, boxB)
        union = self.get_union(boxA, boxB)
        iou = inter / (union - inter)
        return iou


    def build_clsToimgToann(self, clsToann):
        clsToimgToann = {}
        for cls_id in clsToann.keys():
            imgToann = defaultdict(list)
            for ann in clsToann[cls_id]:
                imgToann[ann["image_id"]].append(ann)
            clsToimgToann[cls_id] = imgToann
        return clsToimgToann


    def build_clsToann(self, groundtruths):
        clsToann = defaultdict(list)
        for anno in groundtruths:
            clsToann[anno["category_id"]].append(anno)
        if -1 in clsToann.keys():
            del clsToann[-1]
        return clsToann


    def split_areaRng(self, data):
        res = defaultdict(list)
        for dt in data:
            if self.areaRng[0][0] <= dt["area"] < self.areaRng[0][1]:
                res[self.areaRngLbl[0]].append(dt)
            if self.areaRng[1][0] <= dt["area"] < self.areaRng[1][1]:
                res[self.areaRngLbl[1]].append(dt)
            elif self.areaRng[2][0] <= dt["area"] < self.areaRng[2][1]:
                res[self.areaRngLbl[2]].append(dt)
            elif self.areaRng[3][0] <= dt["area"] < self.areaRng[3][1]:
                res[self.areaRngLbl[3]].append(dt)
        return res
