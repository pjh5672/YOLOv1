import os
import sys
from pathlib import Path

import torch
from torch import nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid



class YoloLoss():
    def __init__(self, num_classes, grid_size, lambda_coord=5.0, lambda_noobj=0.5):
        self.grid_size = grid_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_attributes = (1 + 4) + num_classes
        self.mse_loss = nn.MSELoss(reduction="mean")
        grid_x, grid_y = set_grid(grid_size=self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1))
        self.grid_y = grid_y.contiguous().view((1, -1))


    def __call__(self, predictions, labels):
        self.device = predictions.device
        self.batch_size = predictions.shape[0]
        targets = self.build_batch_target(labels).to(self.device)

        with torch.no_grad():
            iou1 = self.calculate_iou(pred_box_cxcywh=predictions[:, :(self.grid_size * self.grid_size), 1:5], target_box_cxcywh=targets[..., 1:5])
            iou2 = self.calculate_iou(pred_box_cxcywh=predictions[:, (self.grid_size * self.grid_size):, 1:5], target_box_cxcywh=targets[..., 1:5])
            max_iou, best_box = torch.stack((iou1, iou2), dim=-1).max(dim=-1)
            best_box = torch.cat((best_box.eq(0), best_box.eq(1)), dim=-1)

        positive_mask = (targets[..., 0].tile(1,2) * best_box).bool()
        pred_obj = (predictions[..., 0])[positive_mask]
        pred_noobj = predictions[..., 0][~positive_mask]
        pred_box_txty = predictions[..., 1:3][positive_mask]
        pred_box_twth = predictions[..., 3:5][positive_mask]
        pred_cls = predictions[..., 5:][positive_mask]

        true_mask = targets[..., 0].bool()
        target_obj = (targets[..., 0] * max_iou)[true_mask]
        target_box_txty = targets[..., 1:3][true_mask]
        target_box_twth = targets[..., 3:5][true_mask]
        target_cls = targets[..., 5:][true_mask]

        obj_loss = self.mse_loss(pred_obj, target_obj)
        noobj_loss = self.mse_loss(pred_noobj, pred_noobj * 0)
        txty_loss = self.mse_loss(pred_box_txty, target_box_txty)
        twth_loss = self.mse_loss(pred_box_twth.sign() * (pred_box_twth.abs() + 1e-8).sqrt(), (target_box_twth + 1e-8).sqrt())
        cls_loss = self.mse_loss(pred_cls, target_cls)
        multipart_loss = obj_loss + self.lambda_noobj * noobj_loss + self.lambda_coord * (txty_loss + twth_loss) + cls_loss
        return multipart_loss, obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss


    def build_target(self, label):
        target = torch.zeros(size=(self.grid_size, self.grid_size, self.num_attributes), dtype=torch.float32)
        
        if -1 in label[:, 0]:
            return target
        else:
            for item in label:
                cls_id = item[0].long()
                grid_i = (item[1] * self.grid_size).long()
                grid_j = (item[2] * self.grid_size).long()
                tx = (item[1] * self.grid_size) - grid_i
                ty = (item[2] * self.grid_size) - grid_j
                tw = item[3]
                th = item[4]
                target[grid_j, grid_i, 0] = 1.0
                target[grid_j, grid_i, 1:5] = torch.Tensor([tx, ty, tw, th])
                target[grid_j, grid_i, 5 + cls_id] = 1.0
            return target
    
    
    def build_batch_target(self, labels):
        batch_target = torch.stack([self.build_target(label) for label in labels], dim=0)
        return batch_target.view(self.batch_size, -1,  self.num_attributes)
    
    
    def calculate_iou(self, pred_box_cxcywh, target_box_cxcywh):
        pred_box_x1y1x2y2 = self.transform_cxcywh_to_x1y1x2y2(pred_box_cxcywh)
        target_box_x1y1x2y2 = self.transform_cxcywh_to_x1y1x2y2(target_box_cxcywh)
        x1 = torch.max(pred_box_x1y1x2y2[..., 0], target_box_x1y1x2y2[..., 0])
        y1 = torch.max(pred_box_x1y1x2y2[..., 1], target_box_x1y1x2y2[..., 1])
        x2 = torch.min(pred_box_x1y1x2y2[..., 2], target_box_x1y1x2y2[..., 2])
        y2 = torch.min(pred_box_x1y1x2y2[..., 3], target_box_x1y1x2y2[..., 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        union = abs(pred_box_cxcywh[..., 2] * pred_box_cxcywh[..., 3]) + abs(target_box_cxcywh[..., 2] * target_box_cxcywh[..., 3]) - inter
        inter[inter.gt(0)] = inter[inter.gt(0)] / union[inter.gt(0)]
        return inter
    

    def transform_cxcywh_to_x1y1x2y2(self, boxes):
        xc = boxes[..., 0] + self.grid_x.to(self.device)
        yc = boxes[..., 1] + self.grid_y.to(self.device)
        w = boxes[..., 2]
        h = boxes[..., 3]
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        return torch.stack((x1, y1, x2, y2), dim=-1)



if __name__ == "__main__":
    from torch import optim
    from torch.utils.data import DataLoader
    
    from dataloader import Dataset, BasicTransform
    from model import YoloModel

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 448
    num_classes = 1
    batch_size = 2
    device = torch.device('cuda')

    transformer = BasicTransform(input_size=input_size)
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_dataset.load_transformer(transformer=transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=None)
    
    model = YoloModel(num_classes=num_classes, grid_size=7, num_boxes=2).to(device)
    criterion = YoloLoss(num_classes=num_classes, grid_size=model.grid_size, lambda_coord=5.0, lambda_noobj=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer.zero_grad()

    for epoch in range(10):
        model.train()
        for index, minibatch in enumerate(train_loader):
            filenames, images, labels, ori_img_sizes = minibatch
            predictions = model(images.to(device))
            loss = criterion(predictions=predictions, labels=labels)
            loss[0].backward()
            optimizer.step()
            optimizer.zero_grad()

            if index % 50 == 0:
                multipart_loss, obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss = loss
                print(f"[Epoch:{epoch:02d}] loss:{multipart_loss.item():.4f}, obj:{obj_loss.item():.04f}, noobj:{noobj_loss.item():.04f}, txty:{txty_loss.item():.04f}, twth:{twth_loss.item():.04f}, cls:{cls_loss.item():.04f}")
