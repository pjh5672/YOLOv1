import cv2
import torch
from torch.utils.data import DataLoader

from dataloader import Dataset, to_image
from model import YoloModel
from utils import filter_confidence, hard_NMS, visualize_prediction



@torch.no_grad()
def validate(dataloader, model, conf_threshold, class_list, color_list, device):
    model.eval()
    for i, minibatch in enumerate(dataloader):
        filenames, images, labels, ori_img_sizes = minibatch
        predictions = model(images.to(device))
        
        for j in range(len(filenames)):
            if j == 0:
                canvas_img = to_image(images[j])
                prediction = predictions[j]
                grid_size = prediction.shape[-1]
                y_grid, x_grid = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
                y_grid = y_grid.to(device)
                x_grid = x_grid.to(device)
                conf = prediction[[0,5], ...].reshape(1, -1)
                xc = ((prediction[[1,6], ...] + x_grid) * 32).reshape(1,-1).clip(min=0, max=224)
                yc = ((prediction[[2,7], ...] + y_grid) * 32).reshape(1,-1).clip(min=0, max=224)
                w = (prediction[[3,8], ...] * 32).reshape(1,-1).clip(min=0, max=224)
                h = (prediction[[4,9], ...] * 32).reshape(1,-1).clip(min=0, max=224)
                cls_id = torch.max(prediction[10:, ...].reshape(1, -1).tile(1,2) * conf, dim=0).indices
                prediction = torch.cat([conf, xc, yc, w, h, cls_id.unsqueeze(dim=0)], dim=0)
                prediction = prediction.transpose(0,1).cpu().numpy()
                prediction = filter_confidence(prediction, conf_threshold=conf_threshold)

        if i == 0:
            canvas_img = visualize_prediction(image=canvas_img, prediction=prediction, class_list=class_list, color_list=color_list)
            cv2.imwrite('./asset/test-predict.png', canvas_img)
            break



if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    from utils import generate_random_color

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 224
    num_classes = 1
    batch_size = 2
    device = torch.device('cpu')
    checkpoint = torch.load("./model.pt")

    val_dataset = Dataset(yaml_path=yaml_path, phase='val', input_size=input_size)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=None)
    model = YoloModel(input_size=input_size, num_classes=num_classes, device=device, num_boxes=2).to(device)
    model.load_state_dict(checkpoint, strict=True)

    color_list = generate_random_color(num_classes)
    class_list = val_dataset.class_list

    validate(dataloader=val_loader, model=model, conf_threshold=0.3, class_list=class_list, color_list=color_list, device=device)