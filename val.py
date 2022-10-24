import cv2
import torch
from torch.utils.data import DataLoader

from dataloader import Dataset, BasicTransform, to_image
from model import YoloModel
from utils import transform_xcycwh_to_x1y1x2y2, filter_confidence, run_NMS, visualize_prediction



@torch.no_grad()
def validate(dataloader, model, conf_threshold, nms_threshold, class_list, color_list, device):
    model.eval()
    
    for i, minibatch in enumerate(dataloader):
        filenames, images, labels, ori_img_sizes = minibatch
        predictions = model(images.to(device))
        predictions[..., 5:] *= predictions[..., [0]]

        for j in range(len(filenames)):
            if j == 0:
                canvas_img = to_image(images[j])
                prediction = predictions[j].cpu().numpy()
                prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=model.input_size)
                prediction = filter_confidence(prediction=prediction, conf_threshold=conf_threshold)
                prediction = run_NMS(prediction=prediction, iou_threshold=nms_threshold)

        if i == 0:
            canvas_img = visualize_prediction(image=canvas_img, prediction=prediction, class_list=class_list, color_list=color_list)
            cv2.imwrite('./asset/test-predict.png', canvas_img)
            break



if __name__ == "__main__":
    from pathlib import Path
    from utils import generate_random_color

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 224
    num_classes = 1
    batch_size = 8
    device = torch.device('cpu')
    checkpoint = torch.load("./model.pt")

    transformer = BasicTransform(input_size=input_size)
    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_dataset.load_transformer(transformer=transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=None)
    model = YoloModel(input_size=input_size, num_classes=num_classes, device=device, num_boxes=2).to(device)
    model.load_state_dict(checkpoint, strict=True)

    color_list = generate_random_color(num_classes)
    class_list = val_dataset.class_list

    validate(dataloader=val_loader, model=model, conf_threshold=0.4, nms_threshold=0.5, class_list=class_list, color_list=color_list, device=device)