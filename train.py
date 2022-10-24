from torch import optim
from torch.utils.data import DataLoader

from dataloader import Dataset, BasicTransform, AugmentTransform, to_image
from model import YoloModel
from utils import YoloLoss



def train(dataloader, model, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad()

    for index, minibatch in enumerate(dataloader):
        images, labels = minibatch[1], minibatch[2]
        predictions = model(images.to(device))
        loss, items = criterion(predictions=predictions, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if index % 50 == 0:
            obj_loss, noobj_loss, box_loss, cls_loss = items
            print(f"[Epoch:{epoch:02d}] loss:{loss.item():.4f}, obj:{obj_loss.item():.04f}, noobj:{noobj_loss.item():.04f}, box:{box_loss.item():.04f}, cls:{cls_loss.item():.04f}")



if __name__ == "__main__":
    from pathlib import Path

    import torch
    from pycocotools.coco import COCO

    from utils import generate_random_color
    from val import validate

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]

    global color_list, class_list, epoch

    yaml_path = ROOT / 'data' / 'voc.yaml'
    input_size = 224
    num_classes = 20
    batch_size = 128
    num_epoches = 100
    device = torch.device('cuda:0')
    
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_transformer = AugmentTransform(input_size=input_size)
    # train_transformer = BasicTransform(input_size=input_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataset = Dataset(yaml_path=yaml_path, phase='val')
    val_transformer = BasicTransform(input_size=input_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    color_list = generate_random_color(num_classes)
    class_list = train_dataset.class_list
    mAP_file_path = val_dataset.mAP_file_path
    cocoGt = COCO(annotation_file=mAP_file_path)

    model = YoloModel(input_size=input_size, num_classes=num_classes, device=device, num_boxes=2).to(device)
    criterion = YoloLoss(input_size=input_size, num_classes=num_classes, device=device, lambda_coord=5.0, lambda_noobj=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(1, num_epoches+1):
        train(dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer, device=device)
        validate(cocoGt=cocoGt, dataloader=val_loader, model=model, mAP_file_path=mAP_file_path, conf_threshold=0.3, nms_threshold=0.5, class_list=class_list, color_list=color_list, device=device)
    
    torch.save(model.state_dict(), f'./model_voc.pt')
