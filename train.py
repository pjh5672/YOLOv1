from torch import optim
from torch.utils.data import DataLoader

from dataloader import Dataset, to_image
from model import YoloModel
from utils import YoloLoss, visualize_prediction



def train(dataloader, model, criterion, optimizer, device):
    model.train()
    for index, minibatch in enumerate(dataloader):
        images, labels = minibatch[1], minibatch[2]
        predictions = model(images.to(device))
        loss, items = criterion(predictions=predictions, targets=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if index % 10 == 0:
            obj_loss, noobj_loss, bbox_loss, cls_loss = items
            print(f"loss:{loss.item():4f}, obj:{obj_loss.item():.04f}, noobj:{noobj_loss.item():.04f}, bbox:{bbox_loss.item():.04f}, cls:{cls_loss.item():.04f}")



if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    import torch

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from utils import generate_random_color
    from val import validate

    global color_list, class_list

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 224
    num_classes = 1
    batch_size = 5
    num_epoches = 150
    device = torch.device('cuda:0')

    train_dataset = Dataset(yaml_path=yaml_path, phase='train', input_size=input_size)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, sampler=None)
    val_dataset = Dataset(yaml_path=yaml_path, phase='train', input_size=input_size)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=None)
    
    color_list = generate_random_color(num_classes)
    class_list = train_dataset.class_list

    model = YoloModel(input_size=input_size, num_classes=num_classes, device=device, num_boxes=2).to(device)
    criterion = YoloLoss(num_classes=num_classes, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer.zero_grad()

    for epoch in range(1, num_epoches+1):
        print(f" ------------------------------------------- Epoch:{epoch:02d} ------------------------------------------- ")
        train(dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer, device=device)
        validate(dataloader=val_loader, model=model, conf_threshold=0.3, class_list=class_list, color_list=color_list, device=device)
    torch.save(model.state_dict(), f'./model.pt')
