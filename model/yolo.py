import torch
from torch import nn

from backbone import build_resnet18
from head import YoloHead



class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, device, num_boxes=2):
        super().__init__()
        self.stride = 32
        self.device = device
        self.num_boxes = num_boxes
        self.input_size = input_size
        self.num_classes = num_classes
        self.grid_size = input_size // self.stride
        self.backbone, feat_dims = build_resnet18(pretrained=True)
        self.head = YoloHead(in_channels=feat_dims, num_classes=num_classes, num_boxes=num_boxes)
        self.set_grid(input_size)


    def forward(self, x):
        batch_size = x.shape[0]
        out = self.backbone(x)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        pred_obj = out[..., [0,5]].view(batch_size, -1, 1)
        pred_box = torch.cat((out[..., 1:5], out[..., 6:10]), dim=-1).view(batch_size, -1, 4)
        pred_cls = out[..., 10:].view(batch_size, -1, self.num_classes).tile(self.num_boxes,1)

        if self.training:
            return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)
        else:
            pred_box = self.transform_pred_box(pred_box)
            return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)
        

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_size = self.input_size // self.stride
        grid_y, grid_x = torch.meshgrid((torch.arange(self.grid_size), torch.arange(self.grid_size)), indexing="ij")
        self.grid_x = grid_x.contiguous().view((1, -1)).tile(1,self.num_boxes).to(self.device)
        self.grid_y = grid_y.contiguous().view((1, -1)).tile(1,self.num_boxes).to(self.device)


    def transform_pred_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x) * self.stride
        yc = (pred_box[..., 1] + self.grid_y) * self.stride
        w = torch.clamp(pred_box[..., 2] * self.stride, min=0, max=self.input_size)
        h = torch.clamp(pred_box[..., 3] * self.stride, min=0, max=self.input_size)
        return torch.stack((xc, yc, w, h), dim=-1)



if __name__ == "__main__":
    input_size = 224
    num_classes = 1
    inp = torch.randn(1, 3, input_size, input_size)
    device = torch.device('cpu')

    model = YoloModel(input_size=input_size, num_classes=num_classes, device=device).to(device)
    model.train()
    out = model(inp.to(device))
    print(out.shape)

    model.eval()
    out = model(inp.to(device))
    print(out.shape)