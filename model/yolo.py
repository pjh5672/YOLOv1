import sys
from pathlib import Path

import gdown
import torch
from torch import nn

from backbone import build_backbone
from neck import ConvBlock
from head import YoloHead

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid


model_urls = {
    "yolov1-vgg16": "https://drive.google.com/file/d/1yIEFsSXlsOeJVAnt164NBGmZPg8J_ZRm/view?usp=share_link",
    "yolov1-vgg16-bn": "https://drive.google.com/file/d/1NSHsPiJc3EVAo8SQX2HqSpCK3iQVocNa/view?usp=share_link",
    "yolov1-resnet18": "https://drive.google.com/file/d/1EETZU5z4c1lff3zOBk6jHFwBsORd065X/view?usp=share_link",
    "yolov1-resnet34": "https://drive.google.com/file/d/1-AAAFd8ADxquma5u36mOHB9eBM514RzI/view?usp=share_link",
    "yolov1-resnet50": "https://drive.google.com/file/d/1oc8dNiQGImQFy2aXmU7NlupL_13vvib4/view?usp=share_link",
}


class YoloModel(nn.Module):
    def __init__(self, input_size, backbone, num_classes, pretrained=False):
        super().__init__()
        self.stride = 32
        self.grid_size = input_size // self.stride
        self.num_classes = num_classes
        self.backbone, feat_dims = build_backbone(arch_name=backbone)
        self.neck = ConvBlock(in_channels=feat_dims, out_channels=512)
        self.head = YoloHead(in_channels=512, num_classes=num_classes)
        grid_x, grid_y = set_grid(grid_size=self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1))
        self.grid_y = grid_y.contiguous().view((1, -1))
        
        if pretrained:
            download_path = ROOT / "weights" / f"yolov1-{backbone}.pt"
            if not download_path.is_file():
                gdown.download(model_urls[f"yolov1-{backbone}"], str(download_path), quiet=False, fuzzy=True)
            ckpt = torch.load(download_path, map_location="cpu")
            self.load_state_dict(ckpt["model_state"], strict=False)


    def forward(self, x):
        self.device = x.device
        
        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        pred_obj = torch.sigmoid(out[..., [0]])
        pred_box = torch.sigmoid(out[..., 1:5])
        pred_cls = out[..., 5:]
        
        if self.training:
            return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)
        else:
            pred_box = self.transform_pred_box(pred_box)
            pred_score = pred_obj * torch.softmax(pred_cls, dim=-1)
            pred_score, pred_label = pred_score.max(dim=-1)
            return torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1)), dim=-1)


    def transform_pred_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (pred_box[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        w = pred_box[..., 2]
        h = pred_box[..., 3]
        return torch.stack((xc, yc, w, h), dim=-1)



if __name__ == "__main__":
    input_size = 448
    num_classes = 20
    inp = torch.randn(2, 3, input_size, input_size)
    device = torch.device('cuda')

    model = YoloModel(input_size=input_size, backbone="resnet18", num_classes=num_classes, pretrained=True).to(device)
    model.train()
    out = model(inp.to(device))
    print(out.shape)

    model.eval()
    out = model(inp.to(device))
    print(out.device)
    print(out.shape)