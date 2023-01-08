import gdown

url = 'https://drive.google.com/file/d/1EETZU5z4c1lff3zOBk6jHFwBsORd065X/view?usp=share_link'
output_path = './weights/yolov1-resnet18.pt'
gdown.download(url, output_path, quiet=False,fuzzy=True)