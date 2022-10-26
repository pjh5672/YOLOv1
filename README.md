# <div align="center">YOLOv1</div>

---

## [Contents]
1. [Description](#description)   
2. [Contribution](#contact)   

---

## [Description]

This is a repository for reproducing YOLOv1 detection model following the [*original paper*](https://arxiv.org/abs/1506.02640)

### Step 1. Train single image for checking model recognizing  
- **Train: toy_dataset/train_1, Val: toy_dataset/train_1**
- Intentional overfitting without any augmentation
- 448 x 448 px input size
- SGD optimization
- Multi-part, objectness, noobjectness, box, confidence losses should be near 0.0 after 100 epochs with learning rate of 1e-3

<div align="center">
<div> Inference result for 135 epochs </div>
<img src=./asset/toy_1_result.gif width="30%"/>
</div>

### Step 2. Train 9 images for batch-training  
- **Train: toy_dataset/train_2, Val: toy_dataset/train_1**
- Intentional overfitting without any augmentation
- 448 x 448 px input size
- Setting learning rate to 1e-3 with SGD optimization

<div align="center">
<div> Inference result for 135 epochs </div>
<img src=./asset/toy_2_result.gif width="30%"/>
</div>

### Step 3. Train 10 images including one background image
- **Train: toy_dataset/train_3, Val: toy_dataset/train_2**
- Intentional overfitting without any augmentation
- 448 x 448 px input size
- Setting learning rate to 1e-3, momentum of 0.9, weight decay of 0.0005 with SGD optimization
- Exception handling for no-object case
- Best mAP value should be approximately as below

<div align="center">
  <div> No-object case(bird) </div>

<img src=./asset/007102.jpg width="25%" />
</div>

<div align="center">
Best mAP performance training after 150 epochs

```log
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.778
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.439
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.489
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.544
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.544
```

</div>

<div align="center">
  <div> Inference results for 135 epochs </div>

![Train toy-1 dataset](./asset/toy_3_result.gif)
</div>


## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  