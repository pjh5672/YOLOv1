# <div align="center">YOLOv1</div>

---

## [Content]
1. [Description](#description)   
2. [Usage](#usage)  
2-1. [Model Training](#model-training)  
2-2. [Detection Evaluation](#detection-evaluation)  
2-3. [Result Analysis](#result-analysis)  
3. [Contact](#contact)   

---

## [Description]

This is a repository for PyTorch implementation of YOLOv1 following the original paper(https://arxiv.org/abs/1506.02640). You can download related weight files from [HERE !](https://drive.google.com/drive/folders/15qXxbD7RG19uZBhG3NPWwfqt6OdksAPR?usp=sharing)

 - **Performance Table**

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | mAP<br><sup>(@0.5:0.95) | mAP<br><sup>(@0.5) | Params<br><sup>(M) | FLOPS<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| YOLOv1<br><sup>(<u>Paper:page_with_curl:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 448 | *not reported* | 63.4 | *not reported* | *not reported* |
| YOLOv1 VGG16<br><sup>(<u>Paper:page_with_curl:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 448 | *not reported* | 66.4 | *not reported* | *not reported* |
| YOLOv1 ResNet18<br><sup>(<u>Our:star:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 448 | 32.9 | 60.5 | 21.95 | 18.81 |
| YOLOv1 ResNet34<br><sup>(<u>Our:star:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 448 | 35.8 | 66.1 | 32.06 | 33.65 |
| YOLOv1 ResNet50<br><sup>(<u>Our:star:</u>)</br> | PASCAL-VOC | trainval2007+2012 | test2007 | 448 | - | - | 35.07 | 37.58 |

<div align="center">

  ![Train toy-1 dataset](./asset/toy_result.gif)

</div>


## [Usage]

#### Model Training 
 - You can train your own YOLOv1 model using various backbone architectures (ResNet18, ResNet34, ResNet50, ResNet101, VGG16, and VGG16-BN) 

```python
python train.py --exp_name my_test --data voc.yaml --backbone resnet18
```


#### Detection Evaluation
 - You can compute detection metric via mean Average Precision(mAP) with IoU of 0.5, 0.75, 0.5:0.95. I follow the evaluation code with the reference on https://github.com/rafaelpadilla/Object-Detection-Metrics

```python
python val.py --exp_name my_test --data voc.yaml --ckpt_name best.pt
```


#### Result Analysis
 - After training is done, you will get the results shown below.

<div align="center">

  <a href=""><img src=./asset/figure-AP_EP150.png width="60%" /></a>

</div>


```log
2022-11-09 17:41:29 | YOLOv1 Architecture Info - Params(M): 21.95, FLOPS(B): 18.81
2022-11-09 17:43:31 | [Train-Epoch:001] multipart: 19.7455  obj: 0.5801  noobj: 16.4438  box: 1.0034  cls: 5.9263  
2022-11-09 17:45:32 | [Train-Epoch:002] multipart: 5.8577  obj: 0.5340  noobj: 0.6452  box: 0.5048  cls: 2.4772  
2022-11-09 17:47:31 | [Train-Epoch:003] multipart: 4.5413  obj: 0.5853  noobj: 0.0855  box: 0.4324  cls: 1.7515  
2022-11-09 17:49:31 | [Train-Epoch:004] multipart: 4.0832  obj: 0.6127  noobj: 0.1014  box: 0.3966  cls: 1.4368  
2022-11-09 17:51:30 | [Train-Epoch:005] multipart: 3.8923  obj: 0.6203  noobj: 0.1221  box: 0.3785  cls: 1.3186  
2022-11-09 17:53:31 | [Train-Epoch:006] multipart: 3.7505  obj: 0.6301  noobj: 0.1374  box: 0.3659  cls: 1.2219  
2022-11-09 17:55:30 | [Train-Epoch:007] multipart: 3.6210  obj: 0.6238  noobj: 0.1554  box: 0.3523  cls: 1.1581  
2022-11-09 17:57:29 | [Train-Epoch:008] multipart: 3.5554  obj: 0.6247  noobj: 0.1666  box: 0.3469  cls: 1.1130  
2022-11-09 17:59:29 | [Train-Epoch:009] multipart: 3.4773  obj: 0.6212  noobj: 0.1815  box: 0.3383  cls: 1.0738  
2022-11-09 18:01:29 | [Train-Epoch:010] multipart: 3.3752  obj: 0.6123  noobj: 0.1950  box: 0.3311  cls: 1.0098  
2022-11-09 18:01:53 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.336
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.054
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.011
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.039
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.131
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.405

                                                ...

2022-11-09 22:59:17 | [Train-Epoch:135] multipart: 1.6719  obj: 0.3912  noobj: 0.3355  box: 0.1720  cls: 0.2530  
2022-11-09 22:59:40 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.605
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.292
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.036
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.113
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.293
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.399
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.688
```


<div align="center">

<a href=""><img src=./asset/car.png width="22%" /></a> <a href=""><img src=./asset/cat.png width="22%" /></a> <a href=""><img src=./asset/dog.png width="22%" /></a> <a href=""><img src=./asset/person.png width="22%" /></a>

</div>


---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  