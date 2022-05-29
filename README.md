# Target detection model based on YOLOV5 training masks

## Configuring the environment


Here we use anaconda as a management tool for the libraries and the virtual workspace, installed according to the version numbers of the various libraries in requirements.txt.

When doing a pytorch installation (gpu version) the following points need to be noted.

* Make sure to update your graphics card drivers before installing, go to the official website and download the drivers for the corresponding model and install them.
* I am using a 3070Ti GPU, and the 30-series graphics card can only use the cuda11 version.
* Be sure to create a virtual environment so that there are no conflicts between the various deep learning frameworks.

Here I created a python 3.8 environment with Pytorch version 1.10.0 and Cuda version V11.6.124.


## Data pre-processing and labelling

Although the model was trained in yolov5, we used the SSD network for model training in the first part of the pattern recognition coursework program. We had already completed the labeling of the voc format using labelimg at that point.

From this the previous VOC format annotation file was converted here using a program I wrote to convert the voc format to YOLO format, the link to the code base is as follows.


```bash
https://github.com/HuangJiaqi-MISE/VOC2007_to_YOLO
```

After converting the annotation format to YOLO format, we have to modify the dataset configuration file and the marked-up data is placed in the following format to facilitate indexing by the program.

```bash
YOLO_Mask
└─ score
       ├─ images
       │    ├─ test # 下面放测试集图片
       │    ├─ train # 下面放训练集图片
       │    └─ val # 下面放验证集图片
       └─ labels
              ├─ test # 下面放测试集标签
              ├─ train # 下面放训练集标签
              ├─ val # 下面放验证集标签
```

Finally, we create a `mask_data.yaml` file in the data directory, which is the training data configuration file for the YOLOv5 network, and store the following information: dataset path, validation set path, target name, and number of labels in this file.


## Model training


Create a model configuration file `mask_yolov5s.yaml` under models, the contents of which are detailed in the file, which is the network structure configuration file in the YOLOv5 network.

Before training the model, ensure that the following files are available in the code directory: `mask_data.yaml`, `mask_yolov5s.yaml`, `yolov5s.pt`.

Run the program by executing the following code.

```bash
python train.py --data mask_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 70 --batch-size 32 --device 0(or CPU)
```

The trained models and log files can be found in the `train/runs/exp` directory.

Due to the limitation of github to upload files, I uploaded the pre-trained model file `yolov5s.pt` and the final trained model file `best.pt` to Google Drive, here is the link to them: https://drive.google.com/drive/folders/1Oy_ 8JBP18THWslXC1ZJbSw97FEVNF9A5?usp=sharing

Here we use an additional tool for visualising the training results: wandb (see below). This is a training aid similar to tensorboard, which synchronises your local training results to the cloud server by registering an account and filling in a pairing code. The training results uploaded to the server can be displayed in real time in epochs and a new data repository is automatically created for users to share.

Here I have generated the following link to my training results for access: https://wandb.ai/jiaqi-huang/Training-Data-for-masked_whn-via-YOLOv5-by-HuangJiaqi?workspace=user-jiaqi-huang

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/01wandb.png)

You can see that the model I trained has fully converged after 70 Epochs and the LOSS shows a perfect descent curve.

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/02wandb.png)

In addition the evaluation data curves for the models in question can be visualised as follows.

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/03wandb.png)

## Model assessment

Images of the following evaluation indicator curves can be found in the `train/runs/exp` directory.

The most common evaluation metric for target detection is mAP. mAP is a number between 0 and 1, and the closer the number is to 1, the better your model performs. The final mAP_0.5 value for the model training was 0.9615. This is a great result.

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/mAP0.5.png)

In addition, there are two other metrics, recall and precision. Both metrics p and r are simply a way to judge the goodness of the model from one perspective, and are values between 0 and 1, where close to 1 means better performance of the model and close to 0 means worse performance of the model. To evaluate the performance of the target detection comprehensively, the mean average density map is generally used to further assess the goodness of the model. By setting different confidence thresholds, we can obtain the p and r values calculated by the model under different thresholds, in general, the p and r values are negatively correlated, and the curve can be obtained as shown below, where the area of the curve is called AP. In this paper, for example, we can calculate the AP values of two targets with and without masks, and we can average the two sets of AP values to obtain the mAP value of the whole model.

Using the PR-curve as an example, the mean density of the two target detection averages for my model on the validation set is 0.995 versus 0.973.

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/PR_curve.png)

## Model usage

The set of instructions for using the model are all integrated in the `detect.py` directory and you can follow the following reference instructions to detect what you need.

```bash
 # 检测摄像头
 python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source 0  # webcam
 # 检测图片文件
  python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source file.jpg  # image 
 # 检测视频文件
   python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source file.mp4  # video
 # 检测一个目录下的文件
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt path/  # directory
 # 检测网络视频
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'https://youtu.be/NUsoVlDFqZg'  # YouTube video
 # 检测流媒体
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream                            
```



> Author：Huang Jiaqi
> 
> Created: 2022-05-21
> 
> Last updated: 2022-05-27



