# 基于YOLOV5训练是否佩戴口罩的目标检测模型

## 配置环境


这里我们使用anaconda作为库以及虚拟工作区的管理工具，按照requirements.txt中各种库的版本号进行安装。

在进行pytorch安装（gpu版本）时需要注意以下几点：

* 安装之前一定要先更新显卡驱动，去官网下载对应型号的驱动安装。
* 我使用的是一张3070Ti显卡，而30系显卡只能使用cuda11的版本。
* 一定要创建虚拟环境，这样的话各个深度学习框架之间不发生冲突。

这里创建的是python3.8的环境，安装的Pytorch的版本是1.10.0，Cuda版本为V11.6.124。


## 数据预处理与标注

虽然是yolov5的模型训练，但是我们在模式识别课程作业程序的第一部分使用了SSD网络进行模型训练。我们在那时已经使用labelimg完成了voc格式的标注。

由此这里使用我编写的voc格式转化YOLO格式的程序，对之前VOC格式的标注文件进行转化，代码库的链接如下：


```bash
https://github.com/HuangJiaqi-MISE/VOC2007_to_YOLO
```

将标注格式转化为YOLO格式后，我们要修改数据集配置文件，标记完成的数据按照下面的格式进行放置，方便程序进行索引。

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

最后，我们在data目录下创建一个`mask_data.yaml`的文件，这是YOLOv5网络中的训练数据配置文件，将：数据集路径、验证集路径、目标名称、标签数量这些信息存储在该文件中。


## 模型训练


在models下建立一个`mask_yolov5s.yaml`的模型配置文件，内容详见文件，这是YOLOv5网络中的网络结构配置文件。

模型训练之前，请确保代码目录下有以下文件：`mask_data.yaml`，`mask_yolov5s.yaml`，`yolov5s.pt`。

执行下列代码运行程序：

```bash
python train.py --data mask_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 70 --batch-size 32 --device 0(or CPU)
```

在`train/runs/exp`的目录下可以找到训练得到的模型和日志文件。

由于github上传文件的限制，我把预训练模型文件`yolov5_x.pt`以及最终训练得到的模型文件`best.pt`上传至谷歌硬盘，这是它们的链接：https://drive.google.com/drive/folders/1Oy_8JBP18THWslXC1ZJbSw97FEVNF9A5?usp=sharing

这里我们额外使用了一个训练结果可视化工具：wandb（如下图）。这是一个类似于tensorboard的训练辅助工具，它通过注册一个账号并填写配对码即可将你本地的训练结果同步到云服务器。上传至服务器的训练结果可以以epoch为单位实时显示，并自动新建数据仓库，以供使用者分享。

这里我将自己的训练结果生成以下链接以供访问：https://wandb.ai/jiaqi-huang/Training-Data-for-masked_whn-via-YOLOv5-by-HuangJiaqi?workspace=user-jiaqi-huang

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/01wandb.png)

可以看到我训练的模型在70个Epoch后已经完全收敛，LOSS呈现完美的下降曲线。

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/02wandb.png)

此外有关模型的评估数据曲线也可以直观的显示：

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/03wandb.png)

## 模型评估

在`train/runs/exp`的目录下可以找到下列评估指标曲线的图片。

目标检测最常用的评价指标是mAP，mAP是介于0到1之间的一个数字，这个数字越接近于1，就表示你的模型的性能更好。模型训练的最终mAP_0.5值为0.9615。这是一个很棒的结果。

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/mAP0.5.png)

此外还有两个指标，分别是recall和precision，两个指标p和r都是简单地从一个角度来判断模型的好坏，均是介于0到1之间的数值，其中接近于1表示模型的性能越好，接近于0表示模型的性能越差，为了综合评价目标检测的性能，一般采用均值平均密度map来进一步评估模型的好坏。我们通过设定不同的置信度的阈值，可以得到在模型在不同的阈值下所计算出的p值和r值，一般情况下，p值和r值是负相关的，绘制出来可以得到如下图所示的曲线，其中曲线的面积我们称AP，目标检测模型中每种目标可计算出一个AP值，对所有的AP值求平均则可以得到模型的mAP值，以本文为例，我们可以计算佩戴口罩和未佩戴口罩的两个目标的AP值，我们对两组AP值求平均，可以得到整个模型的mAP值，该值越接近1表示模型的性能越好。

以PR-curve为例，我的模型在验证集上的两种目标检测均值平均密度为0.995与0.973。

![](https://raw.githubusercontent.com/HuangJiaqi-MISE/Image-storage/main/PR_curve.png)

## 模型使用

模型的使用指令集全部集成在了`detect.py`目录下，可以按照下面的参考指令检测你需要的内容：

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

