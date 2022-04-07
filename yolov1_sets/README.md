YOLO官网教程：https://pjreddie.com/darknet/yolo/

yolov1：[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)

darknet训练自己的数集:[https://blog.csdn.net/maizousidemao/article/details/103442356](https://blog.csdn.net/maizousidemao/article/details/103442356)

其他参考：\
https://www.cnblogs.com/answerThe/p/11481564.html \ 

https://zhuanlan.zhihu.com/p/92141879?utm_source=qq 

```
需要的权重：
```
wget https://pjreddie.com/media/files/darknet53.conv.74
wget https://pjreddie.com/media/files/yolov3.weights --no-check-certificate
wget https://pjreddie.com/media/files/darknet53.conv.74 --no-check-certificate
```

Ubuntu16.04配置安装darknet+OPENCV:https://blog.csdn.net/gzj_1101/article/details/78651650

测试：
```
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video file>
./darknet detector train cfg/voc.data cfg/yolov3-tiny.cfg | tee person_train_log.txt
./darknet detect cfg/yolov3-tiny.cfg backup/yolov3-tiny_10000.weights +图片地址
```
YOLOv3+opencv识别调用笔记本摄像头: \
./darknet detector demo cfg/voc.data cfg/yolov3-voc.cfg weights/yolov3.weights -c 0

训练：./darknet detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74 -gpus 0,1,2,3
先评估一下测试：./darknet detector valid cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_final.weights







参数说明：

[net]
# Testing       #测试模式
 batch=1    
 subdivisions=1
# Training          #训练模式  每次前向图片的数目=batch/subdivisions
# batch=64          #关于batch与subdivision：
                    #在训练输出中，训练迭代包括8组，这些batch样本又被平均分成subdivision=8次送入网络参与训练，以减轻内存占用的压力
# subdivisions=16   #batch越大，训练效果越好，subdivision越大，占用内存压力越小

width=416        #网络输入的宽、高、通道数
height=416
channels=3      #这三个参数中，要求width==height, 并且为32的倍数，大分辨率可以检测到更加细小的物体，从而影响precision

momentum=0.9      #动量，影响梯度下降到最优的速度，一般默认0.9
decay=0.0005     #权重衰减正则系数，防止过拟合
angle=0          #旋转角度，从而生成更多训练样本 
saturation = 1.5   #调整饱和度，从而生成更多训练样本
exposure = 1.5      #调整曝光度，从而生成更多训练样本
hue=.1         #调整色调，从而生成更多训练样本

learning_rate=0.001  #学习率 ，决定了权值更新的速度，学习率大，更新的就快，但太快容易越过最优值，而学习率太小又更新的慢，效率低，
                     #一般学习率随着训练的进行不断更改，先高一点，然后慢慢降低，一般在0.01--0.001
burn_in=1000            #学习率控制的参数，在迭代次数小于burn_in时，其学习率的更新有一种方式，大于burn_in时，才采用policy的更新方式
max_batches = 50200   #迭代次数，1000次以内，每训练100次保存一次权重，1000次以上，每训练10000次保存一次权重
policy=steps        #学习率策略，学习率下降的方式
steps=40000,45000    #学习率变动步长
scales=.1,.1    #学习率变动因子
                 #如迭代到40000次时，学习率衰减十倍，45000次迭代时，学习率又会在前一个学习率的基础上衰减十倍


[convolutional]
batch_normalize=1    #BN
filters=32    #卷积核数目
size=3       #卷积核尺寸
stride=1    #做卷积运算的步长
pad=1      #如果pad为0,padding由 padding参数指定。
		  #如果pad为1，padding大小为size/2，padding应该是对输入图像左边缘拓展的像素数量
activation=leaky  #激活函数类型


[yolo]
mask = 6,7,8  #使用anchor时使用前三个尺寸
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326  #anchors是可以事先通过cmd指令计算出来的，是和图片数量，width,height以及cluster(就是下面的num的值，
					                                                                   #即想要使用的anchors的数量)相关的预选框，可以手工挑选，也可以通过k-means算法从训练样本中学出

classes=20   #类别 
num=9          #每个grid cell预测几个box,和anchors的数量一致。当想要使用更多anchors时需要调大num，且如果调大num后训练时Obj趋近0的话可以尝试调大object_scale
jitter=.3  #通过抖动来防止过拟合,jitter就是crop的参数
ignore_thresh = .5  #ignore_thresh 指得是参与计算的IOU阈值大小。当预测的检测框与ground true的IOU大于ignore_thresh的时候，参与loss的计算，否则，检测框的不参与损失计算。
                    #目的是控制参与loss计算的检测框的规模，当ignore_thresh过于大，接近于1的时候，那么参与检测框回归loss的个数就会比较少，同时也容易造成过拟合；
				   #而如果ignore_thresh设置的过于小，那么参与计算的会数量规模就会很大。同时也容易在进行检测框回归的时候造成欠拟合。
                   #参数设置：一般选取0.5-0.7之间的一个值，之前的计算基础都是小尺度（13*13）用的是0.7，（26*26）用的是0.5。这次先将0.5更改为0.7。 
truth_thresh = 1
random=1    #如果显存小，设置为0，关闭多尺度训练，random设置成1，可以增加检测精度precision，每次迭代图片大小随机从320到608，步长为32，如果为0，每次训练大小与输入大小一致


```
Jetson nano + yolov5 + TensorRT加速+调用usb摄像头：[https://blog.csdn.net/hahasl555/article/details/116500763](https://blog.csdn.net/hahasl555/article/details/116500763)

YOLO---多个版本的简单认识：[https://www.cnblogs.com/carle-09/p/11326272.html](https://www.cnblogs.com/carle-09/p/11326272.html)
```
(1)测试一张图片---detect
./darknet detect cfg/yolov3.cfg weights/yolov3.weights data/person.jpg
./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights data/person.jpg

(2)测试本地视频---demo
./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights wp_video/person002.mp4

(3)测试usb视频---
./darknet detector demo cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights

(4)测试rstp视频---
./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg weights/yolov3.weights rtsp://admin:admin12345@192.168.?.??/H.264/ch1/sub/av_stream -i 0 -thresh 0.25

./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg weights/yolov3.weights rtsp://admin:admin12345@192.168.?.??/H.264/ch1/sub/av_stream -i 0

-----------to test many pics------------------------
./darknet detect cfg/yolov3.cfg weights/yolov3.weights -i 2 #CPU  enter
Enter Image Path:

./darknet detect cfg/yolov3.cfg weights/yolov3.weights -i 0 #GPU  enter
Enter Image Path: 
```

tensorflow实现YoloV1:[https://github.com/TowardsNorth/yolo_v1_tensorflow_guiyu](https://github.com/TowardsNorth/yolo_v1_tensorflow_guiyu)

使用两种方法创建车辆检测管道：（1）深度神经网络（YOLO 框架）和（2）支持向量机（OpenCV + HOG）：[https://github.com/JunshengFu/vehicle-detection](https://github.com/JunshengFu/vehicle-detection)

[https://pjreddie.com/darknet/](https://pjreddie.com/darknet/)

YOLO（You Only Look Once，包括YOLOv1、YOLOv2、YOLOv3）使用tensorflow，包括/检测和导出pb脚本。将网权重转换为张量流。使用TF_Slim实现YOLO:[https://github.com/Robinatp/YOLO_Tensorflow]

(https://github.com/Robinatp/YOLO_Tensorflow)https://github.com/Robinatp/YOLO_Tensorflow

YOLO 的 TensorFlow 实现，包括训练和测试阶段：[https://github.com/hizhangp/yolo_tensorflow](https://github.com/hizhangp/yolo_tensorflow)

YOLO-V3可视化训练过程中的参数，绘制loss、IOU、avg Recall等的曲线图:[https://blog.csdn.net/qq_34806812/article/details/81459982](https://blog.csdn.net/qq_34806812/article/details/81459982)
