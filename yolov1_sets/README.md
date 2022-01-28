yolov1：[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)

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
