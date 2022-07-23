

论文：https://arxiv.org/abs/2207.02696
源码：https://github.com/wongkinyiu/yolov7

复现过程：https://blog.csdn.net/weixin_41194129/article/details/125950025

![image](https://user-images.githubusercontent.com/36963108/180600683-09bea2c5-f40d-42f9-b293-cda906a42bca.png)


```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt


pip install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple

python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val


wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg

python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source zidane.jpg

下载视频：https://github.com/KangChou/Cver4s/blob/main/toolscv/video.mp4

python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source video.mp4

```


other：

- https://github.com/AlexeyAB/darknet
- https://github.com/WongKinYiu/yolor
- https://github.com/WongKinYiu/PyTorch_YOLOv4
- https://github.com/WongKinYiu/ScaledYOLOv4
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/ultralytics/yolov3
- https://github.com/ultralytics/yolov5
- https://github.com/DingXiaoH/RepVGG
- https://github.com/JUGGHM/OREPA_CVPR2022
- https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose
