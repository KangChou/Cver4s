from core.yolo_model import DarkNet
from core.utils import parse_yolo_v2_model_weights, preprocess_image, \
    draw_boxes, read_class_name, generate_colors, detect
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# tf.keras.backend.set_learning_phase(False)
model = DarkNet()
# 编译网络，生成weights参数结构
model(tf.keras.layers.Input(shape=(608, 608, 3)))

print(model.summary())
# 解析权重参数
yolo_weights = parse_yolo_v2_model_weights(model.weight_shapes, 'yolov2_weights/yolov2.weights')
# 将权重参数导入模型
model.set_weights(yolo_weights)
# 80个类别的名称
class_name = read_class_name('./model_data/coco_classes_chinese.txt')

root_path = 'images'
# 检测文件夹下的所有图片，深度为1
for f in os.listdir(root_path):
    nf = os.path.join(root_path, f)
    if os.path.isfile(nf):
        # 图片放缩成608,608,像素值归一化
        image, image_data, image_shape = preprocess_image(img_path=nf)
        # 进行目标检测算法
        res_class, res_score, res_boxes = detect(image_data, model)
        # image_shape 为原图大小高宽格式，
        # 还原成相对于原图的位置
        res_boxes = res_boxes * np.tile(list(image_shape), 2)
        # 把框画在图片上
        draw_boxes(image, res_score, res_boxes, res_class, class_name, generate_colors(class_name))
        # plt.imshow(image)
        # plt.show()
        # 保存图片
        image.save('images/out/' + f, quality=100)
