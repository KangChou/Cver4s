import tensorflow as tf
from core.utils import preprocess_image, \
    draw_boxes, read_class_name, generate_colors, detect, write_raw_label
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载模型
model = tf.saved_model.load('saved_model/yolo_model_file')
# 80个类别的名称
class_name = read_class_name('./model_data/coco_classes_chinese.txt')
# 图片名称
file = 'car2'
suffix = '.png'

# 图片放缩成608,608,像素值归一化
image, image_data, image_shape = preprocess_image(img_path='images/' + file + suffix)
image.save('train-set/origin/' + file + suffix, quality=100)
# 进行目标检测算法
res_class, res_score, res_boxes = detect(image_data, model)

'''
生成训练集的方法
用网络检测图片，将检测结果作为标签，对于一张图片生成(K,5)的张量，K为目标数量，x, y, w, h, class为目标位置和类别
'''

# 检测结果是边角坐标`[y1, x1, y2, x2]`，所以要转换成x, y, w, h

# 计算高宽
hw = res_boxes[:, 2:4] - res_boxes[:, 0:2]
hw_half = hw / 2
# 计算目标中心
yx = res_boxes[:, 0:2] + hw_half
# 转换成x, y, w, h格式
xywh = tf.keras.backend.concatenate([
    yx[:, 1:2],
    yx[:, 0:1],
    hw[:, 1:2],
    hw[:, 0:1],
], axis=-1)
print(xywh)
# x, y, w, h, label
label = tf.keras.backend.concatenate([xywh, tf.expand_dims(tf.cast(res_class, dtype='float32'), axis=1)])
print(label)

# 将图片标签保存到文件
write_raw_label(label.numpy(),
                'train-set/origin/labeled-' + file + '.txt')

# image_shape 为原图大小高宽格式，
# 还原成相对于原图的位置
res_boxes = res_boxes * np.tile(list(image_shape), 2)
# 把框画在图片上
draw_boxes(image, res_score, res_boxes, res_class, class_name, generate_colors(class_name))

image.save('train-set/labeled/' + file + suffix, quality=100)
