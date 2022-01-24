import tensorflow as tf
from core.utils import preprocess_image, \
    draw_boxes, read_class_name, generate_colors, detect
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载模型
model = tf.saved_model.load('saved_model/yolo_model_file')
# 80个类别的名称
class_name = read_class_name('./model_data/coco_classes_chinese.txt')
# 图片名称
file = 'dog.jpg'
# 图片放缩成608,608,像素值归一化
image, image_data, image_shape = preprocess_image(img_path='images/' + file)
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
image.save('images/out/' + file, quality=100)
