import tensorflow as tf
from core.utils import preprocess_image, \
    draw_boxes, read_class_name, generate_colors, detect
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 在推断模式中，将使用running_mean和running_variance作为bn层的mean和variance
tf.keras.backend.set_learning_phase(False)
# 加载模型
model = tf.saved_model.load('saved_model/overfit_model-1000e')

# 80个类别的名称
class_name = read_class_name('./model_data/coco_classes_chinese.txt')
# 图片名称
file = 'car2.png'
# 图片放缩成608,608,像素值归一化
image, image_data, image_shape = preprocess_image(img_path='train-set/origin/' + file)
# 进行目标检测算法
res_class, res_score, res_boxes = detect(image_data, model)
if len(res_boxes) > 0:
    # image_shape 为原图大小高宽格式，
    # 还原成相对于原图的位置
    res_boxes = res_boxes * np.tile(list(image_shape), 2)
    # 把框画在图片上
    draw_boxes(image, res_score, res_boxes, res_class, class_name, generate_colors(class_name))
    plt.imshow(image)
    plt.show()
    image.save('train-set/detected/'+file, quality=100)
else:
    print('未检测到任何目标')

