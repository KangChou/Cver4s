from core.yolo_model import DarkNet
from core.utils import parse_yolo_v2_model_weights
import tensorflow as tf
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = DarkNet()
# 编译网络，生成weights参数结构
model(tf.keras.layers.Input(shape=(608, 608, 3)))
# 打印出网络结构
print(model.summary())
# 解析权重参数
yolo_weights = parse_yolo_v2_model_weights(model.weight_shapes, 'yolov2_weights/yolov2.weights')
# 将权重参数导入模型
model.set_weights(yolo_weights)
# 导出模型
tf.saved_model.save(model, 'saved_model/yolo_model_file')
