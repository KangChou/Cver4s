import tensorflow as tf
import numpy as np
from core.utils import load_one_dataset
from core.yolo_loss import loss_function
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 推断模式，使用running_mean和running_variance作为bn层的mean、var
# tf.keras.backend.set_learning_phase(False)
# 使用未训练的随机参数的模型计算loss，会得到一个非常大的loss
# model = DarkNet()
# model(tf.keras.layers.Input(shape=(608, 608, 3)))
# print(model.summary())

# 加载模型, 使用已经训练好的模型计算loss 会得到一个非常低的loss
model = tf.saved_model.load('saved_model/overfit_model-1000e')
# 载入训练集合
# (1, 608, 608, 3) 和 (3, 5)
dog_image_data, dog_label, dog_image, dog_image_shape = load_one_dataset('dog', '.jpg')
# (1, 608, 608, 3) 和 (1, 5)
car2_image_data, car2_label, car2_image, car2_image_shape = load_one_dataset('car2', '.png')

# 2个图片样本 (2, 608, 608, 3)
images_data = np.concatenate([dog_image_data, car2_image_data], axis=0)

# 2个图片样本label  (2, ?, 5)
labels = [dog_label, car2_label]

# (Batch,19,19,425)
predictions = model(images_data)

# (Batch, 19, 19, 425) 和 (Batch, ?, 5)
loss = loss_function(predictions, labels)

print('loss:', loss)

