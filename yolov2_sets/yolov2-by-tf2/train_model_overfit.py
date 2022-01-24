from core.yolo_model import DarkNet
import tensorflow as tf
import numpy as np
from core.utils import load_one_dataset
from core.yolo_loss import loss_function
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
训练只有2个样本的训练集，得出一个过拟合模型，并用来检测。
'''


# 训练网络时，由于存在BN层，BN层存在不可训练参数mean和variance
# 需要显示的告诉网络处于训练模式，因此在训练的每个batch过程中，网络会根据mean和variance更新running_mean和running_variance
# 在推断模式中，将使用running_mean和running_variance作为bn层的mean和variance
tf.keras.backend.set_learning_phase(True)
# 新建模型，随机初始化参数
model = DarkNet()
# 下面这行代码必须要，不然save model 后，再load model，model是not callable的
model(tf.keras.layers.Input(shape=(608, 608, 3)))
# print(model.summary())

# 载入训练集合
# (1, 608, 608, 3) 和 (3, 5)
dog_image_data, dog_label, dog_image, dog_image_shape = load_one_dataset('dog', '.jpg')
# (1, 608, 608, 3) 和 (1, 5)
car2_image_data, car2_label, car2_image, car2_image_shape = load_one_dataset('car2', '.png')

# 2个图片样本 (2, 608, 608, 3)
images_data = np.concatenate([dog_image_data, car2_image_data], axis=0)

# 2个图片样本label  (2, ?, 5)
labels = [dog_label, car2_label]

# 因为只有两个样本，所以batch-gradient-descent
# 因为样本极少，显然会过拟合，最终会有一个极小的loss

# 整个样本的迭代次数
num_epoths = 1000
# 学习速率
learning_rate = 0.001
# 设置梯度更新器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
s0 = time.time()
cost = 0
for i in range(num_epoths):
    s1 = time.time()
    with tf.GradientTape() as tape:
        # 使用向前传播算法计算预测结果
        predictions = model(images_data)
        # 计算预测结果和真实label之间的loss
        loss = loss_function(predictions, labels)
    print('迭代: ', i, ', 耗时: ', cost, 's, loss: ', loss.numpy())
    # 进行反向传播算法计算损失函数关于可训练权重参数的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 使用梯度下降算法，更新可训练权重参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    cost = time.time() - s1  # 记录每个epoth的耗时

s3 = time.time()

predictions = model(images_data)
final_loss = loss_function(predictions, labels)
print('迭代: ', num_epoths, ', 耗时: ', cost, 's, loss: ', final_loss.numpy())
# 计算在推断模式下的loss
tf.keras.backend.set_learning_phase(False)

# 计算在推断模式下的loss
predictions = model(images_data)
inference_loss = loss_function(predictions, labels)
print('共迭代', num_epoths, '次, 总耗时: ', s3 - s0, 's, 最终Loss: ', final_loss.numpy(), ', 推断模式下Loss: ',
      inference_loss.numpy())

# 保存过拟合模型
tf.saved_model.save(model, 'saved_model/overfit_model-1000e')


# 共迭代 1000 次, 总耗时:  10817.50330901146 s, 最终Loss:  0.0010638987 , 推断模式下Loss:  0.0068158614


