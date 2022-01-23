import tensorflow as tf
from tensorflow.python.training.tracking import base

class YOLOv1(tf.keras.Model):
  def __init__(self, input_height, input_width, cell_size, boxes_per_cell, num_classes):
    super(YOLOv1, self).__init__()
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(input_height, input_width, 3))
    base_model.trainable = True # backbone parameter도 튜닝 허용함.
    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x) # feature map을 하나의 scalar vector로 pooling함 (1차원)
    output = tf.keras.layers.Dense(cell_size*cell_size*(num_classes + (boxes_per_cell *5)), activation=None)(x)
    
    model = tf.keras.Model(inputs = base_model.input, outputs = output)

    self.model = model

    self.model.summary()

  def call(self, x):
    return self.model(x)

