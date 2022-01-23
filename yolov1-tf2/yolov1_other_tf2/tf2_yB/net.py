import tensorflow as tf
import os
from tensorflow.keras import layers,Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
yolo_net = Sequential([
    layers.Conv2D(192,kernel_size=[7,7],strides=2,padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2]),
    layers.Conv2D(256,kernel_size=[3,3],strides=2,padding='same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],strides=2,padding='same',activation=tf.nn.relu),
    layers.Conv2D(1024,kernel_size=[3,3],strides=2,padding='same',activation=tf.nn.relu),
    layers.Flatten(),
    layers.Dense(4096),
    layers.Dense(7*7*30)
])

