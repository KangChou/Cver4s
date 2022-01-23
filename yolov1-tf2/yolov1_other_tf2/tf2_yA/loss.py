import tensorflow as tf
import numpy as np
from utils import iou

def yolo_loss(predict, labels, each_object_num, num_classes, boxes_per_cell, cell_size, input_width, input_height, coord_scale, object_scale, noobject_scale, class_scale):
  
  #parse only coordinate vector 
  predict_boxes = predict[:,:, num_classes + boxes_per_cell:] # shape = 7*7*8
  predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4]) # shape = 7*7*2*4

  pred_xcenter = predict_boxes[:,:,:,0]
  pred_ycenter = predict_boxes[:,:,:,1]
  pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:,:,:,2]))) # pred_box의 width 값이 input width보다 커지는것을 방지
  pred_sqrt_h = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:,:,:,3])))
  pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32) # float32자료형으로 형 변환
  pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32)

  #parse label(정답값을 구성)
  labels = np.array(labels)
  labels = labels.astype("float32")
  label = labels[each_object_num, :]
  xcenter = label[0]
  ycenter = label[1]
  sqrt_w = tf.sqrt(label[2])
  sqrt_h = tf.sqrt(label[3])

  #calculate iou
  iou_predict_truth = iou(predict_boxes, label[0:4])

  I = iou_predict_truth # 7*7*2
  max_I = tf.reduce_max(I,2,keepdims=True) # cell에 들어있는 2개의 bbox에서 IOU가 더 큰 bbox를 선택함. 
  best_box_mask = tf.cast((I>-max_I), tf.float32) # 정답 mask map 생성

  #set object loss information
  C = iou_predict_truth #정답과 prediction간의 iou를 정답으로 간주
  pred_C = predict[:,:,num_classes:num_classes + boxes_per_cell] # confidence부분 slicing

  # set class loss information
  P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)
  pred_P = predict[:,:,0:num_classes]

  # find object exists cell mask , 정답 xcenter, ycenter를 이용해서 object가 있는 mask map만들기
  object_exists_cell = np.zeros([cell_size, cell_size, 1])
  object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter/input_height), int(cell_size*xcenter/input_width)
  object_exists_cell[object_exists_cell_i][object_exists_cell_j]=1

  # set coord_loss, 
  # object_exists_cell * best_box_mask : 정답 object가 있는 mask map과 best box가 있는 mask map을 연산 = responsible한 bbox coefficient부분을 구현
  coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) + # 절대좌표를 grid cell내의 상대좌표로 변환
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width + # 절대좌표를 전체 그림크기의 좌표로 변환
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) \
               * coord_scale

  # object loss
  object_loss = tf.nn.l2_loss(object_exists_cell*best_box_mask*(pred_C - C))* object_scale #원 논문에는 추가적인 coefficient(object_scale)없음.

  # no object loss
  noobject_loss = tf.nn.l2_loss((1-object_exists_cell)*(pred_C)*noobject_scale) # object가 없는 cell은 정답 confidence가 0

  # prediction class loss
  class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P))*class_scale #원 논문에는 추가적인 coefficient(object_scale)없음.

  #sum loss
  total_loss = coord_loss + object_loss + noobject_loss + class_loss

  return total_loss, coord_loss, object_loss, noobject_loss, class_loss # 개별 loss들 return하여 tensorboard로 시각화
