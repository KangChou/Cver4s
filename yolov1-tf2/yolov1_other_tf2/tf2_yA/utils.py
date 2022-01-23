import cv2
import numpy as np
import tensorflow as tf
import colorsys
from operator import itemgetter

def draw_bounding_box_and_label_info(frame, x_min, y_min, x_max, y_max, label, confidence, color):
  draw_bounding_box(frame, x_min, y_min, x_max, y_max, color)
  draw_label_info(frame, x_min, y_min, label, confidence, color)


def draw_bounding_box(frame, x_min, y_min, x_max, y_max, color):
  cv2.rectangle(
    frame,
    (x_min, y_min),
    (x_max, y_max),
    color, 3)


def draw_label_info(frame, x_min, y_min, label, confidence, color):
  text = label + ' ' + str('%.3f' % confidence)
  bottomLeftCornerOfText = (x_min, y_min)
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 0.8
  fontColor = color
  lineType = 2

  cv2.putText(frame, text,
              bottomLeftCornerOfText,
              font,
              fontScale,
              fontColor,
              lineType)

# nms 대신에 confidence가 가장 높은 box 한 개만 추출
def find_max_confidence_bounding_box(bounding_box_info_list):
  bounding_box_info_list_sorted = sorted(bounding_box_info_list,
                                                   key=itemgetter('confidence'),
                                                   reverse=True)
  max_confidence_bounding_box = bounding_box_info_list_sorted[0]

  return max_confidence_bounding_box

# predicted yolo format을 다시 xmin~ 형태로 변환
def yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, class_name, confidence):
  bounding_box_info = {}
  bounding_box_info['left'] = int(xcenter - (box_w / 2))
  bounding_box_info['top'] = int(ycenter - (box_h / 2))
  bounding_box_info['right'] = int(xcenter + (box_w / 2))
  bounding_box_info['bottom'] = int(ycenter + (box_h / 2))
  bounding_box_info['class_name'] = class_name
  bounding_box_info['confidence'] = confidence

  return bounding_box_info

def iou(yolo_pred_boxes, ground_truth_boxes):
  # Reference : https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/yolo_tiny_net.py#L105
  
  boxes1 = yolo_pred_boxes
  boxes2 = ground_truth_boxes

  boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                     boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
  boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
  boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                     boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
  boxes2 = tf.cast(boxes2, tf.float32)

  # calculate the left up point
  lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
  rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

  # intersection
  intersection = rd - lu

  inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

  mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

  inter_square = mask * inter_square

  # calculate the boxs1 square and boxs2 square
  square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
  square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

  return inter_square / (square1 + square2 - inter_square + 1e-6)


def generate_color(num_classes):
  # Reference : https://github.com/qqwweee/keras-yolo3/blob/e6598d13c703029b2686bc2eb8d5c09badf42992/yolo.py#L82
  # Generate colors for drawing bounding boxes.
  hsv_tuples = [(x / num_classes, 1., 1.)
                for x in range(num_classes)]
  colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
  colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
  np.random.seed(10101)  # Fixed seed for consistent colors across runs.
  np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
  np.random.seed(None)  # Reset seed to default.

  return colors