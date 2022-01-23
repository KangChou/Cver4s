import tensorflow as tf
import numpy as np

def bounds_per_dimension(ndarray):
  return map(
    lambda e: range(e.min(), e.max() + 1),
    np.where(ndarray != 0)
  )


def zero_trim_ndarray(ndarray):
  return ndarray[np.ix_(*bounds_per_dimension(ndarray))]

def process_each_ground_truth(original_image, bbox, class_labels, input_width, input_height):
  
  image = original_image.numpy()
  image = zero_trim_ndarray(image) # 이미지의 크기를 맞추기위해 zero패딩 처리되어있는 부분을 모두 지워버리는 함수.

  # set original width height
  original_h = image.shape[0]
  original_w = image.shape[1]

  width_rate = input_width * 1.0 / original_w
  height_rate = input_width * 1.0 / original_h

  # yolo에 들어가는 고정 사이즈로 resizing
  image = tf.image.resize(image, [input_height, input_width])

  # batch단위로 들어오는 data의 bbox의 배열은 동일한 shape을 갖도록 부족한 bbox는 패딩처리되어 있음. 이 배열에서 정확한 object의 수를 count_nonzero로 counting함
  object_num = np.count_nonzero(bbox, axis=1)[0]
  #initialize labels
  labels = [[0,0,0,0,0]] * object_num
  for i in range(object_num):
    # bbox는 상대좌표로 들어오기 때문에 절대좌표로 변환해줌
    xmin = bbox[i][1] * original_w
    ymin = bbox[i][0] * original_h
    xmax = bbox[i][3] * original_w
    ymax = bbox[i][2] * original_h

    class_num = class_labels[i]

    #voc format을 yolo format으로 변환
    xcenter = (xmin+xmax) * 1.0 / 2 * width_rate
    ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

    box_w = (xmax - xmin) * width_rate
    box_h = (ymax - ymin) * height_rate

    labels[i] = [xcenter, ycenter, box_w, box_h, class_num]

  return [image.nupy(), labels, object_num]
    

