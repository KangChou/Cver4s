import tensorflow as tf
import numpy as np
from core.utils import generateXYOffset


def loss_function(predictions, labels):
    """
    predictions: D-CNN 的输出，维度是(Batch, 19, 19, 425)
    labels:      人工标签，    维度是(Batch, ?, 5)
    """
    if len(predictions) != len(labels):
        raise Exception("predictions和labels的数量不一致")
    # 首先将predictions转换成(Batch, 19, 19, 5, 85)
    predictions = tf.reshape(predictions, [-1, 19, 19, 5, 85])
    # 样本数量
    m = len(labels)
    '''
    为了计算loss，需要将labels转换成(Batch, 19, 19, 5, 85)维度
    创建bool 类型的 (Batch, 19, 19, 5) detectors_mask，为True的位置表示存在对象，对象的(x,y,w,h,class)存储在相应labels2的位置，
    '''
    # 每个单元格有5个锚框5个shape，维度为 (5,2)
    # anchor_box_shape = readAnchorBoxShape('./model_data/yolo_anchors.txt')
    # 单位也是单元格大小 维度为(5,2)
    anchor_box_shape = np.array(
        [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778],
         [9.77052, 9.16828]])
    # (Batch, 19, 19, 5), (Batch, 19, 19, 5, 5)
    detectors_mask, labels2 = convert(labels, anchor_box_shape)

    # 将xy经过sigmoid映射，(Batch, 19, 19, 5, 85)
    predictions = tf.keras.backend.concatenate([tf.sigmoid(predictions[..., 0:2]), predictions[..., 2:85]], axis=-1)

    # (1, 1, 5, 1, 2)
    fixed_anchor_box_shape = np.expand_dims(anchor_box_shape, axis=[0, 1, 3])

    # 计算object_detections为(Batch, 19, 19, 5)真值张量，网络输出框与目标框IOU>=0.6 则为True
    object_detections = []
    i = -1
    for label in labels:  # 遍历每个样本
        i += 1
        '''
        label 维度 (K,5)
        pred  维度 (19, 19, 5, 4)
        '''
        pred = predictions[i][..., 0:4]
        # (19, 19, 5, 1, 4)
        pred = tf.expand_dims(pred, axis=3)
        # (1,  1,  1, K, 4)
        label = np.expand_dims(label, axis=[0, 1, 2])[..., 0:4]

        # 计算每个预测框与每个真实目标框的IOU

        # 将单位转换成单元格大小
        label = np.multiply(label, [19, 19, 19, 19])
        # (1,  1,  1, K, 2)
        xy_label = label[..., 0:2]
        # (1,  1,  1, K, 2)
        wh_label = label[..., 2:4]
        # 19, 19, 2 -> 19, 19, 1, 1, 2
        xy_offset = np.expand_dims(generateXYOffset(), axis=[2, 3])
        # (19, 19, 5, 1, 2) + (19, 19, 1, 1, 2) = (19, 19, 5, 1, 2)，单位转换单元格大小
        xy = pred[..., 0:2] + xy_offset
        # (19, 19, 5, 1, 2) * (1, 1, 5, 1, 2) = (19, 19, 5, 1, 2)，单位转换单元格大小
        wh = tf.exp(pred[..., 2:4]) * fixed_anchor_box_shape

        # 接下来转换成边角坐标
        # 标签转换
        wh_label_half = np.divide(wh_label, 2)
        xy_label_min = xy_label - wh_label_half
        xy_label_max = xy_label + wh_label_half

        # 预测转换
        wh_half = np.divide(wh, 2)
        xy_min = xy - wh_half
        xy_max = xy + wh_half
        # (19, 19, 5, 1, 2), (1,  1,  1, K, 2) =  (19, 19, 5, K, 2)
        intersection_min = np.maximum(xy_min, xy_label_min)
        intersection_max = np.minimum(xy_max, xy_label_max)
        # (19, 19, 5, K, 2)
        intersection_wh = np.maximum(intersection_max - intersection_min, 0)
        # (19, 19, 5, K) , 含义是每个预测框和每个目标框的交集 ， 19, 19, 5 * K 个结果
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        # (19, 19, 5, 1)
        s1 = wh[..., 0] * wh[..., 1]
        # (1,  1,  1, K)
        s2 = wh_label[..., 0] * wh_label[..., 1]
        # (19, 19, 5, 1) + (1,  1,  1, K) - (19, 19, 5, K) = (19, 19, 5, K)
        union = s1 + s2 - intersection
        # (19, 19, 5, K) / (19, 19, 5, K) = (19, 19, 5, K), 含义是每个预测框和每个目标框的IOU ， 19, 19, 5 * K 个结果
        iou = intersection / union
        # (19, 19, 5), 得到每个检测框与目标框的最佳iou
        best_iou = np.max(iou, axis=-1, keepdims=False)
        # 如果IOU大于0.6,表示这个格子的anchor-box识别到了我们人工标记的对象
        # 19, 19, 5
        each_object_detections = best_iou > 0.6
        # 1, 19, 19, 5
        object_detections.append(np.expand_dims(each_object_detections, axis=0))
    # (Batch, 19, 19, 5)
    object_detections = np.concatenate(object_detections, axis=0)
    # 包含目标损失权重
    obj_scale = 5
    # 无目标损失权重
    no_obj_scale = 1
    # 类别损失权重
    class_scale = 1
    # 坐标损失权重
    coordinate_scale = 1
    # 预测置信度 (Batch, 19, 19, 5)
    predictions_confidence = tf.sigmoid(predictions[..., 4])
    # 预测的类别概率分布 (Batch, 19, 19, 5, 80)
    predictions_classes_prob = tf.nn.softmax(predictions[..., 5:85])
    # 预测的坐标位置，xy相对单元格的位置，wh为anchor-box系数的对数,维度(Batch, 19, 19, 5, 4)
    predictions_coordinate = predictions[..., 0:4]

    # (Batch, 19, 19, 5)
    label_class = tf.cast(labels2[..., 4], dtype='int32')
    # 人工标记的类别 (Batch, 19, 19, 5, 80)
    label_class_prob = tf.one_hot(label_class, 80)
    # 人工标记的坐标,经过调整，xy相对单元格的位置，wh为anchor-box系数的对数,维度(Batch, 19, 19, 5, 4)
    label_coordinate = labels2[..., 0:4]

    # (Batch, 19, 19, 5) * (Batch, 19, 19, 5) * (Batch, 19, 19, 5) = (Batch, 19, 19, 5)
    no_obj_conf_loss = no_obj_scale * (1 - detectors_mask) * (1 - object_detections) * tf.square(
        0 - predictions_confidence)
    # (Batch, 19, 19, 5) * (Batch, 19, 19, 5) = (Batch, 19, 19, 5)
    obj_conf_loss = obj_scale * detectors_mask * tf.square(1 - predictions_confidence)
    # 计算识别loss，维度是(Batch, 19, 19, 5)，包含了每个单元格每个anchor-box的置信度loss
    confidence_loss = no_obj_conf_loss + obj_conf_loss
    # (Batch, 19, 19, 5, 1)
    detectors_mask2 = np.expand_dims(detectors_mask, axis=-1)
    # 计算分类loss，只考虑有对象的位置，(Batch, 19, 19, 5, 1) * (Batch, 19, 19, 5, 80) =  (Batch, 19, 19, 5, 80)
    classification_loss = class_scale * detectors_mask2 * tf.square(label_class_prob - predictions_classes_prob)
    # 各个类别的loss求和， 维度是 (Batch, 19, 19, 5)，包含了每个单元格每个anchor-box的分类loss
    classification_loss = tf.reduce_sum(classification_loss, axis=-1, keepdims=False)
    # 计算定位loss，(Batch, 19, 19, 5, 1) * (Batch, 19, 19, 5, 4) = (Batch, 19, 19, 5, 4)
    coordinate_loss = coordinate_scale * detectors_mask2 * tf.square(label_coordinate - predictions_coordinate)
    # x,y,w,h的loss求和， 维度是 (Batch, 19, 19, 5)，包含了每个单元格每个anchor-box的定位loss
    coordinate_loss = tf.reduce_sum(coordinate_loss, axis=-1, keepdims=False)

    # 总loss为 识别loss+分类loss+定位loss , 维度为 (Batch, 19, 19, 5)，包含了每个单元格每个anchor-box的loss
    total_loss = confidence_loss + classification_loss + coordinate_loss
    # print(tf.reduce_sum(total_loss, axis=[-1, -2, -3]))

    # 计算所有样本的平均loss
    loss = 0.5 / m * tf.reduce_sum(total_loss)
    return loss


def convert(labels, anchor_box_shapes):
    # labels   人工标签，    维度是(Batch, ?, 5)

    # 样本数量
    m = len(labels)
    # 创建bool 类型的 (Batch, 19, 19, 5) detectors_mask，为True的位置表示存在对象
    detectors_mask = np.zeros([m, 19, 19, 5])
    # 创建(Batch, 19, 19, 5, 5) labels2，detectors_mask[b,i,j,a]为True的位置,表示labels2[b,i,j,a]存储了目标的(x,y,w,h,class)信息
    labels2 = np.zeros([m, 19, 19, 5, 5])

    n = -1
    # 把labels中的值填入到 labels2和detectors_mask之中
    # 遍历每张图片
    for label in labels:
        n += 1  # 当前处理的样本索引
        # label 维度为(K,5)
        # 遍历每张图片里面的目标
        for obj in label:
            # obj 维度为(5,)
            # 获取每个目标的位置类别信息
            xy = obj[0:2]  # (2,),范围0～1
            wh = obj[2:4]  # (2,),范围0～1
            table_size = [19, 19]  # (2,)

            # 转换单位，单位是单元格大小
            xy = np.multiply(xy, table_size)  # (2,),范围0～19
            wh = np.multiply(wh, table_size)  # (2,),范围0～19
            # 计算目标中心所在的单元格位置
            i = tf.cast(tf.floor(xy[1]), 'int32')  # y 方向的偏移量
            j = tf.cast(tf.floor(xy[0]), 'int32')  # x 方向的偏移量

            # 接下来找出与目标形状最相似的anchor-box
            # 计算目标与每个anchor-box的交并比
            # 1,2
            obj_wh = np.expand_dims(wh, axis=0)
            obj_xy_max = np.divide(obj_wh, 2)
            obj_xy_min = -obj_xy_max
            # 5,2
            anchor_box_xy_max = np.divide(anchor_box_shapes, 2)
            anchor_box_xy_min = -anchor_box_xy_max

            # 计算交集
            intersection_min = np.maximum(obj_xy_min, anchor_box_xy_min)
            intersection_max = np.minimum(obj_xy_max, anchor_box_xy_max)
            intersection_wh = np.maximum((intersection_max - intersection_min), 0)
            # s = w * h, 维度为(5,)
            intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
            # 计算并集
            # 维度为(1,)
            s1 = obj_wh[:, 0] * obj_wh[:, 1]
            # 维度为(5,)
            s2 = anchor_box_shapes[:, 0] * anchor_box_shapes[:, 1]
            # (1,) + (5,) - (5,) = (5,)
            union = s1 + s2 - intersection
            # 计算iou
            iou = intersection / union
            # 得出形状最符合的anchor-box索引
            a = np.argmax(iou)

            if detectors_mask[n, i, j, a]:
                raise Exception('人工标注数据有误，同一个位置存在多次标记')

            # 标记该位置存在对象
            detectors_mask[n, i, j, a] = True
            # 在相应的位置存储目标位置类别信息
            labels2[n, i, j, a] = [
                xy[0] - j,  # x  相对单元格的位置
                xy[1] - i,  # y  相对单元格的位置
                # 我们没有让网络直接学习目标的宽高，而是对应的anchor-box高宽系数的对数
                np.log(wh[0] / anchor_box_shapes[a][0]),  # w  让网络学习anchor-box宽度系数的对数
                np.log(wh[1] / anchor_box_shapes[a][1]),  # h  让网络学习anchor-box高度系数的对数
                obj[4]
            ]

    return detectors_mask, labels2

# 4 14 3
# 9 7 4
# 12 5 4
# 12 8 4
