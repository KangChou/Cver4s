import numpy as np
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import random
import colorsys


def parse_yolo_v2_model_weights(weight_shapes, weights_path):
    weights_file = open(weights_path, 'rb')
    # 丢弃前16个字节
    weights_header = np.ndarray(
        shape=(4,), dtype='int32', buffer=weights_file.read(16))
    print(weights_header)
    # yolo v2 模型weights保存的格式为 [bias / beta, [gamma, mean, variance], conv_weights]
    # tensorflow 模型weights格式为  [conv_weights , [gamma, beta, mean, variance] / bias]
    yolo_weights = []
    # 统计解析了多少参数
    count = 0
    for i in range(len(weight_shapes)):
        conv_weights_shape = weight_shapes[i]
        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        yolo_weights_shape = (conv_weights_shape[-1], conv_weights_shape[-2],
                              conv_weights_shape[0], conv_weights_shape[1])
        print('parse shape', conv_weights_shape)
        print('channels', weight_shapes[i][-1])
        channels = weight_shapes[i][-1]
        # 乘起来计算一共有多少个weight参数
        weights_size = np.product(conv_weights_shape)

        conv_bias = np.ndarray(
            shape=(channels,), dtype='float32', buffer=weights_file.read(channels * 4))
        args = [conv_bias]
        if i < len(weight_shapes) - 1:
            # 如果不是最后一层卷积，那么都采用了bn
            bn_args = np.ndarray(
                shape=(3, channels), dtype='float32', buffer=weights_file.read(3 * channels * 4))
            args = [
                bn_args[0],  # scale gamma
                conv_bias,  # shift gamma
                bn_args[1],  # running mean
                bn_args[2],  # running var
            ]

        yolo_weight = np.ndarray(
            shape=yolo_weights_shape, dtype='float32', buffer=weights_file.read(weights_size * 4))
        conv_weights = np.transpose(yolo_weight, [2, 3, 1, 0])

        count += weights_size
        yolo_weights.append(conv_weights)
        for j in range(len(args)):
            yolo_weights.append(args[j])
            count += len(args[j])
    remaining_args = len(weights_file.read()) // 4
    print('读取了 ', count, '/', count + remaining_args, ' 参数')
    weights_file.close()
    return yolo_weights


def show_weights_shape(weights):
    for i in range(len(weights)):
        sub_weights = weights[i]
        print(np.array(sub_weights).shape)


def preprocess_image(img_path, model_image_size=(608, 608)):
    resized_image, image, image_data, image_shape = preprocess_image0(img_path, model_image_size)
    # 丢弃第一个参数
    return image, image_data, image_shape


def preprocess_image0(img_path, model_image_size):
    image = Image.open(img_path)

    # 将图像缩放成固定大小608 608
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)

    # 得到图片的像素值，608*608*3
    image_data = np.array(resized_image, dtype='float32')
    # 数据归一化处理
    image_data /= 255.
    # 给数据添加一个新维度，数据量维度，得到1*608*608*3，是适配网络的格式
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    # 返回的image shape是(h, w)格式, 图片可能是4通道，但是只考虑rgb通道
    return resized_image, image, image_data[:, :, :, 0:3], tuple(reversed(image.size))


def readAnchorBoxShape(path):
    with open(path) as f:
        s = f.readline().strip()
        anchors = [float(x) for x in s.split(',')]
        # 5,2
        anchor_box_shape = np.reshape(anchors, [-1, 2])
    return anchor_box_shape


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def write_raw_label(labels, label_file):
    """
        labels 是一张图片的人工标签，维度为(K,5)，
        K为目标数量，5表示x，y，w，h，class位置类别信息
    """
    f = open(label_file, 'w')
    for iv in labels:
        s = ""
        for jv in iv:
            s += str(jv) + ","
        s += "\n"
        f.write(s)
    f.close()


def read_class_name(path):
    with open(path) as f:
        lines = f.readlines()
        # 删除一头一尾的空白
        class_name = []
        for x in lines:
            t = x.strip()
            # 忽略空行
            if len(t) > 0:
                class_name.append(t)
    return class_name


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    font = ImageFont.truetype(font='./font/SourceHanSansSC-Bold.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw


def generateXYOffset():
    """
    生成偏移量 19,19,2
    0,0   1,0   2,0   3,0  ...   18,0
    0,1   1,1   2,1   3,1  ...   18,1
                           ...
    0,18  1,18  2,18  3,18 ...   18,18
    由两组数据组成，
    0    1   2   3  ...   w-1
    0    1   2   3  ...   w-1
                    ...

    0    0
    1    1
    2    2
    3    3
    ...  ...  ...
    h-1  h-1

    更一般来讲是0...w-1往下重复h次 , (0...h-1)T 往右重复w次
    """
    w = 19
    h = 19

    A = np.tile(np.reshape(np.arange(0, w, dtype='float32'), [1, w]), [h, 1])
    B = np.tile(np.reshape(np.arange(0, h, dtype='float32'), [h, 1]), w)
    A = np.expand_dims(A, axis=-1)
    B = np.expand_dims(B, axis=-1)
    C = np.concatenate([A, B], axis=-1)
    return C


def non_max_suppression(box_high_scores_class, box_high_scores, high_scores_boxes, threshold=0.5):
    unique_class = tf.unique(box_high_scores_class)
    classified_scores = []
    classified_boxes = []
    for i in range(len(unique_class.y)):
        p = []
        q = []
        index = 0
        # 收集第i个类的所有相关数据
        for j in unique_class.idx:
            if i == j:
                p.append(box_high_scores[index].numpy())
                q.append(high_scores_boxes[index].numpy().tolist())
            index += 1
        # 加上有n个类别
        # 维度是 n,?
        classified_scores.append(p)
        # 维度是 n,?,4
        classified_boxes.append(q)
    res_class = []
    res_score = []
    res_boxes = []

    # 分类进行非最大印制算法
    for i in range(len(unique_class.y)):
        box_index = tf.image.non_max_suppression(
            classified_boxes[i], classified_scores[i], max_output_size=30, iou_threshold=threshold)
        classified_res_score = tf.gather(classified_scores[i], box_index)
        classified_res_boxes = tf.gather(classified_boxes[i], box_index)
        classified_res_class = tf.tile(unique_class.y[i:i + 1], tf.shape(box_index))
        res_score.append(classified_res_score)
        res_boxes.append(classified_res_boxes)
        res_class.append(classified_res_class)
    if len(res_score) == 0:
        print('未检测到任何目标')
    else:
        res_score = tf.keras.backend.concatenate(res_score, axis=0)
        res_boxes = tf.keras.backend.concatenate(res_boxes, axis=0)
        res_class = tf.keras.backend.concatenate(res_class, axis=0)
    return res_class, res_score, res_boxes


# 转换数据，过滤低得分框，各个类非最大值印制，得到网络最终的预测框
def convert_filter_and_non_max_suppression(pred):
    # pred为网络输出，N,19,19,425
    # N,19,19,5,85
    predictions = tf.reshape(pred, [-1, 19, 19, 5, 85])
    # 转换成更有意义的值
    # 相对单元格的位置
    # N,19,19,5,2
    box_xy = tf.sigmoid(predictions[:, :, :, :, 0:2])
    # 相对anchor-box宽高的系数
    # N,19,19,5,2
    box_wh = tf.exp(predictions[:, :, :, :, 2:4])
    # 该位置包含对象的把握
    # N,19,19,5,1
    box_conf = tf.sigmoid(predictions[:, :, :, :, 4:5])
    # 该位置包含的对象关于类别的概率分布
    # N,19,19,5,80
    box_class_prob = tf.nn.softmax(predictions[:, :, :, :, 5:85])
    # 再转换
    # 19,19,2      生成每个单元格相对整张表格的偏移
    xy_offset = generateXYOffset()
    # 1,19,19,1,2  同一个单元格下的所有anchor-box共用偏移
    xy_offset2 = np.expand_dims(xy_offset, axis=[0, -2])
    # 加上偏移，便得到相对整张表格的位置
    box_xy = box_xy + xy_offset2
    # 位置单位是每个单元格的长度，除整张表格的长度，得出一个比例0～1，相对整张表格的位置，或者说相对整张图的位置
    # N,19,19,5,2
    box_xy = box_xy / [19, 19]
    # 每个单元格的5个锚框的形状，维度为 (5,2)
    # anchor_box_shapes = readAnchorBoxShape('./model_data/yolo_anchors.txt')
    anchor_box_shapes = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778],
                         [9.77052, 9.16828]]
    # 转换层相应的格式
    fixed_anchor_box_shape = np.reshape(anchor_box_shapes, [1, 1, 1, 5, 2])
    # box_wh是anchor—box宽高的系数，乘积得到真实的宽高，单位为单元格的长度
    box_wh = box_wh * fixed_anchor_box_shape
    # 得出一个比例0～1，宽高分别是相对整张表格的宽高
    box_wh = box_wh / [19, 19]
    # 转换成边角坐标
    box_wh_half = box_wh / 2
    xy_min = box_xy - box_wh_half
    xy_max = box_xy + box_wh_half
    # N,19,19,5,4
    boxes = tf.keras.backend.concatenate([
        xy_min[:, :, :, :, 1:2],  # y_min
        xy_min[:, :, :, :, 0:1],  # x_min
        xy_max[:, :, :, :, 1:2],  # y_max
        xy_max[:, :, :, :, 0:1],  # x_max
    ], axis=-1)
    # 计算出每个类别的得分，   实际上是转换成81类别的概率分布，其中80个是预定义类别，另外一个是不包含对象的概率
    # N,19,19,5,1 *  N,19,19,5,80 -> N,19,19,5,80
    box_scores = box_conf * box_class_prob
    # 得分最高的类别当作该盒子的预测类别
    # N,19,19,5   包含最高得分的类别
    box_class = tf.argmax(box_scores, axis=-1)
    # N,19,19,5   包含某个类别的最高得分
    box_scores = tf.reduce_max(box_scores, axis=-1)

    # 首先把得分低于0.6的框滤去, 现在改成0.57，效果会更好
    # N,19,19,5
    obj_high_prob_mask = box_scores >= 0.57
    # K,
    box_high_scores = tf.boolean_mask(box_scores, obj_high_prob_mask)
    # K,
    box_high_scores_class = tf.boolean_mask(box_class, obj_high_prob_mask)
    # N,19,19,5,4     N,19,19,5    ->  K,4
    high_scores_boxes = tf.boolean_mask(boxes, obj_high_prob_mask)
    '''
    1. 因为存在多个框同时检测同一个对象的可能，
    之所以会产生这个问题，是因为我们在训练网络的时候，对于本不该存在对象的(grid cell,anchor-box)位置,
    它却输出了一个很吻合(IOU>=0.6)人工标注的框,此时应该计算no-obj loss来惩罚网络,但我们忽略了,原因如下

        1.1 训练网络的时候，人工标注的对象是分配到一对(grid cell,anchor-box)中，然而一个单元格中包含多个anchor-box，
        实际上如果存在一个目标形状和多个anchor-box都接近(IOU接近)，那么对象具体分配到哪一个anchor-box都是合理的，
        因此网络在多个位置都输出了预测框也都是合理的，尽管我们标注的位置仍然只会选择一个最优IOU的(grid cell,anchor-box)位置，
        因此我们可以放宽要求，如果在人工标注位置的附近网络也说存在对象，并且预测框和人工标注框很吻合，那么我们将既不惩罚也不激励网络，保持中立。
        并且这些多余的预测结果可被非最大值印制算法滤去。
        另外一方面如果我们要求的输出非常严格，对这些地方进行 no-obj loss惩罚，这样会拥有太多的负例，因为一张图片，
        网络将预测19*19*5=1805个框，通常人工标注的对象少于100个，那么负例将会是1705个，这可能导致网络最终学会了检测某个位置无对象。

    2. 使用非最大值印制，当多个框同时检测同一个对象时，选择得分最高的框。
    对不同的类别应用一次非最大值印制算法
    '''
    return non_max_suppression(box_high_scores_class, box_high_scores, high_scores_boxes)


# 输出的res_boxes
def detect(image_data, model):
    # image_data 数据维度为 608,608,3
    # 进行向前传播算法
    pred = model(image_data)
    # 转换数据，过滤低得分框，各个类非最大值印制，得到网络最终的预测框
    res_class, res_score, res_boxes = convert_filter_and_non_max_suppression(pred)
    return res_class, res_score, res_boxes


# 载入一个训练样本和label
def load_one_dataset(file_name, suffix):
    image, image_data, image_shape = preprocess_image(img_path='train-set/origin/' + file_name + suffix)
    with open('train-set/origin/labeled-' + file_name + '.txt') as reader:
        lines = reader.readlines()
        # K,5的维度
        label = []
        for x in lines:
            line = x.strip()
            # 忽略空行
            if len(line) > 0:
                vs = line.split(',')
                obj = []
                for v in vs:
                    x = v.strip()
                    if len(x) > 0:
                        obj.append(float(x))
                if len(obj) != 5:
                    raise Exception('每个目标必须包含(x,y,w,h,class)信息')
                    # 还可以对值进行检测，x,y,w,h必须是 0～1，class只能是0～79，这里不做验证
                label.append(obj)
    #   返回(1, 608, 608, 3)和(3, 5)两个张量
    return image_data, label, image, image_shape



