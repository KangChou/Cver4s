import tensorflow as tf
import os, time
import tools,net
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
max_box_num = 56
raw_datasets = tf.data.TFRecordDataset("D://dataset//VOC2012//VOCdevkit//VOC2012//VOC2012.tfrecord")
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'xmin': tf.io.VarLenFeature(tf.int64),
    'ymin': tf.io.VarLenFeature(tf.int64),
    'xmax': tf.io.VarLenFeature(tf.int64),
    'ymax': tf.io.VarLenFeature(tf.int64),
    'label': tf.io.VarLenFeature(tf.int64),
}


def parse_example(example_string):
    '''
    1.给做map函数用，针对一个example
    2.解析该example中的各项数据
    3.image给resize一下
    '''
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    image_data = tf.io.decode_jpeg(feature_dict['image'])
    image_data = tf.cast(tf.image.resize_with_pad(image_data, 448, 448), dtype=tf.int32)
    boxes = tf.stack([tf.sparse.to_dense(feature_dict['xmin']),
                      tf.sparse.to_dense(feature_dict['ymin']),
                      tf.sparse.to_dense(feature_dict['xmax']),
                      tf.sparse.to_dense(feature_dict['ymax'])], axis=1)
    boxes_category = tf.sparse.to_dense(feature_dict['label'])
    return image_data, feature_dict['width'], feature_dict['height'], boxes, boxes_category


raw_datasets = raw_datasets.map(tf.autograph.experimental.do_not_convert(parse_example)).shuffle(1000)


def train_gen(raw_dataset, batch_size, random=True):
    '''
    1.产生一个生成器
    2.对于从mapdataset中一个一个的取出的数据
        2.1 补齐维度以组成batch
        2.2 resize box(image已经在parse_example中resize了)
    3.处理mapdataset迭代余数不够的情况逻辑
    4.这个生成器遍历一个epoch,所以raw_datasets不要设成无限循环，不然不好控制epoch
    5.前面的batchsize就是设定的大小，最后一个batchsize是数据库中的余数
    '''
    it = iter(raw_dataset)
    while 1:
        image_data = []
        label_in_one_fixed_shape = []
        current_for_num_obj = 0  # 本次能获取的样本数，该数组成一个batch，前面是=batch_size,最后小于batch_size
        for i in range(1, batch_size + 1):
            try:
                data = next(it)
            except:
                break
            current_for_num_obj = current_for_num_obj + 1
            image_data.append(data[0])
            num_of_obj = len(data[4])  # 当前样本物体个数
            label = tf.pad(tf.expand_dims(data[4], axis=1), [[0, 0], [0, 3]])  # 增加维度并扩张以便和box concat
            boxes_resized = tools.bounding_box_resize(data[1], data[2], data[3])
            label_in_one = tf.concat([boxes_resized, label], axis=1)[:, 0:-3]  # concat 并削去多余维度
            label_in_one = tf.pad(label_in_one, [[0, 56 - num_of_obj], [0, 0]],
                                  constant_values=1)  # 多余填充的box和label都有背景label=1填充
            label_in_one_fixed_shape.append(label_in_one)
        if current_for_num_obj == 0:
            raise StopIteration#手动报错for循环不能捕获
        elif random:
            index = tf.random.shuffle(tf.convert_to_tensor(range(0, current_for_num_obj)))
            image_data = tf.gather(tf.convert_to_tensor(image_data), index)
            label_in_one_fixed_shape = tf.gather(tf.convert_to_tensor(label_in_one_fixed_shape), index)
        yield image_data, label_in_one_fixed_shape  #[四角绝对坐标+分类]


gen = train_gen(raw_datasets, batch_size=1, random=True)  # 生成器 有限迭代
a = next(gen)
print(a[1][0][0])
print(a[1][0][1])
print(a[1][0][2])

tools.pre_and_wirte_gt(a[1],1)
# out = net.yolo_net(tf.cast(a[0],dtype=tf.float32))
# tools.pre_and_write_det(out,1)
# tools.show_one_image_boxes(a[0][0], a[1][0])
