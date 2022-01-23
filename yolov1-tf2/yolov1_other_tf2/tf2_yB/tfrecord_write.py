import tensorflow as tf
import os
import xml.dom.minidom as xdom
from tensorflow.keras import datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classes_names = open('D://dataset//VOC2012//VOCdevkit//VOC2012//voc_classes.txt', 'r').readlines()
voc_classes = {classes_names[i].strip(): i+1 for i in range(len(classes_names))}
print(voc_classes)

def Prase_Singel_xml(xml_path):
    DOMTree = xdom.parse(xml_path)
    RootNode = DOMTree.documentElement
    #获取图像名称
    image_name = RootNode.getElementsByTagName("filename")[0].childNodes[0].data
    #获取图像宽和高
    size = RootNode.getElementsByTagName("size")
    image_height = int(size[0].getElementsByTagName("height")[0].childNodes[0].data)
    image_width = int(size[0].getElementsByTagName("width")[0].childNodes[0].data)
    #获取图像中目标对象的名称及位置
    all_obj = RootNode.getElementsByTagName("object")
    bndbox_lable_dic = []
    for one_obj in all_obj:
        obj_name = one_obj.getElementsByTagName("name")[0].childNodes[0].data
        obj_label = voc_classes[obj_name]
        bndbox = one_obj.getElementsByTagName("bndbox")
        #获取目标的左上右下的位置
        xmin = int(bndbox[0].getElementsByTagName("xmin")[0].childNodes[0].data)
        ymin = int(bndbox[0].getElementsByTagName("ymin")[0].childNodes[0].data)
        xmax = int(bndbox[0].getElementsByTagName("xmax")[0].childNodes[0].data)
        ymax = int(bndbox[0].getElementsByTagName("ymax")[0].childNodes[0].data)
        bndbox_lable_dic.append([xmin, ymin, xmax, ymax, obj_label])
    return image_name, image_width, image_height, bndbox_lable_dic

def write_to_tfrecord(xml_path, tfrecord_path, voc_img_path):
    writer = tf.io.TFRecordWriter(tfrecord_path)
    for i, single_xml_name in enumerate(os.listdir(xml_path)):

        image_name, image_width, image_height, bndbox_lable_dic = Prase_Singel_xml(os.path.join(xml_path,single_xml_name))
        sigle_img_path = os.path.join(voc_img_path, image_name)
        image_data = open(sigle_img_path, 'rb').read()
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        obj_label = []
        for j in range(len(bndbox_lable_dic)):
            xmin.append(bndbox_lable_dic[j][0])
            ymin.append(bndbox_lable_dic[j][1])
            xmax.append(bndbox_lable_dic[j][2])
            ymax.append(bndbox_lable_dic[j][3])
            obj_label.append(bndbox_lable_dic[j][4])

        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
            'xmin': tf.train.Feature(int64_list=tf.train.Int64List(value=xmin)),
            'ymin': tf.train.Feature(int64_list=tf.train.Int64List(value=ymin)),
            'xmax': tf.train.Feature(int64_list=tf.train.Int64List(value=xmax)),
            'ymax': tf.train.Feature(int64_list=tf.train.Int64List(value=ymax)),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=obj_label))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        print('第{}张图片写入完毕'.format(i), single_xml_name,image_name)
    writer.close()
write_to_tfrecord('D://dataset//VOC2012//VOCdevkit//VOC2012//Annotations',
                  'D://dataset//VOC2012//VOCdevkit//VOC2012//VOC2012.tfrecord',
                  'D://dataset//VOC2012//VOCdevkit//VOC2012//JPEGImages')