import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import os
def show_one_image_boxes(one_image,label_in_one,background_label = 1):
    '''
    1.显示一张图的所有box
    2.显示一张图的格子
    3.显示物体中心点(目前用小矩形代替点将就将就)
    :param one_image:
    :param label_in_one:该图的标签，包括坐标和类别，还有填充的
    :param background_label:resize_with_pad时填充的值，也等于背景的类别标签
    :return:
    '''
    fig, ax = plt.subplots(1)
    ax.imshow(one_image)

    for i in range(0,56):#画boxes
        if int(label_in_one[i][4]) != background_label:
            x_min = label_in_one[i][0]
            x_max = label_in_one[i][2]
            y_min = label_in_one[i][1]
            y_max = label_in_one[i][3]
            h = int(y_max - y_min)
            w = int(x_max - x_min)
            ax.add_patch(patches.Rectangle((int(x_min), int(y_min)), w, h, linewidth=1, edgecolor='r', facecolor='none'))
            ax.add_patch(patches.Rectangle((int((x_min+x_max)/2), int((y_min+y_max)/2)), 1, 1, linewidth=2, edgecolor='r', facecolor='none'))#用小矩形代替点，将就将就
    for line in range(1,7):#画cells
        plt.axhline(line * 448 / 7,lw = 1)
        plt.axvline(line * 448 / 7,lw = 1)

    plt.show()

def bounding_box_resize(w,h,boxes):
    '''
    1.对于一个image的所有box进行resize
    2.根据tf.image.resize_with_pad()的原理来做
    :param w: 原图的w
    :param h: 原图的h
    :param boxes: 一张图的boxes 不包括那些填充的
    '''
    if (int(448 / w * tf.cast(h, dtype=tf.float64)) < 448):  # h边优先缩放，w边padding
        rate_resize = float(448 / w)
        padding_one_half = int((448 - int(int(h) * rate_resize)) / 2)  # 对称padding 所以/2
        x_min_resized = tf.cast(tf.cast(boxes[:, 0:1], dtype=tf.float32) * rate_resize, dtype=tf.int64)
        y_min_resized = tf.cast(tf.cast(boxes[:, 1:2], dtype=tf.float32) * rate_resize,dtype=tf.int64) + padding_one_half
        x_max_resized = tf.cast(tf.cast(boxes[:, 2:3], dtype=tf.float32) * rate_resize, dtype=tf.int64)
        y_max_resized = tf.cast(tf.cast(boxes[:, 3:4], dtype=tf.float32) * rate_resize,dtype=tf.int64) + padding_one_half
        boxes_resized = tf.concat([x_min_resized, y_min_resized, x_max_resized, y_max_resized], axis=1)

    else:  # w边优先缩放，h边padding
        rate_resize = float(448 / h)
        padding_one_half = int((448 - int(int(w) * rate_resize)) / 2)  # 对称padding 所以/2
        x_min_resized = tf.cast(tf.cast(boxes[:, 0:1], dtype=tf.float32) * rate_resize,dtype=tf.int64) + padding_one_half
        y_min_resized = tf.cast(tf.cast(boxes[:, 1:2], dtype=tf.float32) * rate_resize, dtype=tf.int64)
        x_max_resized = tf.cast(tf.cast(boxes[:, 2:3], dtype=tf.float32) * rate_resize,dtype=tf.int64) + padding_one_half
        y_max_resized = tf.cast(tf.cast(boxes[:, 3:4], dtype=tf.float32) * rate_resize, dtype=tf.int64)
        boxes_resized = tf.concat([x_min_resized, y_min_resized, x_max_resized, y_max_resized], axis=1)
    return boxes_resized

def cell_cal(box_loc):#输入box坐标xmin,ymin,xmax,ymax, 返回cell的坐标,cell坐标从[0,0]开始
    x_cen = (box_loc[0] + box_loc[2]) / 2
    y_cen = (box_loc[1] + box_loc[3]) / 2
    return [int(x_cen / (448 / 7)) , int(y_cen / (448 / 7))]

def iou(box1,box2,cell_loc):
    #box1=[x,y,w,h],为预测值，x,y为物体中心坐标，且是相对于该cell的，w,h是实际高宽相对于图片高宽的比值的平方根
    #box2=[xmin,ymin,xmax,ymax]  四角的绝对坐标
    #cell_loc为cell坐标，用来推算box1的绝对坐标
    x = int(box1[0] * (448 / 7) + cell_loc[0] * (448 / 7))#计算绝对中心坐标
    y = int(box1[1] * (448 / 7) + cell_loc[1] * (448 / 7))
    w = int(box1[2] * box1[2] * 448)#计算绝对高宽
    h = int(box1[3] * box1[3] * 448)
    x1_min = x - int(w / 2)#计算四角绝对坐标
    x1_max = x + int(w / 2)
    y1_min = y - int(h / 2)
    y1_max = y + int(h / 2)
    x_iou_min = max(x1_min,box2[0])#计算交集的四角绝对坐标
    y_iou_min = max(y1_min,box1[1])
    x_iou_max = min(x1_max,box2[2])
    y_iou_max = min(y1_max,box2[3])
    if (x_iou_max > x_iou_min) and (y_iou_max > y_iou_min):#判断计算出的IOU区域是否正常
        s_iou = (x_iou_max - x_iou_min) * (y_iou_max - y_iou_min)
        s1 = (x1_max - x1_min) * (y1_max - y1_min)
        s2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return float(s_iou / (s1 + s2 - s_iou))
    else:
        return 0.

def pre_and_write_det(out_net,file_num):#file_num由上级决定，一张图片一个file_num
    out_net = tf.reshape(tf.squeeze(out_net),(7,7,30))
    for catag in range(1,20):#分类进行,背景类即使有预测也舍去，应为实际使用时，也会丢弃背景类
        num = 0 #每一类会产生2个长7*7*2的列表,num用于记录第一套预测填的位置
        box_98 = [0 for x in range(0,98)]
        c_98 = [0 for x in range(0,98)]
        for i in range(0,7):
            for j in range(0,7):
                stick_30 = out_net[i][j]#取出一个30长条
                c_98[num] = (stick_30[catag] * stick_30[20])#第catag个分类值*第一套置信度
                c_98[num + 49] = (stick_30[catag] * stick_30[21])

                #把预测坐标转化为四角绝对坐标
                x1 = stick_30[22] * (448 / 7) + i * (448 / 7)
                y1 = stick_30[23] * (448 / 7) + j * (448 / 7)
                w1 = stick_30[24] * stick_30[24] * 448
                h1 = stick_30[25] * stick_30[25] * 448
                # 把预测坐标转化为四角绝对坐标
                x2 = stick_30[26] * (448 / 7) + i * (448 / 7)
                y2 = stick_30[27] * (448 / 7) + j * (448 / 7)
                w2 = stick_30[28] * stick_30[28] * 448
                h2 = stick_30[29] * stick_30[29] * 448
                #可能还需加入越界判断，因为有padding，那就需要图片的真实尺寸信息
                box1 = [x1 - int(w1 / 2), y1 - int(h1 / 2), x1 + int(w1 / 2), y1 + int(h1 / 2)]
                box2 = [x2 - int(w2 / 2), y2 - int(h2 / 2), x2 + int(w2 / 2), y2 + int(h2 / 2)]
                box_98[num] = box1
                box_98[num + 49] = box2
                num = num + 1
        c_98 = [float(x) for x in c_98]
        box_98 = [[int(x[0]),int(x[1]),int(x[2]),int(x[3])] for x in box_98]

        index = tf.image.non_max_suppression(box_98,c_98,1)#可能需要转成tensor
        boxes_to_be_wroten = tf.gather(box_98,index)
        c_to_be_wroten = tf.gather(c_98,index)#这里的c可以考虑用原始的c
        write_det(boxes_to_be_wroten,c_to_be_wroten,catag,file_num)


def write_det(boxes,cs,catag,file_num):#img_num表示文件名，只要det 和 gt能对于上就行，一个文件表示一个图片的结果
    classes_names = open('D://dataset//VOC2012//VOCdevkit//VOC2012//voc_classes.txt', 'r').readlines()
    classes_names = [x.strip() for x in classes_names]
    det = open(('dets\\{}.txt'.format(file_num)),'a')
    for i,box in enumerate(boxes):
        str = '{} {} {} {} {} {}\n'.format(classes_names[catag],cs[i],box[0],box[1],box[2],box[3])
        det.write(str)
    det.close()

def pre_and_wirte_gt(label_in_one,file_num):
    gt = open('gts\\{}'.format(file_num),'a')
    classes_names = open('D://dataset//VOC2012//VOCdevkit//VOC2012//voc_classes.txt', 'r').readlines()
    classes_names = [x.strip() for x in classes_names]
    label_in_one = tf.squeeze(label_in_one)
    for i in range(0,56):
        if int(label_in_one[i][4]) != 1:
            x_min = label_in_one[i][0]
            x_max = label_in_one[i][2]
            y_min = label_in_one[i][1]
            y_max = label_in_one[i][3]
            c = label_in_one[i][4]
            str = '{} {} {} {} {}\n'.format(classes_names[c-1],x_min,y_min,x_max,y_max)
            gt.write(str)
    gt.close()