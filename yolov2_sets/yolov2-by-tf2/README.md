# yolov2 by tf2


---

使用 **tensorflow2.0** 实现的 **YOLO**_(you only look once)_ v2算法

_[YOLO](https://pjreddie.com/darknet/yolo/) 是一个实时目标检测系统_

该项目实现了该系统的全部细节，包括模型构建、Loss函数、目标检测、以及如何**从零训练**一个目标检测系统，以便扩展至训练自定义的目标检测系统。

---

## 算法效果

检测结果

<img alt="" src="https://user-images.githubusercontent.com/19931702/113099260-447a1580-922c-11eb-8d5a-8707fb75b16e.png">
<img alt="" src="https://user-images.githubusercontent.com/19931702/113131897-5c639080-9250-11eb-9bac-c527abb3c55c.png">
<img alt="" src="https://user-images.githubusercontent.com/19931702/113132879-9a14e900-9251-11eb-80b5-2f19ad441275.jpg">




---

## 如何使用

#### 环境要求
1. Anconda        2020.11
2. Conda          4.9.2
3. Python         3.7.10
4. Tensorflow     2.0.0

---

1. 检测图片中的目标

```
python test_tect.py
```

2. 保存yolo_v2模型

```
python save_model.py
```

3. 使用保存的yolo_v2模型检测图片
```
python load_model_and_test_detect.py
```

4. 从零训练网络，在2个样本的训练集上训练一个过拟合的模型
```
python train_model_overfit.py
```

5. 使用前面训练的过拟合模型进行目标检测
```
python load_overfit_model_and_test_detect.py
```

## 算法原理

---

#### 1. 网络架构

<img width="1099" alt="" src="https://user-images.githubusercontent.com/19931702/113086704-c2322700-9214-11eb-90b8-dcd510d48bc5.png">




#### 2. 网络的输出

---

网络最终输出一个`(19,19,425)`的张量，我们可以将其转换成`(19,19,5,85)`的张量，
其中`[i,j,a,:]`表示在第`[i,j]`单元格的第`a`个`anchor-box`的预测结果，由`4`部分组成

1. `[0:2]` 存储了对象中心在单元格内的位置`(x,y)`坐标
   - 经过`sigmoid`函数映射，范围限制在`0～1`
2. `[2:4]` 存储了对象的宽高`(w,h)`
   - 经过`exp`函数映射得到关于对应`anchor-box`的宽高系数，必须大于`0`
3. `[4:5]` 存储了该`anchor-box`包含对象的置信度(概率)
   - 经过`sigmoid`函数映射，范围限制在`0～1`
4. `[5:85]` 存储了该`anchor-box`包含的对象关于`80`个类别的概率分布
   - 经过`softmax`归一化，范围限制在`0～1`，总和为`1`

_注意网络输出值并没有经过上面所描述的映射，也就是说我们让网络学会的是这些函数的输入值。_

---

#### 3. 人工标签

维度是`(K,5)`的张量，`K`为图片中包含的对象数量，每个对象由`3`部分决定

1. `[0:2]` 存储了对象中心在整张图片的相对位置`(lx,ly)`，范围在`0～1`
2. `[2:4]` 存储了对象的宽高`(lw,lh)`，范围在`0～1`
3. `[4:5]` 存储了对象的类别`class`，范围在`0～79`

#### 4. 人工标签转换

人工标签的维度是`(K,5)`，显然与网络输出维度不符合，为了计算网络`loss`，我们将
人工标签转换得到`(19,19,5)`维度的张量`detectors_mask`和
`(19,19,5,5)`维度的张量`matching_true_boxes`

---

1. `detectors_mask` 

元素为`bool`类型，`detectors_mask[i,j,a]`为`True`，
则表明在第`[i,j]`单元格的第`a`个`anchor-box`的存在对象

---

2. `matching_true_boxes` 
   
若`detectors_mask[i,j,a]`为`True`,
则`matching_true_boxes[i,j,a,:]`存储了对象信息，由`3`部分决定

   1. `[0:2]` 存储了对象中心在单元格内的位置`(x,y)`坐标
      - 变换规则，`(lx,ly) * (19,19) = (19lx ,19ly) = (u,v)`，
      `(x,y) = (u,v) - floor((u,v))`
   2. `[2:4]` 存储了对象的宽高`(w,h)`，与网络输出的宽高含义一致
      - 变换规则，`(lw,lh) ⊙ (19,19) = (19lw ,19lh) = (p,q)`，
      `(w,h) = log((p,q) / (anchorW,anchorH))`
   3. `[4:5]` 存储了对象的类别`class`，范围在`0～79`
      - 可以使用`one_hot`函数转换成相应的`softmax`标签
    
---

#### 5. Loss函数计算


<img width="623" alt="loss" src="https://user-images.githubusercontent.com/19931702/112642948-c4942a00-8e7e-11eb-929c-a6b39623e536.png">


---

总`Loss` = 识别`Loss`+分类`Loss`+定位`Loss`，均采用平方差`Loss`，上述表达式中`detectors_mask` 前面提到过


1. `object_detections`

`object_detections(i,j,a)` 为`True`，表示该位置预测框与一个真实对象框很吻合(具体是**IOU**>_threshold=0.6_)
，此时即使该位置本不应存在对象即`detectors_mask(i,j,a)=False`也不做`no-obj loss`计算。原因如下：
    
- 训练网络的时候，人工标注的对象是分配到一对`(grid cell,anchor-box)`中，然而一个单元格中包含多个anchor-box，
    实际上如果存在一个目标形状和多个`anchor-box`都接近(`IOU`接近)，那么对象具体分配到哪一个`anchor-box`都是合理的，
    因此网络在多个位置都输出了预测框也都是合理的，尽管我们标注的位置仍然只会选择一个最优`IOU`的`(grid cell,anchor-box)`位置，
    因此我们可以放宽要求，如果在人工标注位置的附近网络也说存在对象，并且预测框和人工标注框很吻合，那么我们将既不惩罚也不激励网络，保持中立。
    并且这些多余的预测结果可被非最大值印制算法滤去。
    另外一方面如果我们要求的输出非常严格，对这些地方进行 `no-obj loss`惩罚，这样会拥有太多的负例，因为一张图片，
    网络将预测`19*19*5=1805`个框，通常人工标注的对象少于`100`个，那么负例将会是`1705`个，这可能导致网络最终学会了检测某个位置无对象。

---

2. 有`4`个权重系数，这里实现上分别取值为`lambda_obj=5`、`lambda_noobj=1`、`lambda_coord=1`、`lambda_class=1`

3. 字母`N`表示类别的数量，`yolov2`系统中是`80`

---

#### 6. 随机梯度下降减少Loss值

<img width="326" alt="0" src="https://user-images.githubusercontent.com/19931702/112708294-16bf6480-8eec-11eb-9aa2-921b7315ce20.png">

<img width="193" alt="1" src="https://user-images.githubusercontent.com/19931702/112708320-3a82aa80-8eec-11eb-998d-910442ba4f74.png">

<img width="486" alt="2" src="https://user-images.githubusercontent.com/19931702/112708364-8897ae00-8eec-11eb-975a-79f1be00f910.png">

<img width="336" alt="3" src="https://user-images.githubusercontent.com/19931702/112708396-c1d01e00-8eec-11eb-8332-6c2dfb9f91f3.png">

<img width="625" alt="4" src="https://user-images.githubusercontent.com/19931702/112708408-dca29280-8eec-11eb-8f8f-c834180008d7.png">

<img width="622" alt="5" src="https://user-images.githubusercontent.com/19931702/112708586-c77a3380-8eed-11eb-828a-7bd3f2e9c970.png">

<img width="624" alt="6" src="https://user-images.githubusercontent.com/19931702/112708595-cfd26e80-8eed-11eb-8941-1e6ab651f603.png">

<img width="482" alt="7" src="https://user-images.githubusercontent.com/19931702/112708684-6868ee80-8eee-11eb-8b52-088ee7f07c2f.png">

<img width="624" alt="8" src="https://user-images.githubusercontent.com/19931702/112708689-774fa100-8eee-11eb-8871-004d08b42dc1.png">

<img width="424" alt="9" src="https://user-images.githubusercontent.com/19931702/112708691-7a4a9180-8eee-11eb-90b2-906e3a6992dd.png">

<img width="533" alt="10" src="https://user-images.githubusercontent.com/19931702/112708779-f349e900-8eee-11eb-8eb7-c07633fc2fdf.png">

---

关于反向传播算法如何计算参数梯度的实现可以参考我的另外两个项目实现

- [CNN](https://github.com/970814/convolutionNerualNetwork)
- [MLP](https://github.com/970814/fcBpNerualNetwork)

如果使用`tensorflow`框架，梯度计算将由框架自动完成，意味着我们只需要实现向前传播算法和损失函数，这是使用框架实现模型的一个极大好处。

---

#### 7. 进行预测

输入图片是一个`(608,608,3)`的张量，经过我们训练的`D-CNN`(具有`23`个卷积层的深度卷积网络)后，
网络最终输出一个`(19,19,425)`的张量，转换成`(19,19,5,85)`的张量，也就是`(19,19,5)`个目标检测结果，
其中每个检测结果有 该位置存在对象的置信度`conf = [4:5]`，`80`个类别的概率分布`prob=[5:85]`，
因此每个类别的得分为`score = conf * prob`，然后我们取得分最高的类别作为检测结果，
因为有`(19,19,5)`个目标检测结果，因此有`19*19*5=1805`个得分，我们过滤掉得分较低的预测`(score<threshold=0.6)`,
剩下的都是具有较高得分的检测结果，但因为训练时，_我们没惩罚包含对象的附近的检测结果_，因此网络存在多位置检测同一个对象的可能，
所以接下来我们对各个类使用非最大值印制算法过滤，得出最终的目标检测结果。这部分原理和实现可以
参考`utils.py`文件中的`convert_filter_and_non_max_suppression`函数

---

### 参考资料

- [YAD2K](https://github.com/allanzelener/YAD2K)
- [简单粗暴TensorFlow2](https://tf.wiki/zh_hans/basic/models.html)
- [吴恩达深度学习YOLO算法](https://www.bilibili.com/video/BV1F4411y7o7?p=31)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) 
- [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
- [Final Layers and Loss Function](https://medium.com/oracledevs/final-layers-and-loss-functions-of-single-stage-detectors-part-1-4abbfa9aa71c)
- [YOLO Loss 细节](https://www.jianshu.com/p/e6582dfa6bb3?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)
- [Yolov2 损失函数细节](https://zhuanlan.zhihu.com/p/56079893)
- [BatchNormalization 踩坑经历](https://zhuanlan.zhihu.com/p/64310188)

