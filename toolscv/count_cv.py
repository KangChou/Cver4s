import cv2
import numpy as np
from time import sleep

largura_min=80 #最小矩形宽度
altura_min=80 #最小矩形高度

offset=6 #允许像素误差

pos_linha=550 #计数行位置

delay= 60 #FPS do vídeo进行毫秒级的延时

detec = []
carros= 0


#获取中心点坐标	
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video2.mp4')
#subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()
#背景减除
subtracao = cv2.createBackgroundSubtractorMOG2()

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) #    time.sleep(0.1) # 休眠0.1秒
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    blur = cv2.GaussianBlur(grey,(3,3),5)#高斯平滑滤波 cv2.GaussianBlur() 参数1：图像、参数2：滤波器大小、参数3：标准差


    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))#cv2.dilate()膨胀：将前景物体变大，理解成将图像断开裂缝变小（在图片上画上黑色印记，印记越来越小） 
     #cv2.erode 进行腐蚀操作，去除边缘毛躁.例如锯齿形的边缘

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))#构造卷积核
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #提取图像轮廓 cv2.findContours()

    
    #cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255,127,0), 3)
    cv2.line(frame1, (pos_linha,25), (pos_linha,1200), (255,127,0), 3)
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min) #满足矩形框的条件就画框，否则继续
        if not validar_contorno:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        #图像加框 cv2.rectangle() 参数1：图像 参数2：左上角坐标 参数3：右下角坐标 参数4：框的颜色 参数5：框的粗细
        centro = pega_centro(x, y, w, h) #获取中心点坐标
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec: #将所以中心点找到
            if y<(pos_linha+offset) and y>(pos_linha-offset): #在直线的两侧有可允许的偏差
                carros+=1  #有车就计数加1
                #cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0,127,255), 3)  
                cv2.line(frame1, (pos_linha, 25), (pos_linha, 1200), (0,127,255), 3)
                detec.remove((x,y))  #计数结束后清除该车在场景中的重复判断
                print("car is detected : "+str(carros))        
       
    cv2.putText(frame1, "Vehicle Count : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    #给图片加文本:# 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
	
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilatada)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()

"""
	0、cv2.findContours#轮廓检测也是图像处理中经常用到的。OpenCV-Python接口中使用cv2.findContours()函数来查找检测物体的轮廓。
	参数1：图像
	参数2：提取规则。cv2.RETR_EXTERNAL：只找外轮廓，cv2.RETR_TREE：内外轮廓都找。
	参数3：输出轮廓内容格式。cv2.CHAIN_APPROX_SIMPLE：输出少量轮廓点。cv2.CHAIN_APPROX_NONE：输出大量轮廓点。
	输出参数1：图像  输出参数2：轮廓列表  输出参数3：层级

	
	1、cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算先腐蚀(瘦)后膨胀(胖)叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域。这类形态学操作用cv2.morphologyEx()函数实现：
	2、v2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算闭运算则相反：先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除/"闭合"物体里面的小黑洞，所以叫闭运算）

	https://www.cnblogs.com/XJT2018/p/9958895.html



	3、OpenCV 中 getStructuringElement() 与 morphologyEx() 函数用法

	getStructuringElement() 与 morphologyEx() 两个函数使用时，经常是放在一起的，后者函数中用到的参数是由前者提供

	    cv2.getStructuringElement()
		返回一个特定大小与形状的结构元素用于形态学操作，生成的结构学元素进一步传入 Opencv 的 erode、dilate、morphologyEx 函数中完成形态学操作，除此之外，也可以自己构建一个任意形状的二进制掩码，作为结构元素

	3.1、getStructuringElement参数详讲

	    shape: 元素形状，OpenCV 中提供了三种，MORPH_RECT(矩阵)，MORPH_CORSS(交叉形状)，MORPH_ELLIPSE(椭圆) ；
	    ksize ，结构元素的大小；
	    anchor，元素内的描点位置，默认为 (-1,-1)表示形状中心；值得注意的时，只有 MORPH-CROSS 形状依赖 描点位置，其它情况 描点仅调节其他形态运算结果偏移了多少
	3.2、morphologyEx参数详讲
	对图像进行形态学转换；利用最基本腐蚀、膨胀形态学操作；所有操作可直接在源图像上实现，针对多通道图像，其中每个通道都是独立处理，这个方法常用来提取图像中某类不规则形状的区域
	src ，预处理的图像；
	op ，形态操作的类型，可选择下面列表中的一种类型，一般参数多选为 cv2.MORPH_CLOSE
	kernel: 结构元素，来自于 getStructuringElement 方法

	针对上面这两种方法的用法，这里给一个例子，提取下方图片中水印部分，也就是中间黑色圆环，和下方字体

	去水印案例：https://www.cnblogs.com/zeroing0/p/14131552.html
	https://www.cnblogs.com/zeroing0/p/14127411.html
	4、cv2.line() 直线   cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → img

	img，背景图
	pt1，直线起点坐标
	pt2，直线终点坐标
	color，当前绘画的颜色。如在BGR模式下，传递(255,0,0)表示蓝色画笔。灰度图下，只需要传递亮度值即可。
	thickness，画笔的粗细，线宽。若是-1表示画封闭图像，如填充的圆。默认值是1.
	lineType，线条的类型，
	https://blog.csdn.net/weixin_42618420/article/details/106097270

	5、enumerate就是枚举的意思，把元素一个个列举出来，第一个是什么，第二个是什么，所以他返回的是元素以及对应的索引。
	https://blog.csdn.net/HiSi_/article/details/108127173
	6、# cv2.boundingRect(img)获得外接矩形--->矩形边框（Bounding Rectangle）是说，用一个最小的矩形，把找到的形状包起来。
	参数说明：x，y, w, h 分别表示外接矩形的x轴和y轴的坐标，以及矩形的宽和高， cnt表示输入的轮廓值
	https://zhuanlan.zhihu.com/p/140512752
	7、cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩行
	参数解释
	第一个参数：img是原图
	第二个参数：（x，y）是矩阵的左上点坐标
	第三个参数：（x+w，y+h）是矩阵的右下点坐标
	第四个参数：（0,255,0）是画线对应的rgb颜色
	第五个参数：2是所画的线的宽度
	8、GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None):
	使用高斯滤波器模糊图像
	Argument:
		src: 原图像
		dst: 目标图像
		ksize: 高斯核的大小；(width, height)；两者都是正奇数；如果设为0，则可以根据sigma得到；
		sigmaX: X方向的高斯核标准差；
		sigmaY: Y方向的高斯核标准差；
			如果sigmaY设为0，则与sigmaX相等；
			如果两者都为0，则可以根据ksize来计算得到；
		（推荐指定ksize，sigmaX，sigmaY）
		borderType: pixel extrapolation method

	https://blog.csdn.net/qq_44868807/article/details/106121194

"""
