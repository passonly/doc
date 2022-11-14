# 8.图像金字塔

### 8.1 图像金字塔介绍

**图像金字塔**是图像中多尺度表达的一种，最主要用于图像的分割，是一种以多分辨率来解释图像的有效但概念简单的结构。简单来说, 图像金字塔是同一图像不同分辨率的子图集合.

图像金字塔最初用于机器视觉和图像压缩，一幅图像的金字塔是一系列以金字塔形状排列的分辨率逐步降低，且来源于同一张原始图的图像集合。其通过梯次向下采样获得，直到达到某个终止条件才停止采样。金字塔的底部是待处理图像的高分辨率表示，而顶部是低分辨率的近似。我们将一层一层的图像比喻成金字塔，层级越高，则图像越小，分辨率越低。

![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/77ac2dc64c30496ba74419ba8140bb2b.png)

**常见两类图像金字塔**

**高斯金字塔 ( Gaussian pyramid)**: 用来向下/降采样，主要的图像金字塔
**拉普拉斯金字塔(Laplacian pyramid)**: 用来从金字塔低层图像重建上层未采样图像，在数字图像处理中也即是预测残差，可以对图像进行最大程度的还原，配合高斯金字塔一起使用。 

### 8.2 高斯金字塔

**高斯金字塔**是通过高斯平滑和亚采样获得一系列下采样图像.

原理非常简单, 如下图所示:

![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/70ccb539c28b45bd9bd74c8a33f42e6f.png)

原始图像 M * N -> 处理后图像 M/2 * N/2.

每次处理后, 结果图像是原来的1/4.

<img src="https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/8f03c73ec6314c4b9032433cac031d4c.png" style="zoom:67%;" />

注意: 向下采样会丢失图像信息.

- 向上取样

  向上取样是向下取样的相反过程, 是指图片从小变大的过程.

  ![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/dba93326ec384953a7ddf22ceccb97f3.png)

- pyrDown 向下采样

``` python
import cv2
import numpy as np


img = cv2.imread('./lena.png')

print(img.shape)
dst = cv2.pyrDown(img)

print(dst.shape)

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src="https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/9303cc1cbc484e958664173bc67dfd6b.png" style="zoom:67%;" />

- pyrUp 向上采样

  ``` python
  # 向上采样
  # 向下采样
  import cv2
  import numpy as np
  
  
  img = cv2.imread('./lena.png')
  
  print(img.shape)
  dst = cv2.pyrUp(img)
  
  print(dst.shape)
  
  cv2.imshow('img', img)
  cv2.imshow('dst', dst)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

- 取样可逆性研究

​		在根据向上和向下取样的原理, 我们能够发现图像在变大变小的过程中是有信息丢失的. 即使把图片变回原来大小,图片也不是原来的图片了, 而是损失了一定的信息.

``` python
# 研究采样中图像的损失
import cv2
import numpy as np


img = cv2.imread('./lena.png')

# 先放大, 再缩小
dst = cv2.pyrUp(img)
dst = cv2.pyrDown(dst)

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.imshow('loss', img - dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src="https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/763cde83d2c44b46b552386fadf3197f.png" style="zoom:67%;" />

### 8.3 拉普拉斯金字塔

![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/d3cb797d2a1c4a998b086365e6c6b024.png)

将降采样之后的图像再进行上采样操作，然后与之前还没降采样的原图进行做差得到残差图！为还原图像做信息的准备！

也就是说，拉普拉斯金字塔是通过源图像减去先缩小后再放大的图像的一系列图像构成的。保留的是残差！

![](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/0f4d58bea3754c71b21b4be7dd6806c2.png)

![](D:\课件\opencv\图像金字塔\img\68fde618ac83422da4f67044b0745ad4.png)

``` python
# 研究采样中图像的损失
import cv2
import numpy as np


img = cv2.imread('./lena.png')

dst = cv2.pyrDown(img)
dst = cv2.pyrUp(dst)

lap0 = img - dst
cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.imshow('lap0', lap0)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src="https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/533/1630072391000/926965483a884b8fba10393faca587dd.png" style="zoom:67%;" />