---
layout:     post
title:      SIFT解析
subtitle:   Scale Invariant Feature Transform
date:       2018-07-05
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - loss function
    - tensorflow
---

## 概况
SIFT是用于提取图片中的特征, 可进行不同图片之间的比对. SIFT提取的特征具有对缩放, 旋转等不变性.

对于图像比对和识别, SIFT能够提取一系列特征并保存, 新图片中提取的特征会在前一个特征库中查找最相近的特征, 就可以一一对应上了.

### keypoint, descriptor
关键点是描述子的位置, 描述子是用来表示关键点周围的局部特征
![sift](http://zihuaweng.github.io/post_images/sift/001.png)
关键点是要找一个局部与其他不同的点
![sift](http://zihuaweng.github.io/post_images/sift/002.png)

#### Gaussian平滑
平滑后, 突出了边缘, 以及那些像素点强度变化很大的区域.
#### Difference-of-Gaussians
![sift](http://zihuaweng.github.io/post_images/sift/004.png)
首先看下面一部分, 对原图使用不同的高斯平滑算子得到了不同的结果, 如下图:
 ![sift](http://zihuaweng.github.io/post_images/sift/006.png)
然后得到的相邻结果两两相减得到Difference-of-Gaussians.

接下来, 将原图缩小, 再做一遍上面的步骤, 得到另外的Difference-of-Gaussians.
最后得到的图片:
![sift](http://zihuaweng.github.io/post_images/sift/005.png)
可以看到, 效果图片中, 只保留了角点, 边缘和其他细节, 变化大的局部, 颜色相同的部分都被忽略了.

#### 消去边缘
但这里有个问题, 边缘不适合比对, 因为沿着边缘, 很多局部其实是相同的, 所以很难找到局部差异点, 所以需要把边缘去除掉

SIFT使用Hessian特征值比率来比较

### SIFT描述子

### 比对
计算描述子的距离, 容易引起错误比对

所以需要区别outlier和inlier
## RANSAC



## 应用

## 人脸上面的应用
http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

IntraFace uses SIFT features for feature mapping and trains a descent method by a linear regression on training set in order to ex- tract 49 points

![comparison](http://zihuaweng.github.io/post_images/region_proposal/comparison.png)

## Reference
1. https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
2. https://blog.csdn.net/zddblog/article/details/7521424
3. https://www.youtube.com/watch?v=oT9c_LlFBqs