---
layout:     post
title:      多种region_proposal算法比较
subtitle:   
date:       2017-12-24
author:     zihuaweng
header-img: img/post_bg_region.png
catalog: true
tags:
    - Object_detection
    - region_proposal
    - 论文笔记
---

## 概况
物体检测通常的做法是选出一定数目的候选框，然后针对候选框做图像识别。候选框的选择有以下几种算法：
sideing window 滑窗：传统做法是设定一些框，在图片上不断滑动，得到候选窗，所以一张图片有可能选出的候选框达\\( 10^4-10^7 \\)个之多。
detection proposals：由于滑窗的计算量很大，就有了detection proposals这一类算法，一般认为可能为物体的区域与背景有明显的特征差异，我们尽可能选出可能是物体的那些框的

## detection proposals分类 
(文中只考虑有开源代码的方法)
1. Grouping proposal methods: 将图片分割成小块然后聚合多个可能含有物体的候选区域。根据生成候选区域的方法，可以分成三类： 
- grouping superpixels (SP， 聚合superpixels): **SelectiveSearch**, RandomizedPrim’s, Rantalankila, Chang, 
- graph cut (GC，使用不同种子然后graph cut): CPMC, Endres, Rigor
- edge contours (EC,根据边缘轮廓提取区域): Geodesic, MCG
2. Window  scoring methods: 根据候选框含有物体的可能性给每一个框打分，这种方法只返回bounding boxes，速度更快，但是位置信息没有那么准确。有下面几种实现：
**Objectness**， Rahtu， **Bing**， EdgeBoxes， Feng， Zhang， RandomizedSeeds
3. 其他 proposal methods：
ShapeSharing， Multibox（神经网络学习）
4. 对比的baseline方法：
Uniform， Gaussian， SlidingWindow， Superpixels

![comparison](http://zihuaweng.github.io/post_images/region_proposal/comparison.png)

### 其中一些算法介绍：
1. SelectiveSearch: 通过合并superpixels生成候选框。需要自定义特征和合并条件。目前RCNN系列都采用了这种方式。
    - 解释以下superpixels，superpixels可以理解为将一张图片分割成比像素更大的一个个小块。例如下面的这张图片分割，图c是将原图a分割成200个superpixels结果：
    ![superpixels](http://zihuaweng.github.io/post_images/region_proposal/superpixels.png)
2. Objectness： 较早的一种proposal方法。是根据图像中比较突出的位置选择一些初始框，然后根据选中区域的颜色，边缘，位置，大小等特征打分。
3. Bing: 训练了一个简单的线性分类器来通过类似滑窗的方式来过滤候选框，速度惊人地快，在CPU上能够达到ms级别。
4. EdgeBoxes： 不需要学习参数，结合滑窗，通过计算窗口内边缘个数进行打分，最后排序。
5. MCG： generates hierarch-
ical segmentations
6. Rigor: computes
multiple graph cut segmentations,

   
## 评价一：结果复现的鲁棒性repeatability
好算法应该有比较好的复现能力，对相似的图片提取结果具有一致性的。验证的方法是对PASCAL的图片做了各种扰动（如下图），然后看还能检测出来相同的object的recall是多少，根据IoU的严格与否能够得到一条曲线，最后计算曲线下面积得到repeatability。

![扰动效果](http://zihuaweng.github.io/post_images/region_proposal/change_image.png)

## 评价二：是否会漏检recall
这里有三种评价方式：
1. 固定proposal数量，根据不同IoU值计算recall
2. 固定IoU阈值，根据不同的proposal数量计算recall
    - 结果：MCG， EdgeBox，SelectiveSearch, Rigor和Geodesic最好。
3. 文中新定义的average recall (AR)，根据在IoU为0.5-1之间proposal数量计算recall
    - 结果：
        - MCG在各种proposal下表现最好
        - proposal小于1000时，endres和CPMC效果都比较好
        - proposal大于1000时，Rigor和SelectiveSearch表现更好

![recall效果](http://zihuaweng.github.io/post_images/region_proposal/ap_result.png)

## 评价三：用于物体检测时效果
1. 检验算法在物体检查中的效果 
使用各种算法得到1000个proposal然后用DPM（LM-LLDA）和RCNN和FastRCNN的物体检测效果评价算法
- 从下面结果看出，IoU越大，物体检测效果越好，所以提高定位能够提高最终效果。
![IoU效果](http://zihuaweng.github.io/post_images/region_proposal/iou_gt.png)
- 在LM-LLDA和RCNN中表现最好的前五个比较类似，都有MCG,SelectiveSearch,Rigor
![LM-LLDA和RCNN效果](http://zihuaweng.github.io/post_images/region_proposal/detection_result.png)
2. 验证了AR和mAP相关后可以用AR结果对proposal算法进行微调。
结合AR结果，为EdgeBoxes选择更好的参数可以在物体检测中得到1.6个点的提升
![detection提升](http://zihuaweng.github.io/post_images/region_proposal/edgeboxes.png)
## 最后结论
1. 目前为止repeatability不是评价proposal方法的一个重要指标，因为表现好的算法例如Bing在后面的检测效果中并不如repeatability表现的差的SelectiveSearch和EdgeBoxes。
2. 定位越准（IoU越大）的proposal方法在物体检测中效果越好。所以不是IoU0.5效果就会好。
3. SelectiveSearch,Rigor,MCG和EdgeBoxes在物体检测表现更好,而且各自的效果差不多。其中EdgeBoxes在速度和结果上表现都不错。

## Reference
1. [What makes for effective detection proposals?](https://arxiv.org/abs/1502.05082)
2. [superpixels](http://ttic.uchicago.edu/~xren/research/superpixel)
3. [BING: Binarized Normed Gradients for Objectness Estimation at 300fps](http://ieeexplore.ieee.org/document/6909816/)
4. blog.csdn.net/baobei0112/article/details/47950963

