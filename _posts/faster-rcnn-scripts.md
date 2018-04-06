---
layout:     post
title:      faster rcnn 源码解读（全）
subtitle:   对py-faster-rnn，caffe实践代码的全部解读
date:       2018-04-04
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - faster rcnn
    - caffe
    - c++
    - python
---

这篇博客主要想记录一下学习Ross Girshick大神的[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)源码的笔记，方便以后翻看。

博主项目里只用tensorflow,无奈很多好的模型都用caffe，加上caffe确实比tensorflow好读懂，所以顺带学一遍caffe，顺便学习贾大神写的caffe c++源码。

faster rcnn结构还是比较复杂的，下面开始解读一下源码：

### 训练过程概况：

由train_faster_rcnn_alt_opt.py可以看到整个训练过程，分为一下几个部分：

- Stage 1 RPN, init from ImageNet model
- Stage 1 RPN, generate proposals
- Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model
- Stage 2 RPN, init from stage 1 Fast R-CNN model
- Stage 2 RPN, generate proposals
- Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model

主要是两个循环：训练RPN，用RPN生成的proposal训练Fast R-CNN，再训练RPN，再用RPN生成的proposal训练Fast R-CNN。

我们可以获取所有阶段的pt文件看模型结构，重点看以下几个环节：
RPN训练：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/rpn_train.png)
proposal生成：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/rpn_test.png)
Fast R-CNN训练：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/fastrcnn.png)
Fast R-CNN测试：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/faster_test.png)

其中包含了几个重要的layer,下面会展开。

RoIDataLayer, AnchorTargetLayer, ProposalLayer, ProposalTargetLayer, ROIPooing, SmoothL1Loss layer

rpn_cls_prob_reshape =  （1， 2*9， 14， 14）
前面九个是背景概率，后面就个是前景概率

### anchor target layer:
针对每个anchor生成target和label。分类的label为1（物体），0（非物体），-1（忽略）。如果分类lable>0则进行box regression

1. Generate proposals from bbox deltas and shifted anchors
首先得到原有图片大小：rpn_cls_score中feature map × feat_stride（16）

接着得到anchor(9个，包含了三种大小，尺寸的组合)对应原图中bbox，一共有14x14x9个

保留那些不超出图片大小的anchor（shape =（n, 4））（下面计算只用这些数量的anchor,但是最后返回包含去掉的anchor）

通过覆盖度阈值（inter_area / anchor_area U gt_area），给所有anchor标签1,0，,1

计算bbox_targets（shape=（N，4）），bbox_targets是anchor与gt的偏差，由下面公式得到, P=anchor, G=gt

![screenshot from 2017-12-18 15-24-52](https://user-images.githubusercontent.com/13395833/34094306-9c477f2c-e407-11e7-82bb-59191df6c6ee.png)

bbox_inside_weights（shape=(N, 4)），标记为物体的anchor值为（1,1,1,1），其余为（0,0,0,0），保证在计算smooth l1的时候，我们是对前景做回归，忽略其他。

bbox_outside_weights（shape=(N, 4)），如果TRAIN.RPN_POSITIVE_WEIGHT（p） < 0， 前景和背景的权重都为 1 / num(anchor), 如果0 < TRAIN.RPN_POSITIVE_WEIGHT(p) < 1，前景权重为 p * 1 / {num positives}，背景为（1-p）/ {num negetive}，其余为（0,0,0,0）。这样做是因为前景和背景的数量差异很大。

bbox_inside_weights 和 bbox_outside_weights详细使用看下面smoothl1loss层解析。

最后四个返回值都加上因为超出图片大小而去掉的anchor，reshape后，返回四个值的大小为：
A = anchor数量（9）
- labels (1, 1, A * height, width)
- bbox_targets (1, A * 4, height, width)
- bbox_inside_weights (1, A * 4, height, width)
- bbox_outside_weights (1, A * 4, height, width)

### proposal layer:
将RPN结果(per-anchor scores and bbox regression estimates)转为原图上的object proposals。
1. Generate proposals from bbox deltas and shifted anchors
- 首先得到原有图片大小：pool5中feature map × feat_stride（16）
- 接着得到anchor(9个，包含了三种大小，尺寸的组合)对应原图中bbox，一共有14x14x9个
- 最后通过rpn_bbox_pred(由1*1卷积学习到的feature map)回归出映射得到的bbox

rpn_bbox_pred = bbox_deltas

可以理解为anchor与预测的偏移量
![screenshot from 2017-12-18 15-24-52](https://user-images.githubusercontent.com/13395833/34094306-9c477f2c-e407-11e7-82bb-59191df6c6ee.png)

反推回去，由anchor计算得到一个predict box(proposals), G冒号=预测，P=anchor
~~~
proposals = bbox_transform_inv(anchors, bbox_deltas)
~~~


2. clip predicted boxes to image
proposals = clip_boxes(proposals, im_info[:2])
im_info = (H, W, scaler)
第一步已经得到具体的位置了，这一步其实只是把超出图片大小的proposal给裁剪成在图片范围内
3. 将proposal长宽小于阈值的去掉
4. 选择rpn_cls_prob_reshape较大的前pre_nms_topN个proposal(一般为2000个)
5. 非最大抑制，留下after_nms_topN个proposal（一般为300个，阈值为iou0.7）

### smoothl1loss layer
计算损失函数论文使用：


Lreg(ti, t∗i ) = R(ti − t∗i ), R是smoothl1loss

rpn_bbox_pred  anchor与预测的偏移量 ti

rpn_bbox_targets  anchor与gt的偏差 t*i

rpn_bbox_inside_weights p*i

计算smoothl1时，需要乘以rpn_bbox_outside_weights，即w_out * SmoothL1，如果正负样本为1:1，如文章所说，rpn_bbox_outside_weights为1 / num(anchor)，就是1/Nreg。如果正负样本数量差异很大，则应该乘以具体的rpn_bbox_outside_weights，而不是直接1/Nreg。

~~~~~c++
caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
  if (has_weights_) {
    // apply "inside" weights
    caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w_in * (b0 - b1)
  }
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;

  if (has_weights_) {
    // apply "outside" weights
    caffe_gpu_mul(
        count,
        bottom[3]->gpu_data(),
        errors_.gpu_data(),
        errors_.mutable_gpu_data());  // d := w_out * SmoothL1(w_in * (b0 - b1))
  }
~~~~~

smoothloss1:
https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/smooth_L1_loss_layer.cu

## Reference:
https://blog.csdn.net/wfei101/article/details/77778462
