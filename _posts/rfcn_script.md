---
layout:     post
title:      R-FCN源码分析（全）
subtitle:   py-R-FCN, caffe实现分析
date:       2018-04-09
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - R-FCN
    - caffe
    - 源码
---

R-FCN的实现和Faster-RCNN类似，中间有类似的部分我会省略，具体可以参考我写的[Faster-RCNN源码解析](https://zihuaweng.github.io/2018/04/04/faster-rcnn-scripts/)

R-FCN的实现也是分为approximate joint training（end2end），  alternative optimization(分开训练)两种。
这个博客我们就讲解alternative optimization的方式，从train_rfcn_alt_opt_5stage.py开始分析。


主要分为以下几个环节：
- Stage 0 RPN, compute normalization means and stds
- Stage 1 RPN, init from ImageNet model
- Stage 1 RPN, generate proposals (train dataset)
- Stage 1 RPN, generate proposals (test dataset)
- Stage 1 R-FCN using RPN proposals, init from ImageNet model
- Stage 2 RPN, init from stage1 R-FCN model
- Stage 2 RPN, generate proposals
- Stage 2 R-FCN using Stage-2 RPN proposals, init from ImageNet model
- Stage 3 RPN, init from stage1 R-FCN model
- Stage 3 RPN, generate test proposals only

可以看到很多步骤和Faster RCNN的训练网络类似，很多变量都是一致的。

## compute normalization means and stds
 
计算targets（targets_dx, targets_dy, targets_dw, targets_dh）平均值，标准差。得到mean,std。

这里使用了rpn_test_prototxt，不是为了计算，也没载入预训练参数，主要是为了计算不同大小的image（h, w = i in xrange(50, cfg.TRAIN.MAX_SIZE + 10)）进去网络后得到的feature map的h,w都是多少，用于计算anchor的位置和大小。

得到了原图的anchor之后，bbox_transform推出targets
    
    mean = (mean_targets_dx, mean_targets_dy, mean_targets_dw, mean_targets_dh)
    std = (std_targets_dx, std_targets_dy, std_targets_dw, std_targets_dh)
    cfg.TRAIN.RPN_NORMALIZE_MEANS = stage0_anchor_stats['means']
    cfg.TRAIN.RPN_NORMALIZE_STDS = stage0_anchor_stats['stds']
    
    
## RPN, init from ImageNet model

这一部分和Faster-RCNN的RPN train部分一样，具体就不展开了。

## RPN, generate proposals

这一部分和Faster-RCNN的RPN test (generate proposals)一样。

这里有两个rpn_generate过程，一个是用rpn生成train dataset的proposals，一个是生成test dataset的proposals。

## R-FCN using RPN proposals, init from ImageNet model
![comparison](http://zihuaweng.github.io/post_images/rfcn/rfcn_train.png)

打开模型的时候我是凌乱的，怎么论文写的这么简单，操作起来竟然这么复杂。。。





## image

![comparison](http://zihuaweng.github.io/post_images/region_proposal/comparison.png)

## Reference:
1. https://github.com/YuwenXiong/py-R-FCN

