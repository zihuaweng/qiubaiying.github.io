---
layout:     post
title:      梯度下降优化算法总结
subtitle:   尝试root mean squared prop
date:       2018-04-09
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - R-FCN
    - caffe
    - 源码
---

### 梯度下降算法:
1. Batch gradient descent: 每次计算所有数据(m)得到损失函数结果,计算梯度,更新参数
2. Stochastic gradient descent: 每次只计算一个样本,得到损失函数结果,计算梯度,更新参数
3. Mini-batch gradient descent: 每次计算一个mini-batch(n, 1<n<m)个样本,得到损失函数结果,计算梯度,更新参数

### Momentum


![comparison](http://zihuaweng.github.io/post_images/region_proposal/comparison.png)

1. http://ruder.io/optimizing-gradient-descent/index.html#nesterovacceleratedgradient

