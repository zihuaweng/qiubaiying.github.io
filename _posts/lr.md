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

# 最近花了点时间分清了线性回归模型和逻辑回归的区别。首先总结一下两者的区别。

1）线性回归要求变量服从正态分布，logistic回归对变量分布没有要求。
2）线性回归要求因变量是连续性数值变量，而logistic回归要求因变量是分类型变量。
3）线性回归要求自变量和因变量呈线性关系，而logistic回归不要求自变量和因变量呈线性关系
4）logistic回归是分析因变量取某个值的概率与自变量的关系，而线性回归是直接分析因变量与自变量的关系


总之,
logistic回归与线性回归实际上有很多相同之处，最大的区别就在于他们的因变量不同，其他的基本都差不多，
正是因为如此，这两种回归可以归于同一个家族，即广义线性模型（generalizedlinearmodel）。
这一家族中的模型形式基本上都差不多，不同的就是因变量不同，如果是连续的，就是多重线性回归，如果是二项分布，就是logistic回归。
logistic回归的因变量可以是二分类的，也可以是多分类的，但是二分类的更为常用，也更加容易解释。
所以实际中最为常用的就是二分类的logistic回归。

## 逻辑回归（二分类回归模型）
\\[ P(Y = 1) = exp(wx) / (1 + exp(wx)) \\]
（与sigmoid公式类似）
逻辑回归模型将线性函数转化为概率，当线性模型的值越接近正无穷，概率值就越接近1；线性模型的值越接近负无穷，概率值就越接近0。

得到概率之后，判断标签。

## 线性回归

![comparison](http://zihuaweng.github.io/post_images/region_proposal/comparison.png)

1. https://blog.csdn.net/gcs1024/article/details/77478404

