---
layout:     post
title:      神经网络框架汇总(持续更新)
subtitle:   记录现有的一些神经网络框架
date:       2018-06-28
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - 神经网络
    - 深度学习
---

深度学习火了这么久, 每天都有新的论文, 大多数都是原有的结构再修改的, 但也不乏有很多新的网络结构(一直在追赶大牛的脚步很心累).
最近看了一个汇总的网站, 于是乎觉得应该记录下来, 下次见到各种XXNN, XXN不会眼花, 2333333

先上一张汇总图镇楼
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/001.png)

下面只列出一些比较常用的:
## Feed forward neural networks (FF or FFNN) and perceptrons (P) (1958)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/002.png)
感知器是一层的网络, 只有输入和输出.

## Autoencoders (AE) (1998)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/003.png)
自编码器用来自动编码信息(压缩而不是加密). 网络的结构和沙漏很像, 中间隐藏层节点少于输入和输出节点. 而且经常是两边对称的.
一般中间层节点是最少的, 也是信息压缩最多的地方. 左边为编码层, 右边为解编码层. 中间为输出的编码结果.
给定一个输入, 设置error为输入与输出的误差, 通过backpropagation可以训练出一个自编码器. 而且自编码器的编码层和解编码层的权重可以使一样的.

## sparse autoencoders (SAE) (2007)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/004.png)
与自编码器的差别是中间隐藏层比输入和输出节点要多. 这样的网络可以从数据中提取更多细节特征.

## Variational autoencoders (VAE) (2013)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/005.png)
变分自编码器和自编码器的差别在于前者加入了输入数据样本的分布概率.

## Denoising autoencoders (DAE) (2008)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/006.png)
自编码器的输入变成带有噪声干扰的输入, 计算损失时, 对比输出和原始数据的差异.
因为有些时候, 数据中的一些细节经常改变, 导致网络输出错误结果, 加入噪声能够避免学习更多细节, 而只是关注在较为粗略的特征上.


## Deep belief networks (DBN) (2007)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/007.png)
这个网络主要是将RBMs 和 VAEs堆叠起来, 每个网络需要编码前一个网络. 这样的组合也被曾为greedy training.
这里greedy指的是用局部最优解得到一个相对较好的解

## Convolutional neural networks (CNN or deep convolutional neural networks, DCNN) (1998)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/008.png)
卷积网络主要用于图像处理, 也可以用于其他数据, 比如语音等.

## Deconvolutional networks (DN) (2010)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/009.png)
可以被翻译成翻卷积网络


## Deep convolutional inverse graphics networks (DCIGN) (2015)
![all_network](http://zihuaweng.github.io/post_images/neuralnetworks/010.png)
这个网络前半部分是一个卷积网络, 提取特征后编码成概率,


1. http://www.asimovinstitute.org/neural-network-zoo/


