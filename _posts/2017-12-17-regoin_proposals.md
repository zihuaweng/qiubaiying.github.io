---
layout:     post
title:      多种regoin_proposal比较
subtitle:   
date:       2017-12-24
author:     zihuaweng
header-img: 
catalog: true
tags:
    - Object_detection
    - regoin_proposal
    - 论文笔记
---

## 概况
物体检测通常的做法是选出一定数目的候选框，然后针对候选框做图像识别。候选框的选择有以下几种算法：
sideing window 滑窗：传统做法是设定一些框，在图片上不断滑动，得到候选窗，所以一张图片有可能选出的候选框达\\( 10^4-10^7 \\)个之多。
detection proposals：由于滑窗的计算量很大，就有了detection proposals这一类算法，一般认为可能为物体的区域与背景有明显的特征差异，我们尽可能选出可能是物体的那些框的

## Reference
[What makes for effective detection proposals?](https://arxiv.org/abs/1502.05082)
[]()
[BING: Binarized Normed Gradients for Objectness Estimation at 300fps](http://ieeexplore.ieee.org/document/6909816/)
