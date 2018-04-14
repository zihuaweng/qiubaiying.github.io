---
layout:     post
title:      gpu cpu差别（全）
subtitle:   从硬件上和软件上区别gpu,cpu
date:       2018-04-15
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - cpu
    - cpu
    - cuda
---

在优化线上模型部署的时候遇到了cpu,gpu并发跟不上的问题，所以特意研究了一下gpu,cpu差别。把整合的内容记录下来，分别有硬件，软件角度分析。

## 硬件
GPU为什么设计的

GPU的设计主要针对解决那些可以分解成成千上万个小块并且可以独立运行的问题，适用于计算量大，简单，多并行的计算。所以设计了很多小的数据单元执行同样的任务。
GPU由多个流处理器簇（SM）排列组成，而每个流处理簇又由多个流处理器（SP）组成。SM可以看成是一个CPU的核，如果持续增加SM的数量，GPU性能会持续提高。
而CPU的设计主要是用来运行少量比较复杂的任务。设计上是典型双核或者四核设备，有数据Cache和流程控制器。能够处理多任务串行的问题。
![cpu_gpu](http://zihuaweng.github.io/post_images/cpu_gpu/cpu_gpu.png)
![cpu_gpu_1](http://zihuaweng.github.io/post_images/cpu_gpu/cpu-and-gpu.jpg)

### 线程上设计
CPU，GPU支持线程的方式不同，CPU每个核只有少量寄存器，在执行多任务的时候，
上下文比较昂贵，因为需要将寄存器里的数据保存到RAM，重新执行任务时，需要从RAM中恢复。
而GPU上下文切换只需要设计一个寄存器组调度着，用于将当前寄存器里的内容换进换出，
速度比保存到RAM快好几个数量级。

CPU在线程数增加的时候，上下文切换时间会很多，通常这个时候在进行I/O操作或者内存获取，
这个时候CPU会被闲置。而GPU采用的是数据并行模式，他需要成千上万的线程操作。利用有效
的工作池保证GPU不会被闲置。GPU默认是并行模式，SM每次可以计算32个数，而CPU只计算1个。
GPU的每个SM都有一个内部访问的共享内存和寄存器，SP与寄存器交流速度很快，几乎不用等。
处理的数据可以在放在共享内存，不用担心上下文切换需要转移。

## Reference
1. https://blog.csdn.net/abcjennifer/article/details/42436727

