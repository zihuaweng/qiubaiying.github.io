---
layout:     post
title:      tensorflow 多线程计算
subtitle:   cpu, gpu的多线程并发计算
date:       2018-04-11
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - tensorflow
    - gpu
    - 并发
---

tensorflow默认会使用全部识别到的gpu,占满所有的显存，这样做的好处是通过减少内存碎片来高效利用gpu显存。

我理解的是，程序使用的时候，内存会被分配成各自的小块利用，这样就无法得到连续的内存块，大多数情况会浪费资源。默认沾满所有的内存的话可以就不会存在有的部分被闲置用不了的显存块了。



cpu:
config = tf.ConfigProto(device_count={"CPU": 2},
                        inter_op_parallelism_threads=2,
                        intra_op_parallelism_threads=1
                        
http://nooverfit.com/wp/tensorflow%E5%A6%82%E4%BD%95%E5%85%85%E5%88%86%E4%BD%BF%E7%94%A8%E6%89%80%E6%9C%89cpu%E6%A0%B8%E6%95%B0%EF%BC%8C%E6%8F%90%E9%AB%98tensorflow%E7%9A%84cpu%E4%BD%BF%E7%94%A8%E7%8E%87%EF%BC%8C%E4%BB%A5/                        
                        
                        
## Reference
1. https://blog.csdn.net/rockingdingo/article/details/55652662
2. https://www.tensorflow.org/programmers_guide/using_gpu#allowing_gpu_memory_growth
3. 
