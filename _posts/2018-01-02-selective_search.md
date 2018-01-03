---
layout:     post
title:      region_proposal算法(二)
subtitle:   SelectiveSearch
date:       2017-12-26
author:     zihuaweng
header-img: img/post_bg_region.png
catalog: true
tags:
    - Object_detection
    - region_proposal
    - SelectiveSearch
    - EdgeBox
---

## SelectiveSearch

## 算法概况


- **解决问题**：因为图像是多层次的，一张图片可能包含一个桌子，桌子上面有一个碗，碗里面有一个勺子，所以描述一张图片需要做图像分割，需要把他们都区别出来是很难的。而且我们还需在分割时保持物体的完整性，比如：能够将不同颜色的猫咪分割成一群猫咪，将红色的车身和黑色的轮胎分割成一辆完整的车。

- **SelectiveSearch**： SelectiveSearch则结合了exhaustive search:（尽可能找到画中所有的物体部分）和segmentation（将不同物体分割出来），首先用 graph-based的分割方法将图片打散成像素级别的小块（oversegments），如下图：
![segmentation](http://zihuaweng.github.io/post_images/selective_search/segmentation.png)

然后给每块区域加上bounding boxes，自下而上的根据颜色、纹理、大小及形状兼容性聚合这些小块，再重复这些步骤知道没有可以合并的小块为止，如下图。
![bottom_up](http://zihuaweng.github.io/post_images/selective_search/bottom_up.png)

结果可以得到1000-2000个不等的bounding boxes，这样比传统的滑窗方式recall高很多。

- **特点**： 
	- **能够识别图片中不同大小的物体**，可以使用hierarchical算法完成
	- **使用多种方式聚合元素**。物体可根据颜色，纹理或者轮廓等各种标准区分开，因为不同场景下，物体有可能仅因某种特征区分开。
	- **计算速度快**

### Hierarchical grouping （分割算法）
hierarchical grouping + Bottom-up grouping （打散成小块后自下而上的合并）
1. 首先将图片切割成一个个小快作为初始区域[2]，这样每一块不仅含有的信息比较多，而且不太可能跨度多个物体。
2. 用一个贪婪算法将这些小块组合起来：
	- 计算相邻小块的相似度，两个最相似的小块合并，接着计算组合后的所有小块之间的相似度，重复操作直到所有小块合并成原始图片
![hierarchical](http://zihuaweng.github.io/post_images/selective_search/hierarchical.png)

### Diversification Strategies （合并策略）
使用以下三种不同方式聚合元素
- 不同色彩空间： 包含各种一定程度的不变属性的色彩空间,RGB,HSV等
- 不同计算相似度方法： 整合：
	- 颜色相似度\\( s_{color} \\)（直方图）
	- 纹理相似度\\( s_{texture} \\)（fast SIFT-like， 纹理直方图）
	- 面积\\( s_{size} \\) (面积较小的小块优先聚合：相似大小的小块聚合，这样聚合会在图片的各个局部发生)
	- 匹配\\( s_{fill} \\) (有重合部分的小块优先合并)四种相似度计算方法。
\\[ s = a_1s_{color} + a_2s_{texture} + a_3s_{size} + a_4s_{fill} \\]
（具体的计算公式可以参考selectivesearch的文章）

### 最后结果
![result_1](http://zihuaweng.github.io/post_images/selective_search/result_1.png)
![result_2](http://zihuaweng.github.io/post_images/selective_search/result_2.png)

## 代码实现
python有一个selectivesearch 的库可以直接使用，这里使用opencv实现，主要是笔者用自己的数据测试后opencv的实现效果比较好，主要是速度更快。
~~~~~ python
#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''
 
import sys
import cv2
 
if __name__ == '__main__':
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
 
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);
 
    # read image
    im = cv2.imread(sys.argv[1])
    # resize image
    newHeight = 200
    newWidth = int(im.shape[1]*200/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))    
 
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
 
    # Switch to fast but low recall Selective Search method
    if (sys.argv[2] == 'f'):
        ss.switchToSelectiveSearchFast()
 
    # Switch to high recall but slow Selective Search method
    elif (sys.argv[2] == 'q'):
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    else:
        print(__doc__)
        sys.exit(1)
 
    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))
     
    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 50
 
    while True:
        # create a copy of original image
        imOut = im.copy()
 
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break
 
        # show output
        cv2.imshow("Output", imOut)
 
        # record key press
        k = cv2.waitKey(0) & 0xFF
 
        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
~~~~~
其中ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation(), switchToSelectiveSearchQuality(), switchToSelectiveSearchFast()可以根据需求设置适合的k, segma。具体参考[opencv](https://docs.opencv.org/trunk/d6/d6d/classcv_1_1ximgproc_1_1segmentation_1_1SelectiveSearchSegmentation.html#a53c44312781ded2945c4abb1aa650351)

选中图片按m可以查看更多框,l可以查看少一点框,q退出

## Reference:
1. [Segmentation as Selective Search for Object Recognition](https://www.koen.me/research/selectivesearch/)
2. [Efficient Graph-Based Image Segmentation](http://cs.brown.edu/~pff/segment/)
3. [Learn Open CV](http://www.learnopencv.com/selective-search-for-object-detection-cpp-python/)
