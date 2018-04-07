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

由train_faster_rcnn_alt_opt.py可以看到整个训练过程，分为一下几个部分,下面文章会就这几个训练过程讲解代码：

- Stage 1 RPN, init from ImageNet model
- Stage 1 RPN, generate proposals
- Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model
- Stage 2 RPN, init from stage 1 Fast R-CNN model
- Stage 2 RPN, generate proposals
- Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model

主要是两个循环：训练RPN，用RPN生成的proposal训练Fast R-CNN，再训练RPN，再用RPN生成的proposal训练Fast R-CNN。

我们可以获取所有阶段的pt文件看模型结构，重点看以下几个环节：
RPN train：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/rpn_train.png)
RPN test, proposal生成：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/rpn_test.png)
Fast R-CNN train：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/fastrcnn.png)
Fast R-CNN test：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/faster_test.png)

其中包含了几个重要的layer,下面会展开。

RoIDataLayer, AnchorTargetLayer, ProposalLayer, ProposalTargetLayer, ROIPooing, SmoothL1Loss layer


### RPN train
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/rpn_train.png)
1. vgg提取特征（最后是conv5_3,relu5_3)
2. rpn_conv(3x3卷积,shape=(1,512,h,w))
3. 两个1x1卷积：
- rpn_cls_score，shape=(1,18,h,w)，可以理解为9个anchor在每个像素（h,w）对应的原图上是否是物体的概率
- rpn_bbox_pred，shape=(1,36,h,w)， 可以理解为9个anchor的4个坐标点在每个像素（h,w）对应的原图上的偏差
4. AnchorTargetLayer得到每个anchor对应的label,bbox坐标
5. 分别计算各自的loss（rpn_loss_cls(SoftmaxWithLoss), rpn_loss_bbox(SmoothL1Loss)）

#### AnchorTargetLayer:
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


#### SmoothL1Loss layer
计算损失函数论文使用：
![loss_function](http://zihuaweng.github.io/post_images/faster_rcnn/loss_function.jpg)

- Lreg(ti, t∗i ) = R(ti − t∗i ), R是smoothl1loss
- rpn_bbox_pred  anchor与预测的偏移量 ti
- rpn_bbox_targets  anchor与gt的偏差 t*i
- rpn_bbox_inside_weights p*i

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

### RPN test (generate proposals)
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/rpn_test.png)
1. vgg提取特征（最后是conv5_3,relu5_3)
2. rpn_conv(3x3卷积,shape=(1,512,h,w))
3. 两个1x1卷积：
- rpn_cls_score，shape=(1,18,h,w)，可以理解为9个anchor在每个像素（h,w）对应的原图上是否是物体的概率，包涵下面几个操作
    - rpn_cls_score_reshape(shape=(1,2,9xh,w))，这里的reshape是方便后面softmax计算
    - rpn_cls_pro = softmax(rpn_cls_score_reshape)
    - rpn_cls_pro_reshape(shape=(1,18,h,w)), 前面九个是背景概率，后面就个是前景概率,后面计算会用到的是前景概率
- rpn_bbox_pred，shape=(1,36,h,w)， 可以理解为9个anchor的4个坐标点在每个像素（h,w）对应的原图上的偏差
4. ProposalLayer结合上面结果生成原图上的object proposals

#### ProposalLayer:
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

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
~~~


2. 第一步已经得到具体的位置了，这一步其实只是把超出图片大小的proposal给裁剪成在图片范围内
~~~python
proposals = clip_boxes(proposals, im_info[:2])
im_info = (H, W, scaler)
~~~
3. 将proposal长宽小于阈值的去掉
4. 选择rpn_cls_prob_reshape较大的前pre_nms_topN个proposal(一般为2000个)
5. 非最大抑制，留下after_nms_topN个proposal（一般为300个，阈值为iou0.7）
~~~python
keep 是要保留的anchor的index

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
~~~

### Fast R-CNN train：
![rpn_training](http://zihuaweng.github.io/post_images/faster_rcnn/fastrcnn.png)
1. vgg提取特征，得到conv5_3, relu5_3
1. roi_pool5: 使用原图roi进行ROIPooing，得到pool5
2. pool5进入vgg后两层fc6,fc7
3. 两个全连接层
- cls_score，获取21（20个种类+1个背景）种分类概率
- bbox_pred,获取84个点（21 × 4个坐标点）
4. 分别计算loss（loss_cls(SoftmaxWithLoss), loss_bbox(SmoothL1Loss)）

#### ROIPooing
Forward 源码
首先计算原图rois映射到feature map的坐标，即原始坐标 x spacial_scale(大小为所有stride的乘积分之一)，然后针对pool后每个像素点进行计算，即pool后每个像素点都代表原先的一块区域，这个区域大小为bin_h= roi_height / pooled_ height, bin_w=roi_width / pooled_width。每个像素点的值就是该feature map的区域中最大值，并记录最大值所在的位置。
~~~c++
template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); // conv5_3 feature map
  const Dtype* bottom_rois = bottom[1]->cpu_data(); // 原图roi位置信息
  // Number of ROIs
  int num_rois = bottom[1]->num(); 
  int batch_size = bottom[0]->num(); // conv5_3 feature map的batch size
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    // 记录原图roi对应在conv5_3 feature map的4个坐标点
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    // 计算一个pool后的像素点对应roi feature map多少个像素点
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));
          // height_是conv5_3 height
          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), 
          );
          // width_是conv5_3 width
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }
          // 一个pool像素点的值等于对应conv5_3整个bin_size里最大的值，并且记录该最大值的位置返回
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}
~~~



## Reference:
https://github.com/rbgirshick/caffe-fast-rcnn
https://github.com/rbgirshick/py-faster-rcnn
https://blog.csdn.net/wfei101/article/details/77778462
https://blog.csdn.net/xyy19920105/article/details/50420779
