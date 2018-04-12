---
layout:     post
title:      python web sever--gunicorn
subtitle:   gunicorn使用解析
date:       2018-04-12
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - python web sever
    - gunicorn
---


gunicorn 是一个WSGI HTTP server

    Web Server <=====WSGI=====> Python APP
    
WSGI 其实是一种网关协议, 用以把http请求参数转换为python服务可以读取的内容, 这样在python的框架里能根据请求参数进行不同的业务处理了

Gunicorn 就是这么一种WSGI的实现, 他也是个web server, 可以直接提供对外的web服务, 但在一般的部署中, 他只是作为python服务的运行的容器, 运行和管理python应用程序

通过这个容器, 从Nginx转发的请求就能转发到某个python app了

下面讲一下gunicorn接一些参数设置：

## worker types:

gunicorn有以下几种worker types:
- Sync Workers
- Async Workers
- Tornado Workers
- AsyncIO workers

通过设置worker-class，可以设定为以下几种
sync, gevent, eventlet, tornado, gaiohttp, and gthread, the default worker type is sync.

### Sync Workers：
一个worker每次只会应答一个请求，后面的请求会一个个排队。
![worker1](http://zihuaweng.github.io/post_images/python_web/sync_worker_type1.png)

### Async Workers
这里包含几种：
- Gevent
- Eventlet

这两种都是基于Greenlet库的Python库，可以实现协程。这种模式可以服务等待请求处理结束时，转换成其他的协同程序，等上一个请求结束时返回原来的请求。

这里gevent和eventlet都是用了green threads. 不过这个是program level而不是OS level， 前者会被认为是阻塞性的，需要通过异步io来处理这个问题.

大多数情况下, 我们的服务中, 导致性能低下的原因是I/O, 比如对数据库的读写, 发送Http请求等等, 当进程进行I/O的时候, 是不占用CPU时间的, 这个时候, CPU可以被腾出来处理其他请求，gevent就可以起到这个作用。

![worker2](http://zihuaweng.github.io/post_images/python_web/sync_worker_type2-275x300.png)

### Tornado Workers
这个worker是要和Tornado配合使用的，Tornado是一个python框架，网络库，可以提供异步io非阻塞型模型处理长延时请求。
![worker3](http://zihuaweng.github.io/post_images/python_web/sync_worker_type3-300x230.png)

### AsyncIO Workers
- gthread
- gaiohttp

gaiohttp使用aiohttp库，在服务端和客户端执行异步io操作。支持web socket协议。

gthread是一种全线程worker，worker与线程池保持连接，线程会等待接收请求。

![worker4](http://zihuaweng.github.io/post_images/python_web/asyncio-207x300.png)



## 讲一下web socket
web socket这是html5出的协议，http也是一种协议，但是是不支持持久链接的协议，而web socket是一种可以持续链接的协议。与http是有交集，但是是两个不同的协议。http的一个链接中（keep-alive），可以发送多个request，对应接收各自的response（一一对应的），而且response是被动的，由客户端发起，不能服务器主动发起（ long poll 和 ajax轮询）。

web socket可以请求服务器由http升级为web socket，升级后，web socket服务器就可以主动推送消息到客户端。这样，只需要一次http请求，就做到源源不断的信息传送了。（在程序设计中，这种设计叫做回调，就是有了消息自动通知，不用再次询问）

用nginx等转接客户端的请求的话，请求会被转接到客服（handler），但是客服如果处理不过来的时候，在web socket建立后，可以与nginx建立持久链接，有消息的时候客服告诉nginx，然后nginx统一发送给客户端，这样就不用占用本身速度慢的客服，可以解决客服处理速度慢的问题。



## Reference
1. https://www.spirulasystems.com/blog/2015/01/20/gunicorn-worker-types/
2. https://www.zhihu.com/question/20215561


