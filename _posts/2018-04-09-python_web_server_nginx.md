---
layout:     post
title:      python web sever--nginx
subtitle:   nginx解析
date:       2018-04-12
author:     zihuaweng
header-img: img/post_header.jpg
catalog: true
tags:
    - python web sever
    - nginx
---


开发后的flask项目部署在线上服务器时，可以用以下的一个流程
![process](http://zihuaweng.github.io/post_images/python_web/process.png)

今天先记录什么是nginx。

nginx是一个http的sever，客户端通过http传过来的请求，传入nginx后，转发给WSGI HTTP server（gunicorn）在传给Python APP。

在开发环境, 我们一般直接运行Python服务, 启动了某个端口(一般是5000端口), 然后通过该端口进行开发调试

但线上环境一般不会直接这样提供服务, 一般的线上服务需要通过 Nginx 将外部请求转发到Python服务

- 隐藏python服务, 避免直接将python服务暴露出去,我们可以使用域名来代替ip,端口地址
- 提高web服务的连接处理能力(Nginx)
- 作为反向代理, 提升python整体服务处理能力

nginx还有几个作用：

- nginx可以做负载，监听一个端口，然后获得请求后分发给多个不同的客服（handler），均衡每个服务的请求处理亮。
![nginx](http://zihuaweng.github.io/post_images/python_web/nginx.png)

- 另外主要的一个功能就是接收用户请求，从文件中读取HTML，返回有界面网页。

我们可以配置Nginx如下：
~~~
upstream flask_servers {
    server 127.0.0.1:9889;
}

server {
    listen 80;
    server_name dev.simple-cms.com;

    access_log  /data/logs/nginx/simple_cms_access.log main;
    error_log /data/logs/nginx/simple_cms_error.log debug;

    location / {
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_redirect off;
        proxy_pass  http://flask_servers;
    }

    location ^~ /static {
        root /data/www;
    }
}
~~~

## Reference
1. https://www.u3v3.com/ar/1384
2. http://nginx.org/en/

