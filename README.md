# 基于MatConvNet的mnist数据集网络训练及测试demo

### 简介
使用MatConvNet库，针对mnist数据集进行数据打包，并搭建一个网络后训练，最后测试的demo

### 准备
- MATLAB (最好为2014a及以上版本)
- MatConvNet-1.0-bata23

### 文件说明
```create_imdb.m```   对mnist的图片数据打包成imdb的格式  
```create_net.m```    使用dagnn创建一个未训练的初始网络  
```train_net.m```     训练网络  
```getBatch.m```      训练时所需的获取batch的函数  
```test.m```          使用训练好的网络对单张图片做测试  
```acc.m```           使用imdb中的测试集测试训练好的网络的准确率
