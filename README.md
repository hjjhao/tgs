# 笔记


## 第一版
文件list: dataset.py train_1.py predict.py 模型：unet.py

Model: 最原版的UNet

Score：0.70979

## 第二版
文件list：dataset.py train_2.py predict.py 模型：res_unet.py


Model: 使用残差结构的UNet,修改了最底层的BasicBlock

Score: 0.72990

## 第三版
文件list：dataset_3.py train_3.py predict_3.py 模型：res_unet_with_depth.py

为什么感觉没有提升。。。

Model：使用残差结构的UNet,并在较深的层加入了depth信息

Score：0.72502

因为文件越来越多，代码越来越乱，我觉得换个题写。。。顺便再整理下代码# tgs
