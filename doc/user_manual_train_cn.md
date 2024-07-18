<!--
 * @Author: Qing Hong
 * @Date: 2024-03-12 15:20:13
 * @LastEditors: QingHong
 * @LastEditTime: 2024-03-13 13:52:27
 * @Description: file content
-->
# 用户手册

## MM_train安装
请参考MM安装

## MM运行方式
在控制台通过输入
```
mmtrain your_tconfig
```
或者
```
python 3rd/Kousei/train_start.py your_config
```
执行训练，tconfig文件默认存放于motionmodel/train/config目录下

## tconfig文件
tconfig文件主要分为以下四块：**Main,Mask,Root,Algorithms,Paramaters**
### Main
用户需要在此指定运行环境,cpu代表以cpu方式执行，mps代表使用mac芯片中的显存进行推理，cuda使用gpu编号进行推理
```
gpu_type = cpu  
gpu_type = mps
gpu_type = cuda
```

需要先选择算法版本,0为2k训练,1为1k训练
```
version = 0
```
然后需要选择基础算法模型,基础算法模型由算法部门提供
```
initial_config = 0
```

### Training data settings

其次需要指定训练文件目录
可以单独指定训练目录，或者将训练数据放在train/data目录下
```
data_root = data
```


训练数据由下面所有数据混合而成，可以根据需要进行选择
Data weighting and augmentation, the three values are [data weighting, minimum resize factor, maximum resize factor], where the resize calculation formula is 2**n. If it's 0, no augmentation is done; if it's 1, it's doubled, and so on.
```
sintel_clean = [0,1,1.2]
sintel_final = [0,1,1.2]
things_clean = [0,1,1.2]
things_final = [0,1,1.2]
unreal_clean = [1,0,0.1]
unreal_final = [1,0,0.1]
spring_clean = [0,0,0.1]
unreal_MRQ_clean = [1,0,0.1]
unreal_MRQ_final = [0,0,0.1]
```

### Depth and Disparity
通过设定cal_depth= 1来开启depth模式（3D模式下生成disparity)
```
cal_depth = 1
```

## 3D

