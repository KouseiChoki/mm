# 用户手册

## MM安装
通过执行tools/install_environment.sh (mac用户执行tools/install_environment_m2.sh)一键安装

## MM运行方式
在控制台通过输入
```
mm your_config
```
或者
```
python motionmodel/start.py your_config
```
执行MM文件，config文件默认存放于mm根目录下

## config文件
config文件主要分为以下四块：**Main,Mask,Root,Algorithms,Paramaters**
### Main
用户需要在此指定运行环境,cpu代表以cpu方式执行，mps代表使用mac芯片中的显存进行推理，数字代表使用gpu编号进行推理
```
gpu = cpu  
gpu = mps
gpu = 0,1,2
```

其次需要指定framestep来设定输出间隔，framestep=1时只输出间隔为1的MM（mv0，mv1）
比如framestep=1,2,3时 输出(mv0,mv1,mv2,mv3,mv4,mv5)
```
frame_step = 1,2,3
```
### Root
在此选项中，用户需要指定输入文件路径和输出文件路径
```
root = pattern/1k
output = result/1k
```
之后会对该路径下所有scene执行MM推理，文件结构如下图所示：
![Alt text](image-1.png)

如果只需要推理单个scene，可以通过
```
enable_single_input_mode_2D = 1
root_2D =  path_to_your_scene
```
来单独执行MM

### Depth and Disparity
通过设定cal_depth= 1来开启depth模式（3D模式下生成disparity)
```
cal_depth = 1
```

## 3D
通过设定3D_mode = 1来开启3D模式
可以使用enable_single_input_mode来单独指定left_root和right_root目录，否则会默认使用以下结构
![Alt text](image.png)

```
注：使用默认结构时需要指定左右文件夹名称
left_dir_name = left
right_dir_name = right
```


### Algorithms
用户可以通过更改算法名称来更换MM核心算法，支持的算法请参考mmalgo文件中的注释
```
algorithm = kousei
```

如果需要指定其他版本算法，可以添加版本日期，如：
```
algorithm = kousei-v1-221013
```


### Paramater
用户可以通过设定resize_rate来控制图像平滑值
```
resize_rate_x = 1
resize_rate_y = 1
```
这个值控制在0.5-1之间，值越小，所消耗的内存越小，得到的mv越平滑。
对于大图，mv质量欠佳的情况下可以尝试改小来提升平滑性

对于图像有film_border或者扩边的情况，需要添加:
```
film_border = 0,0,0,0
```
来排除边框影响，4个数值分别是，上下边框长度，左右边框宽度


对于mask模式，如果输入mask是非二值的情况，可以通过修改threshold来控制虚边
```
threshold =0
```
threshold取值为0-1，对于前景mask的情况下，0代表外圈，1代表内圈，背景mask则反之


## MM更新方法
执行tools文件夹中的update_mm.sh来更新
```
./tools/update_mm.sh
```