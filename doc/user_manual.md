# User Manual

## MM Installation
Install with one click by executing tools/install_environment.sh (Mac users execute tools/install_environment_m2.sh)

## MM运行方式
In the console, input
```
mm your_config
```
or
```
python motionmodel/start.py your_config
```
to run MM files. The config file is stored in the MM root directory by default.

## Config File
The config file is mainly divided into five sections: **Main,Mask,Root,Algorithms,Paramater**
### Main
Users need to specify the runtime environment here, cpu means execution with the CPU, mps means using the VRAM in the Mac chip for inference, and numbers represent using GPU numbers for inference.
```
gpu = cpu  
gpu = mps
gpu = 0,1,2
```

Next, you need to specify frame_step to set the output interval. When frame_step=1, it only outputs MM at an interval of 1 (mv0, mv1).
For example, with frame_step=1,2,3, it outputs (mv0, mv1, mv2, mv3, mv4, mv5).
```
frame_step = 1,2,3
```
### Root
In this section, users need to specify the input file path and output file path.
```
root = pattern/1k
output = result/1k
```
Then MM inference is performed for all scenes in this path, as shown in the image below:

![Alt text](image-1.png)

To run MM on a single scene, you can use
```
enable_single_input_mode_2D = 1
root_2D =  path_to_your_scene
```

### Depth and Disparity
Enable depth mode (generate disparity in 3D mode) by setting cal_depth= 1
```
cal_depth = 1
```

## 3D
Enable 3D mode by setting 3D_mode = 1.
You can use enable_single_input_mode to specify left_root and right_root directories separately, otherwise, the following structure is used by default.
![Alt text](image.png)

```
###Note: When using the default structure, you need to specify the names of the left and right folders.
left_dir_name = left
right_dir_name = right
```


### Algorithms
Users can change the core MM algorithm by changing the algorithm name. Please refer to the mmalgo in the config file for supported algorithms.
```
algorithm = kousei-v1
```

To specify other versions of the algorithm, you can add the version date, such as:
```
algorithm = kousei-v1-221013
```


### Paramater
Users can control the image smoothing value by setting the resize_rate.
```
resize_rate_x = 1
resize_rate_y = 1
```
This value controls between 0.5-1. The smaller the value, the less memory consumed and the smoother the mv.
For large images, if the mv quality is poor, try reducing it to improve smoothness.

For images with film_border or extended edges, add:
```
film_border = 0,0,0,0
```
to exclude the border effect. The four numbers represent the length of the top and bottom borders and the width of the left and right borders, respectively.

For mask mode, if the input mask is non-binary, modify threshold to control the edge blur.
```
threshold =0
```
The threshold value is 0-1. For foreground masks, 0 represents the outer ring, and 1 represents the inner ring. For background masks, it's the opposite.



## MM update
```
./tools/update_mm.sh
```