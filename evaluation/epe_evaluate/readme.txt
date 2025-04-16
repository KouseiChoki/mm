1.安装 anaconda : https://www.anaconda.com/download
conda env create -f conda_env.yml
conda activate evaluation
如果已拥有mm环境，则跳过上面，直接使用conda activate mm

2.运行 python /Users/qhong/Documents/1117test/MM/mm/evaluation/epe_evaluate/evaluate.py SRC_PATH TARGET_PATH
目标目录需要保持一致，如SRC_PATH指定单scene的image文件,TARGET_PATH也要指定到image。 如若指定根目录（跑多scene），TARGET_PATH中需要有对应文件，详情查看附件图片介绍