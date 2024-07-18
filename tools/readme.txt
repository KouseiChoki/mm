安装step:
1.安装anaconda:https://www.anaconda.com/,安装完后关闭即可
2.(下载)打开终端输入 cd ~/Documents/ &&git clone -b latest --depth 1 http://10.35.116.93/mm/motionmodel.git ,如果提示需要安装git，则git安装完后需要再输入一遍
3.安装brew,打开终端输入(需输入密码)：
  国外使用： cd ~/Documents/motionmodel/tools && sudo ./brew.sh
  国内使用： /bin/bash -c "$(curl -fsSL https://gitee.com/ineo6/homebrew-install/raw/master/install.sh)"
  安装完成后输入:brew install openexr
4.(配置)打开终端输入 conda create -n mm python=3.9 出现选项时输入Y
5.(安装)打开终端输入  cd ~/Documents/motionmodel/tools && conda activate mm &&sudo ./install_environment_m2.sh ，需输入密码

用法参考doc下的usage.txt
