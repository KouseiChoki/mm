安装step:
安装过程需要管理员权限，如果没有则需要下所有指令前添加sudo XXXXXX
1.安装anaconda:https://www.anaconda.com/,根据官网提示下载安装

2.(下载)打开终端输入:
cd ~/Documents/ &&git clone https://github.com/KouseiChoki/mm.git
如果提示需要安装git，则git安装完后需要再输入一遍

3.(配置)打开终端输入 conda create -n mm python=3.10 出现选项时输入Y (
需要外网环境，如果因网络连接问题可以切换成其他源：
默认源:
conda config --add channels conda-forge
清华源:
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

切换完毕后再次运行配置指令即可
)

4.(安装)打开终端输入  cd ~/Documents/mm/tools && conda activate mm &&sudo ./install_environment_m2.sh ，需输入密码

用法参考doc下的usage.txt




