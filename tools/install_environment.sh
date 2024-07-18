# conda create -n mm python==3.9
###
 # @Author: Qing Hong
 # @Date: 2023-09-20 12:03:09
 # @LastEditors: QingHong
 # @LastEditTime: 2024-01-15 10:32:07
 # @Description: file content
### 
conda activate mm
pip install --upgrade pip setuptools wheel
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
python -c "import imageio; imageio.plugins.freeimage.download()"
conda install -c conda-forge openexr-python
python -c "import torch; print(torch.cuda.is_available())"
./shortcut_zsh.sh
./shortcut_bp.sh
source ~/.bash_profile
# pip install git+https://github.com/jamesbowman/openexrpython.git
# sudo apt-get install libopenexr-dev zlib1g-dev
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
# brew install openexr

