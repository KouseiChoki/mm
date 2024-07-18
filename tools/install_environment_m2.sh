chmod +x *
###
 # @Author: Qing Hong
 # @Date: 2023-09-20 12:03:09
 # @LastEditors: QingHong
 # @LastEditTime: 2024-01-25 16:41:57
 # @Description: file content
### 
conda install pytorch::pytorch torchvision torchaudio -c pytorch
# /bin/bash  brew.sh
export CFLAGS="-I/opt/homebrew/include/OpenEXR -I/opt/homebrew/include/Imath" && export LDFLAGS="-L/opt/homebrew/lib"&& pip install -r requirements.txt
python -c "import imageio; imageio.plugins.freeimage.download()"
python -c "import torch; print(torch.backends.mps.is_available());print(torch.backends.mps.is_built())"
./shortcut_zsh.sh
./shortcut_bp.sh
# brew install htop
# htop

