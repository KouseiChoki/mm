s_DIR=$(cd "$(dirname "$0")"; pwd)
###
 # @Author: Qing Hong
 # @Date: 2023-12-12 13:16:48
 # @LastEditors: Qing Hong
 # @LastEditTime: 2024-07-19 10:30:53
 # @Description: file content
### 
PARENT_DIR=$(dirname "$s_DIR")
if grep -q "mm()" ~/.zshrc; then
    echo "命令 'mm()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mm()' 到 ~/.zshrc。"
    echo 'mm(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python start.py "$@"
            }' >> ~/.zshrc
fi

if grep -q "mmexrreader()" ~/.zshrc; then
    echo "命令 'mmexrreader()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmexrreader()' 到 ~/.zshrc。"
    echo 'mmexrreader(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python algo/conversion_tools/exr_processing/cal_mv.py "$@"
    }' >> ~/.zshrc
fi

if grep -q "mmevaluation()" ~/.zshrc; then
    echo "命令 'mmevaluation()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmevaluation()' 到 ~/.zshrc。"
    echo 'mmevaluation(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python algo/conversion_tools/MM_evaluate.py "$@"
    }' >> ~/.zshrc
fi

if grep -q "mmalgo()" ~/.zshrc; then
    echo "命令 'mmalgo()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmalgo()' 到 ~/.zshrc。"
    echo 'mmalgo(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python algo/conversion_tools/algorithm_center.py
    }' >> ~/.zshrc
fi

if grep -q "mmupdate()" ~/.zshrc; then
    echo "命令 'mmupdate()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmupdate()' 到 ~/.zshrc。"
    echo 'mmupdate(){
        conda activate mm
        cd '"$PARENT_DIR"'
        git reset --hard && git pull
        ./tools/shortcut_zsh.sh 
        ./tools/shortcut_bp.sh 
    }' >> ~/.zshrc
fi


if grep -q "mmboundingboxreader()" ~/.zshrc; then
    echo "命令 'mmboundingboxreader()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmboundingboxreader()' 到 ~/.zshrc。"
    echo 'mmboundingboxreader(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python algo/conversion_tools/MM_boundingbox_reader.py "$@"
    }' >> ~/.zshrc
fi
if grep -q "mmtrain()" ~/.zshrc; then
    echo "命令 'mmtrain()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmtrain()' 到 ~/.zshrc。"
    echo 'mmtrain(){
        conda activate mm
        cd '"$PARENT_DIR"'
        cd 3rd/Kousei
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        python train_start.py "$@"
    }' >> ~/.zshrc
fi
if grep -q "mmtb()" ~/.zshrc; then
    echo "命令 'mmtb()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmtb()' 到 ~/.zshrc。"
    echo 'mmtb(){
        conda activate mm
        tensorboard --logdir '"$PARENT_DIR"'/train/logs --port 1234
    }' >> ~/.zshrc
fi
if grep -q "mmcolortrans()" ~/.zshrc; then
    echo "命令 'mmcolortrans()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmcolortrans()' 到 ~/.zshrc。"
    echo 'mmcolortrans(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python algo/conversion_tools/exr_processing/color_convertion/color_transform.py  "$@"
    }' >> ~/.zshrc
fi

if grep -q "mmevaluate()" ~/.zshrc; then
    echo "命令 'mmevaluate()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmevaluate()' 到 ~/.zshrc。"
    echo 'mmevaluate(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python evaluation/mm_evaluate.py "$@"
    }' >> ~/.zshrc
fi

if grep -q "mmzipsplit()" ~/.zshrc; then
    echo "命令 'mmzipsplit()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmzipsplit()' 到 ~/.zshrc。"
    echo 'mmzipsplit(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python algo/conversion_tools/zip_slice.py "$@"
    }' >> ~/.zshrc
fi

# source ~/.zshrc
if grep -q "mmply()" ~/.zshrc; then
    echo "命令 'mmply()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmply()' 到 ~/.zshrc。"
    echo 'mmply(){
        conda activate mm
        cd '"$PARENT_DIR"'
        python algo/conversion_tools/pointcloud/cal_ply.py "$@"
    }' >> ~/.zshrc
fi

# source ~/.zshrc
if grep -q "mmd()" ~/.zshrc; then
    echo "命令 'mmd()' 已存在于 ~/.zshrc 中，跳过添加。"
else
    # 如果命令不存在，则添加到 ~/.zshrc
    echo "添加命令 'mmd()' 到 ~/.zshrc。"
    echo 'mmd(){
        conda activate mm
        cd '"$PARENT_DIR"'
        export PYTORCH_ENABLE_MPS_FALLBACK=1 && python 3rd/depth/metric_depth/start.py "$@"
    }' >> ~/.zshrc
fi
echo "patch 3.19note 新版需要手动安装opencolorio,在控制台输入mm,左侧有(mm)的情况下输入pip install opencolorio"
echo "patch 7.09note 新版需要手动安装plyfile,在控制台输入mm,左侧有(mm)的情况下输入pip install plyfile"
echo "patch 7.19note 新版mmd需要升级matplotlib,左侧有(mm)的情况下输入pip install matplotlib==3.9.1"
