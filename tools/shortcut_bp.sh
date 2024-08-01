# s_DIR=$(cd "$(dirname "$0")"; pwd)
###
 # @Author: Qing Hong
 # @FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
 # @LastEditors: Qing Hong
 # @LastEditTime: 2024-08-01 13:31:42
 # @Description: 
 #          ▄              ▄
 #         ▌▒█           ▄▀▒▌     
 #         ▌▒▒▀▄       ▄▀▒▒▒▐
 #        ▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐     ,-----------------.
 #      ▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐     (Wow,kousei's code)
 #    ▄▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▐     `-,---------------' 
 #   ▐▒▒▒▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▒▌  _.-'   ,----------.
 #   ▌▒▒▐▄█▀▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐         (surabashii)
 #  ▐▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▒▒▒▒▒▒▒▀▄▌        `-,--------' 
 #  ▌▒▀▄██▄▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌      _.-'
 #  ▌▀▐▄█▄█▌▄▒▀▒▒▒▒▒▒░░░░░░▒▒▒▐ _.-'
 # ▐▒▀▐▀▐▀▒▒▄▄▒▄▒▒▒▒▒░░░░░░▒▒▒▒▌
 # ▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒░░░░░░▒▒▒▐
 #  ▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌
 #  ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▐
 #   ▀▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄▒▒▒▒▌
 #     ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
 #       ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
 #          ▒▒▒▒▒▒▒▒▒▒▀▀
 # When I wrote this, only God and I understood what I was doing
 # Now, God only knows
### 
# ###
#  # @Author: Qing Hong
#  # @Date: 2024-01-09 18:12:50
#  # @LastEditors: Qing Hong
#  # @LastEditTime: 2024-07-09 15:44:47
#  # @Description: file content
# ### 
# PARENT_DIR=$(dirname "$s_DIR")
# if grep -q "mm()" ~/.bash_profile; then
#     echo "命令 'mm()' 已存在于 ~/.bash_profile 中，跳过添加。"
# else
#     # 如果命令不存在，则添加到 ~/.bash_profile
#     echo "添加命令 'mm()' 到 ~/.bash_profile。"
#     echo 'mm(){
#         conda activate mm
#         cd '"$PARENT_DIR"'
#         python start.py "$@"
#             }' >> ~/.bash_profile
# fi

# if grep -q "mmexrreader()" ~/.bash_profile; then
#     echo "命令 'mmexrreader()' 已存在于 ~/.bash_profile 中，跳过添加。"
# else
#     # 如果命令不存在，则添加到 ~/.bash_profile
#     echo "添加命令 'mmexrreader()' 到 ~/.bash_profile。"
#     echo 'mmexrreader(){
#         conda activate mm
#         cd '"$PARENT_DIR"'
#         python algo/conversion_tools/exr_processing/cal_mv.py "$@"
#     }' >> ~/.bash_profile
# fi

# if grep -q "mmevaluation()" ~/.bash_profile; then
#     echo "命令 'mmevaluation()' 已存在于 ~/.bash_profile 中，跳过添加。"
# else
#     # 如果命令不存在，则添加到 ~/.bash_profile
#     echo "添加命令 'mmevaluation()' 到 ~/.bash_profile。"
#     echo 'mmevaluation(){
#         conda activate mm
#         cd '"$PARENT_DIR"'
#         python algo/conversion_tools/MM_evaluate.py "$@"
#     }' >> ~/.bash_profile
# fi

# if grep -q "mmalgo()" ~/.bash_profile; then
#     echo "命令 'mmalgo()' 已存在于 ~/.bash_profile 中，跳过添加。"
# else
#     # 如果命令不存在，则添加到 ~/.bash_profile
#     echo "添加命令 'mmalgo()' 到 ~/.bash_profile。"
#     echo 'mmalgo(){
#         conda activate mm
#         cd '"$PARENT_DIR"'
#         python algo/conversion_tools/algorithm_center.py
#     }' >> ~/.bash_profile
# fi

# if grep -q "mmboundingboxreader()" ~/.bash_profile; then
#     echo "命令 'mmboundingboxreader()' 已存在于 ~/.bash_profile 中，跳过添加。"
# else
#     # 如果命令不存在，则添加到 ~/.zshrc
#     echo "添加命令 'mmboundingboxreader()' 到 ~/.bash_profile"
#     echo 'mmboundingboxreader(){
#         conda activate mm
#         cd '"$PARENT_DIR"'
#         python algo/conversion_tools/MM_boundingbox_reader.py "$@"
#     }' >> ~/.bash_profile
# fi


# if grep -q "mmupdate()" ~/.bash_profile; then
#     echo "命令 'mmupdate()' 已存在于 ~/.bash_profile 中，跳过添加。"
# else
#     # 如果命令不存在，则添加到 ~/.bash_profile
#     echo "添加命令 'mmupdate()' 到 ~/.bash_profile"
#     echo 'mmupdate(){
#         conda activate mm
#         cd '"$PARENT_DIR"'
#         git reset --hard && git pull
#         ./tools/shortcut_zsh.sh 
#         ./tools/shortcut_bp.sh 
#     }' >> ~/.bash_profile
# fi

# echo "patch 3.19note 新版需要手动安装opencolorio,在控制台输入mm,左侧有(mm)的情况下输入pip install opencolorio"
# echo "patch 3.19note 新版需要手动安装plyfile,在控制台输入mm,左侧有(mm)的情况下输入pip install plyfile"
echo "bash_profile 更新已停用"