from functools import partial
import torch.nn as nn

from model import mamba_extractor
from model import mamba_estimation

LOG = 'VFIMamba'
LOCAL = 2

'''==========Model config=========='''
def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4], M=False, version=1,
                      ssm_version=1, ssd_dstate=64, ssd_headdim=64, ssd_chunk=256,
                      **overrides):
    '''结构展开器: 从少量参数推导完整架构dict。
    推导公式与类绑定固定在本文件; 一切实验可变的结构字段通过 **overrides
    从 yaml 的 model 段透传覆盖 multiscalecfg (如 local_cfg / local_hidden_dims)。

    local_cfg 说明 (局部精化各级配置, 每级 [scale, down, c倍率, blocks]):
        不传时 flow_estimation 内部默认 [[2,4,1.0,8],[1,4,1.0,8]]
        (等效原版工作分辨率 1/8 → 1/4)。
        三级含全分辨率示例 (yaml的model段):
            local_cfg: [[2, 4, 1.0, 8], [1, 2, 1.0, 8], [1, 1, 0.5, 4]]
        → 工作分辨率 1/8 → 1/2 → 1/1
        注意: 改动 local_cfg 即改动网络结构, checkpoint 不兼容, 需体现在 exp_name。

    flow_num_stages 默认2, 对应 1/16→1/8 learned-feature heads;
        设为3时再加入 1/4 head, 改动网络结构。
    '''
    # 保留旧模型的两级(1/16,1/8)默认行为; 新训练可在yaml显式设为3,
    # 启用1/4 learned-feature flow head。这样旧checkpoint推理不会悄悄多出随机分支。
    flow_num_stages = int(overrides.pop('flow_num_stages', 2))
    max_flow_stages = len(depth) - 2
    if not 1 <= flow_num_stages <= max_flow_stages:
        raise ValueError(f'flow_num_stages must be in [1,{max_flow_stages}], got {flow_num_stages}')

    backbonecfg = {
        'embed_dims': [(2**i)*F for i in range(len(depth))],
        'motion_dims': [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads': [8*(2**i)*F//32 for i in range(len(depth)-3)],
        'mlp_ratios': [4 for i in range(len(depth)-3)],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': depth,
        'window_sizes': [W for i in range(len(depth)-3)],
        'conv_stages': 3,
        'version': version,
        'ssm_version': ssm_version,   # 1=S6(Mamba1/现状), 2=SSD(Mamba2); 改动=backbone全量重训
        'ssd_dstate': ssd_dstate,
        'ssd_headdim': ssd_headdim,
        'ssd_chunk': ssd_chunk,
    }
    multiscalecfg = {
        'embed_dims': [(2**i)*F for i in range(len(depth))],
        'motion_dims': [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths': depth,
        'num_heads': [8*(2**i)*F//32 for i in range(len(depth)-3)],
        'window_sizes': [W, W],
        'scales': [4*(2**i) for i in range(len(depth)-2)],
        'hidden_dims': [4*F for _ in range(flow_num_stages)],
        'c': F,
        'M': M,
        'local_hidden_dims': 4*F,
        'local_num': 2,          # 兼容保留; local_cfg 存在时由其长度决定, 此字段不再被读取
        # 1=plain Unet, 2=旧全局MHA, 3=低维pooled-KV residual attention
        'version': version,
    }
    multiscalecfg.update(overrides)
    return backbonecfg, multiscalecfg


MODEL_CONFIG = {
    'LOGNAME': LOG,
    'MODEL_TYPE': (mamba_extractor, mamba_estimation),
    'MODEL_ARCH': init_model_config(
        F = 32, #32
        depth = [2, 2, 2, 3, 3],
        M = False,
        version = 1
    )
}
