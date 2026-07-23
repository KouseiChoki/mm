import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .warplayer import warp
from .refine import *

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17, zero_init=False,
                 compact_feature=False):
        super(Head, self).__init__()
        self.scale = scale
        feature_channels = in_planes * 2 // (4 * 4)
        if compact_feature:
            # 1/4 feature head不先PixelShuffle到全分辨率。用1x1投影压通道,
            # 在1/4网格完成匹配/残差预测, 再上采样flow, 避免全分辨率192c卷积。
            self.feature_transform = nn.Sequential(
                nn.Conv2d(in_planes * 2, feature_channels, 1, 1, 0),
                nn.PReLU(feature_channels),
            )
            self.work_scale = scale
        else:
            self.feature_transform = nn.Sequential(
                nn.PixelShuffle(2), nn.PixelShuffle(2))
            self.work_scale = scale // 4
        self.conv = nn.Sequential(
                                  conv(feature_channels + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  
        if zero_init:
            nn.init.zeros_(self.conv[-1][0].weight)
            nn.init.zeros_(self.conv[-1][0].bias)

    def forward(self, motion_feature, x, flow): 
        motion_feature = self.feature_transform(motion_feature)
        if self.work_scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.work_scale,
                              mode="bilinear", align_corners=False)
        if flow is not None:
            if self.work_scale != 1:
                flow = F.interpolate(flow, scale_factor=1. / self.work_scale,
                                     mode="bilinear", align_corners=False)
                flow = flow * (1. / self.work_scale)
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.work_scale != 1:
            x = F.interpolate(x, scale_factor=self.work_scale,
                              mode="bilinear", align_corners=False)
        flow = x[:, :4] * self.work_scale
        mask = x[:, 4:5]
        return flow, mask

class IFBlock(nn.Module):
    """局部精化块。

    scale : 输入预下采样倍率 (2=在1/2分辨率图像上工作, 1=全分辨率图像)
    down  : conv主体的内部下采样倍率:
              4 = 原版行为 (conv0两次stride2, 卷积主体在 输入/4 分辨率)
              2 = 浅下采样 (卷积主体在 输入/2 分辨率, 小物体细节保留更多)
              1 = 零下采样 (卷积主体在 输入原分辨率, 计算/显存开销最大)
    blocks: convblock 层数 (down=1 时建议减少以控制全分辨率下的计算量)

    卷积主体的实际工作分辨率 = 全分辨率 / (scale * down)。
    """

    def __init__(self, in_planes, c, scale, down=4, blocks=8, zero_init=False):
        super(IFBlock, self).__init__()
        assert down in (1, 2, 4), f'down must be 1/2/4, got {down}'
        self.scale = scale
        self.down = down

        if down == 4:
            self.conv0 = nn.Sequential(
                conv(in_planes, c//2, 3, 2, 1),
                conv(c//2, c, 3, 2, 1),
                )
        elif down == 2:
            self.conv0 = nn.Sequential(
                conv(in_planes, c//2, 3, 2, 1),
                conv(c//2, c, 3, 1, 1),
                )
        else:                                   # down == 1: 全分辨率卷积主体
            self.conv0 = nn.Sequential(
                conv(in_planes, c//2, 3, 1, 1),
                conv(c//2, c, 3, 1, 1),
                )

        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(blocks)])

        if down == 1:
            self.lastconv = nn.Conv2d(c, 5, 3, 1, 1)          # 已在目标分辨率, 无需转置上采样
        else:
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)  # ×2

        if zero_init:
            nn.init.zeros_(self.lastconv.weight)
            nn.init.zeros_(self.lastconv.bias)

        # 预测网格 → 全分辨率 的上采样倍率 (同时也是flow数值的还原倍率):
        #   down>=2: 预测网格 = 全分辨率/(scale*down/2)
        #   down==1: 预测网格 = 全分辨率/scale
        self.up_factor = scale * down // 2 if down > 1 else scale

    def forward(self, x, flow):
        scale = self.scale
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        if self.up_factor != 1:
            tmp = F.interpolate(tmp, scale_factor = self.up_factor, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * self.up_factor
        mask = tmp[:, 4:5]
        return flow, mask
    
class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone
        zero_init_residual_heads = kargs.get('zero_init_residual_heads', False)
        compact_quarter_head = kargs.get('compact_quarter_head', False)
        self.block = nn.ModuleList([Head( kargs['embed_dims'][-1-i], 
                            kargs['scales'][-1-i], 
                            kargs['hidden_dims'][-1-i],
                            7 if i==0 else 18,
                            zero_init=zero_init_residual_heads and i > 0,
                            compact_feature=(compact_quarter_head
                                             and kargs['scales'][-1-i] == 4))
                            for i in range(self.flow_num_stage)])

        # 局部精化配置: 每级 [scale, down, c倍率, blocks]
        #   卷积主体工作分辨率 = 1/(scale*down); flow输入含timestep通道(18)
        #   默认两级: 等效原版工作分辨率 (1/8, 1/4)
        #   开启全分辨率级 (config的multiscalecfg里传):
        #     'local_cfg': [[2, 4, 1.0, 8], [1, 2, 1.0, 8], [1, 1, 0.5, 4]]
        #     → 工作分辨率 1/8 → 1/2 → 1/1; 全分辨率级用半宽通道+4层控制开销
        local_cfg = kargs.get('local_cfg', None)
        if local_cfg is None:
            local_cfg = [[2, 4, 1.0, 8], [1, 4, 1.0, 8]]
        self.local_num = len(local_cfg)
        base_c = kargs['local_hidden_dims']
        local_zero_init = kargs.get('local_zero_init', False)
        self.local_block = nn.ModuleList([
            IFBlock(18, c=max(int(base_c * cr) // 2 * 2, 16), scale=s, down=d,
                    blocks=b, zero_init=local_zero_init)
            for (s, d, cr, b) in local_cfg
        ])

        self.version = int(kargs['version'])
        self.refine_res_scale = float(kargs.get('refine_res_scale', 0.25))
        if not 0.0 <= self.refine_res_scale <= 1.0:
            raise ValueError(
                f'refine_res_scale must be in [0, 1], '
                f'got {self.refine_res_scale}')
        if self.version == 3:
            self.unet = UnetWithResidualAttention(
                kargs['c'] * 2, kargs['M'],
                attn_dim=kargs.get('refine_attn_dim', 512),
                attn_heads=kargs.get('refine_attn_heads', 8),
                kv_pool=kargs.get('refine_kv_pool', 4))
        elif self.version == 2:
            self.unet = UnetWithAttention(kargs['c'] * 2, kargs['M'])
        else:
            self.unet = Unet(kargs['c'] * 2, kargs['M'])

    def _compose_prediction(self, merged, refine_output):
        if self.version == 3:
            res = refine_output * self.refine_res_scale
            pred = merged + res
            # 训练时保留越界像素梯度；评估/推理输出仍限制到合法图像范围。
            if not self.training:
                pred = torch.clamp(pred, 0, 1)
            return res, pred
        res = refine_output[:, :3] * 2 - 1
        return res, torch.clamp(merged + res, 0, 1)

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        return y0, y1

    def calculate_flow(self, imgs, timestep, local=False, af=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        if af is None:
            af = self.feature_bone(img0, img1)
        timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        for i in range(self.flow_num_stage):
            if flow != None:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1),
                    flow
                    )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, timestep), 1),
                    None
                    )

        if local:
            for i in range(self.local_num):
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])

                flow_d, mask_d = self.local_block[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d

        return flow, mask

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        res, pred = self._compose_prediction(merged, tmp)
        return pred,warped_img0,warped_img1,mask_


    def forward(self, x, local=False, timestep=0.5, scale=0):
        if scale > 0: 
            x_o = x
            x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        af = self.feature_bone(img0, img1)
        timestep = (x[:, :1].clone() * 0 + 1) * (timestep.float() if type(timestep) is not float else timestep)
        for i in range(self.flow_num_stage):
            if flow != None:
                flow_d, mask_d = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                            torch.cat((img0, img1, timestep), 1), None)
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        
        if scale>0:
            img0, img1 = x_o[:, :3], x_o[:, 3:6]
            af1 = self.feature_bone(img0, img1)
            scale = img0.shape[3] / flow.shape[3]
            flow = F.interpolate(flow, scale_factor = scale, mode="bilinear", align_corners=False) * scale
            mask = F.interpolate(mask, scale_factor = scale, mode="bilinear", align_corners=False)
            # timestep 常数图与图像尺寸保持同步 (scale>0 + local 时必需)
            timestep = F.interpolate(timestep, scale_factor = scale, mode="bilinear", align_corners=False)
            mask_ = torch.sigmoid(mask)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_ + warped_img1 * (1 - mask_))

        if local:
            # flow_list 保留前面 learned-feature heads, 用于所有阶段的flow监督。
            # mask/merged 仍只返回local阶段, 保持原有图像重建loss量级。
            merged = []
            mask_list = []
            
            for i in range(self.local_num):
                flow_d, mask_d = self.local_block[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d

                mask_list.append(torch.sigmoid(mask))
                flow_list.append(flow)
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        
        if scale: 
            c0, c1 = self.warp_features(af1, flow)
        else:
            c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res, pred = self._compose_prediction(merged[-1], tmp)
        return flow_list, mask_list, res, warped_img0, warped_img1, merged, pred
