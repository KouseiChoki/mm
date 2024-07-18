import torch
import torch.nn as nn
import timm
import numpy as np
#from .twins_ft import _twins_svt_large_jihao
from .twins_mps import twins_svt_large_mps
class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True, del_layers=True,mps=False):
        super().__init__()
        if mps:
            self.svt = twins_svt_large_mps(pretrained=pretrained)
        else:
            self.svt = timm.create_model('twins_svt_large', pretrained=pretrained)

        if del_layers:
            del self.svt.head
            del self.svt.patch_embeds[2]
            del self.svt.patch_embeds[2]
            del self.svt.blocks[2]
            del self.svt.blocks[2]
            del self.svt.pos_block[2]
            del self.svt.pos_block[2]
               
    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            if i == 0:
                x_16 = x.clone()
            if i == layer-1:
                break
        
        return x
    
    def extract_ml_features(self, x, data=None, layer=2):
        res = []
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            x, size = embed(x)
            if i == layer-1:
                x1 = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer-1:
                break
        
        return x1, x
    
    def compute_params(self):
        num = 0

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):
            
            for param in embed.parameters():
                num += np.prod(param.size())
            for param in blocks.parameters():
                num += np.prod(param.size())
            for param in pos_blk.parameters():
                num += np.prod(param.size())
            for param in drop.parameters():
                num += np.prod(param.size())
            if i == 1:
                break
        return num

class convnext_large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.convnext = timm.create_model('convnext_large', pretrained=pretrained)
    
    def forward(self, x, data=None, layer=2):

        x = self.convnext.stem(x)
        x = self.convnext.stages[0](x)
        x = self.convnext.stages[1](x)
        return x
    
    def compute_params(self):
        num = 0

        for param in self.convnext.stem.parameters():
            num += np.prod(param.size())
        for param in self.convnext.stages[0].parameters():
            num += np.prod(param.size())
        for param in self.convnext.stages[1].parameters():
            num += np.prod(param.size())

        return num

class convnext_Xlarge_4x(nn.Module):
    def __init__(self, pretrained=True, del_layers=True):
        super().__init__()
        self.convnext = timm.create_model('convnext_xlarge_in22k', pretrained=pretrained)

        # self.convnext.stem[0].stride = (2, 2)
        # self.convnext.stem[0].padding = (1, 1)

        if del_layers:
            del self.convnext.head
            del self.convnext.stages[1]
            del self.convnext.stages[1]
            del self.convnext.stages[1]
        
        # print(self.convnext)
            
    
    def forward(self, x, data=None, layer=2):

        x = self.convnext.stem(x)
        x = self.convnext.stages[0](x)
        return x

class convnext_base_2x(nn.Module):
    def __init__(self, pretrained=True, del_layers=True):
        super().__init__()
        self.convnext = timm.create_model('convnext_base_in22k', pretrained=pretrained)

        self.convnext.stem[0].stride = (2, 2)
        self.convnext.stem[0].padding = (1, 1)

        if del_layers:
            del self.convnext.head
            del self.convnext.stages[1]
            del self.convnext.stages[1]
            del self.convnext.stages[1]
        
        # print(self.convnext)
            
    
    def forward(self, x, data=None, layer=2):

        x = self.convnext.stem(x)
        x = self.convnext.stages[0](x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        mul = input_dim // 3

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64 * mul)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64 * mul)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64 * mul)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64 * mul, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64 * mul
        self.layer1 = self._make_layer(64 * mul,  stride=2)
        self.layer2 = self._make_layer(96 * mul, stride=2)
        self.layer3 = self._make_layer(128 * mul, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128 * mul, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class BasicEncoder2(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder2, self).__init__()
        self.norm_fn = norm_fn
        mul = input_dim // 3

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64 * mul)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64 * mul)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64 * mul)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64 * mul, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64 * mul
        self.layer1 = self._make_layer(64 * mul,  stride=1)
        self.layer2 = self._make_layer(96 * mul, stride=2)
        self.layer3 = self._make_layer(128 * mul, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128 * mul, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
    


class BasicEncoder3(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder3, self).__init__()
        self.norm_fn = norm_fn
        mul = input_dim // 3

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64 * mul)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64 * mul)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64 * mul)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(input_dim, 64 * mul, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64 * mul
        self.layer1 = self._make_layer(64 * mul,  stride=1)
        self.layer2 = self._make_layer(64 * mul, stride=1)
        self.layer3 = self._make_layer(96 * mul, stride=2)
        self.layer4= self._make_layer(96 * mul, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(96 * mul, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
    

if __name__ == "__main__":
    m = convnext_Xlarge_2x()
    input = torch.randn(2, 3, 64, 64)
    out = m(input)
    print(out.shape)
