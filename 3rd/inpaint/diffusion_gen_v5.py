import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils import Attention, Mlp, MaskAttn2 as MaskAttn

#implement partial convolutions and mat aware transformer stage

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)

def unshuffle(x, r):
    [B, C, H, W] = list(x.size())
    Hr = H//r[0]
    Wr = W//r[1]
    x = x.reshape(B, C, Hr, r[0], Wr, r[1])
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C*(r[0]*r[1]), Hr, Wr)
    return x

def shuffle(x, r):
    [B, C, H, W] = list(x.size())
    x = x.reshape(B, C//(r[0]*r[1]),r[0],r[1], H,W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, -1, H*r[0], W*r[1])
    return x


def unshuffle2B(x, r):
    [B, C, H, W] = list(x.size())
    Hr = H//r[0]
    Wr = W//r[1]
    x = x.reshape(B, C, Hr, r[0], Wr, r[1])
    x = x.permute(0, 3, 5, 1, 2, 4)
    x = x.reshape(B*(r[0]*r[1]), C, Hr, Wr)
    return x

def shuffle2B(x, r):
    [B, C, H, W] = list(x.size())
    x = x.reshape(B//(r[0]*r[1]),r[0],r[1], C, H,W)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(-1, C, H*r[0], W*r[1])
    return x

def Token2TV(x, x_size): # B of tokenized images to torch vision format
    B,N,C = x.shape # tokenized B
    H,W = x_size
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').permute(0,2,1).reshape(B,C,H,W).to(device) # tensor view order of the B
    else:
        x = x.permute(0,2,1).reshape(B,C,H,W)# tensor view order of the B
    return x

def TV2token(x): # B of torch vision images to tokenized format
    B,C,H,W = x.shape # tensor view order of the B
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').view(B,C,-1).transpose(1,2).to(device) # tokenized B
    else:   
        x = x.view(B,C,-1).transpose(1,2) # tokenized B
    return x

class LeakyHardTanh(nn.Module):
    def __init__(self,min_val=0,max_val=1,slope=0.05):
        super().__init__()
        self.ht = nn.Hardtanh(min_val=min_val*(1-slope),max_val=max_val*(1-slope))
        self.slope = slope
    def forward(self, x):
        x = x*self.slope+self.ht(x*(1-self.slope))

""" partial convolutions. When stride = 2, the PConv2dE version is used. When stride = 1, the PConv2d is used"""
class PConv2dE(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=False):
        super(PConv2dE, self).__init__()
        """the even kernel size version is used for downsampling. Wtih stride = 2, the effect is a LPF + 2x2 to 1 pixel 
            mapping in the down conversion. This will match an upconversion process where you use pixel shuffle which
            maps 1x1 to a 2x2 output"""
        # whether the mask is multi-channel or not
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.outCh = out_channels
        self.inCh = in_channels
        self.cf0 = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False)
        #self.cf1 = torch.ones_like(self.cf0.weight)
        #self.cf1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=True)
        self.wmp2D = nn.MaxPool2d(2,stride=stride,padding=0)
        
    def forward(self, input, mask=None):
        if self.cf0.weight.type() != input.type():
            self.cf0.weight = self.cf0.weight.to(input)
        #if self.cf1.weight.type() != input.type():
        #    self.cf1.weight = self.cf1.weight.to(input)
        if mask is not None:
            with torch.no_grad():
                maxMask = self.wmp2D(mask)
            update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), self.cf0.weight.abs(), bias=None, stride=self.stride, padding=self.padding)
            #denom = F.relu(update_mask) + 1e-6
            denom = update_mask + 1e-6
            raw_numer = self.cf0(input*mask)
            output = raw_numer/denom
            return output, maxMask
        else:
            output = self.cf0(input)
            return output, mask


class PConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=False):
        super(PConv2d, self).__init__()
        """The plain version is more like the version in the literature. It can use either an odd or even kernel. The
            window for the mask is based on the kernel size instead of being fixed at 2x2 """
        # whether the mask is multi-channel or not
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.outCh = out_channels
        self.inCh = in_channels
        self.cf0 = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False)
        #self.cf1 = torch.ones_like(self.cf0.weight)
        #self.cf1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=True)
        self.cmp2D = nn.MaxPool2d(kernel_size,stride=stride,padding=padding)

        
    def forward(self, input, mask=None):
        if self.cf0.weight.type() != input.type():
            self.cf0.weight = self.cf0.weight.to(input)
        #if self.cf1.weight.type() != input.type():
        #    self.cf1.weight = self.cf1.weight.to(input)
        if mask is not None:
            with torch.no_grad():
                maxMask = self.cmp2D(mask)
                #update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), self.cf1, bias=None, stride=self.stride, padding=self.padding)
            #update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), self.cf1.weight, bias=self.cf1.bias, stride=self.stride, padding=self.padding)
            update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), self.cf0.weight.abs(), bias=None, stride=self.stride, padding=self.padding)
            denom = F.relu(update_mask) + 1e-6
            raw_numer = self.cf0(input*mask)
            output = raw_numer/denom
            return output, maxMask
        else:
            output = self.cf0(input)
            return output, mask

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class AttnBlock(nn.Module):
    """
    An attention block followed by MLP
    """
    def __init__(self, t_dim, hidden_size, num_heads, skip=True, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.skip = skip
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop,proj_drop=drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class MaskAttnBlock(nn.Module):
    """
    An attention block followed by MLP
    """
    def __init__(self, t_dim, hidden_size, num_heads, skip=True, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.skip = skip
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MaskAttn(hidden_size, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop,proj_drop=drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class AttnLayer(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.pSize = patchSize[0] * patchSize[1]
        if self.pSize == 1:
            self.patch_en = False
        else:
            self.patch_en = True
        self.conv_pin = nn.Conv2d(channels,hidden_size,1) # essentially a bottleneck through the attention
        self.conv_pout = nn.Conv2d(hidden_size,channels,1)
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.ModuleList([
            AttnBlock(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                      mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten)
            ])
        self.patchSize = patchSize
    def forward(self, x,t):
        h = self.act_fn(self.normal_1(x))
        if self.patch_en:
            h = unshuffle2B(h,self.patchSize)
            h = self.conv_pin(h)
        else:
            h = self.conv_pin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        pt = t.view(t.shape[0],1,t.shape[1])
        pt = pt.expand(-1,self.pSize,-1)
        pt = pt.reshape(-1,pt.shape[-1])
        for block in self.attn:
            h = block(h,pt) 
        h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_pout(h)
            h = shuffle2B(h,self.patchSize)
        else:
            h = self.conv_pout(h)
        return h    

class MaskAttnLayer(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.num_atten = num_atten
        self.conv_pin = nn.Conv2d(channels,hidden_size,1) # essentially a bottleneck through the attention
        self.conv_pout = nn.Conv2d(hidden_size,channels,1)
        self.pSize = patchSize[0]*patchSize[1]
        self.patchSize = patchSize
        if self.pSize == 1:
            self.patch_en = False
        else:
            self.patch_en = True
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.maskAttn = MaskAttnBlock(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                      mlp_ratio=mlp_ratio,drop=drop)
        if num_atten > 1:
            self.attn = nn.ModuleList([
                AttnBlock(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                        mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten-1)
                ])
        else:
            self.attn = nn.ModuleList([
                ])
        self.patchSize = patchSize
    def forward(self, x,t,m=None):
        h = self.act_fn(self.normal_1(x))
        if self.patch_en:
            h = unshuffle2B(h,self.patchSize)
            h = self.conv_pin(h)
            if m is not None:
                m = unshuffle2B(m,self.patchSize)
        else:
            h = self.conv_pin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        if m is not None:
            m = TV2token(m)
        pt = t.view(t.shape[0],1,1,t.shape[1])
        pte = pt.expand(-1,self.patchSize[0],self.patchSize[1],-1).clone()
        pte = pte.reshape(t.shape[0]*self.patchSize[0]*self.patchSize[1],t.shape[1])
        if self.num_atten > 0: # this should always be true if this class is used!
            h = self.maskAttn(h,pte,m)
        if self.num_atten > 1:
            for block in self.attn:
                h = block(h,pte) 
        h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_pout(h)
            h = shuffle2B(h,self.patchSize)
        else:
            h = self.conv_pout(h)
        return h    
    
class ResnetBlock(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = AttnLayer(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
                                       num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout_rate)

    def forward(self, x, t):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        # add in timestep embedding
        t_emb = self.dense_1(self.act_fn(t))[:, :, None, None]
        scale, shift = torch.chunk(t_emb, 2, dim=1)
        h = self.normlize_2(h) * (1 + scale) + shift
            
        # group 3
        h = self.act_fn(h)
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + x
        if self.apply_attention:
            h = self.attention(h,t)

        return h
class MaskResnetBlock(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = PConv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = PConv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1)
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = MaskAttnLayer(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
                                       num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout_rate)

    def forward(self, x, t, m=None):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h, _m = self.conv_1(h,m) # stride = 1 so mask doesn't change

        # group 2
        # add in timestep embedding
        t_emb = self.dense_1(self.act_fn(t))[:, :, None, None]
        scale, shift = torch.chunk(t_emb, 2, dim=1)
        h = self.normlize_2(h) * (1 + scale) + shift
            
        # group 3
        h = self.act_fn(h)
        h = self.dropout(h)
        h,_m = self.conv_2(h,m)

        # Residual and attention
        h = h + x
        if self.apply_attention:
            h = self.attention(h,t)
            #m = None # no more masks

        return h, m

class ResnetDwn(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.block1 = ResnetBlock(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlock(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t):
        h = self.act_fn(self.normlize_1(x))
        h = self.conv1(x)
        h = self.block1(h,t)
        h = self.block2(h,t)
        return h

class MaskResnetDwn(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = PConv2dE(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.block1 = MaskResnetBlock(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = MaskResnetBlock(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t, m=None):
        h = self.act_fn(self.normlize_1(x))
        h,m = self.conv1(x,m)
        h,m = self.block1(h,t,m)
        h,m = self.block2(h,t,m)
        return h,m

class ResnetUp(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=2*in_channels)
        self.block1 = ResnetBlock(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlock(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels, kernel_size=3, stride=1, padding=1)
        self.act_fn = nn.SiLU()
        self.ps = nn.PixelShuffle(2)
    def forward(self, x, y, t):
        h = torch.concat([x,y],dim=1)
        h = self.act_fn(self.normlize_1(h))
        h = self.conv_in(h)
        h = self.block1(h,t)
        h = self.block2(h,t)
        h = self.act_fn(self.normlize_2(h))
        h = self.conv_out(h)
        h = self.ps(h)
        return h

class MaskResnetUp(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False, no_mask = True):
        super().__init__()
        self.no_mask = no_mask
        if self.no_mask:
            self.conv_in = nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        else:
            self.conv_in = nn.Conv2d(in_channels=2*in_channels+2, out_channels=in_channels, kernel_size=1, stride=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=2*in_channels)
        self.block1 = ResnetBlock(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlock(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels, kernel_size=3, stride=1, padding=1)
        self.act_fn = nn.SiLU()
        self.ps = nn.PixelShuffle(2)
    def forward(self, x, y, t,m=None):
        # tried with and without the below. The next two lines seem to help
        if not self.no_mask:
           x = x*m + y*(1-m) # first make sure all of x has "valid" data
        h = torch.concat([x,y],dim=1)
        h = self.act_fn(self.normlize_1(h))
        if not self.no_mask: # attention is applied symetrically, so once
            h = torch.concat([h,m,1-m],dim=1) # concat the mask so conv is mask aware
        h = self.conv_in(h)
        h = self.block1(h,t)
        h = self.block2(h,t)
        h = self.act_fn(self.normlize_2(h))
        h = self.conv_out(h)
        h = self.ps(h)
        return h

class Model(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512,):
        super().__init__()
        self.t_embedder = TimestepEmbedder(time_emb_dims)
        self.econv6 = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5 = ResnetDwn(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5])
        self.enc4 = ResnetDwn(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4])
        self.enc3 = ResnetDwn(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3])
        self.enc2 = ResnetDwn(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2])
        self.enc1 = ResnetDwn(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1])
        self.enc0 = ResnetDwn(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0])
        self.mid = AttnLayer(num_atten=con_attn_num,patchSize=patchSizes[0],channels=base_channels*channel_mult[0],hidden_size=attn_channels,
                                t_dim=time_emb_dims,num_heads=num_heads,mlp_ratio=mlp_ratio)
        self.dec0 = ResnetUp(base_channels*channel_mult[0],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0])
        self.dec1 = ResnetUp(base_channels*channel_mult[1],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1])
        self.dec2 = ResnetUp(base_channels*channel_mult[2],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2])
        self.dec3 = ResnetUp(base_channels*channel_mult[3],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3])
        self.dec4 = ResnetUp(base_channels*channel_mult[4],base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4])
        self.dec5 = ResnetUp(base_channels*channel_mult[5],base_channels,dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5])
        self.dconv6 = nn.Conv2d(base_channels,3,3,stride=1,padding=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.act_fn = nn.SiLU()
    
    def forward(self, x, t):
        t = self.t_embedder(t) 
        e6 = self.econv6(x)
        e5 = self.enc5(e6,t)
        e4 = self.enc4(e5,t)
        e3 = self.enc3(e4,t)
        e2 = self.enc2(e3,t)
        e1 = self.enc1(e2,t)
        e0 = self.enc0(e1,t)
        d0 = self.mid(e0,t)
        d1 = self.dec0(d0,e0,t)
        d2 = self.dec1(d1,e1,t)
        d3 = self.dec2(d2,e2,t)
        d4 = self.dec3(d3,e3,t)
        d5 = self.dec4(d4,e4,t)
        out = self.dec5(d5,e5,t)
        out = self.act_fn(self.normlize_1(out))
        out = self.dconv6(out)
        return out

class MaskModel(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512,):
        super().__init__()
        self.t_embedder = TimestepEmbedder(time_emb_dims)
        self.econv6 = PConv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5 = MaskResnetDwn(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5])
        self.enc4 = MaskResnetDwn(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4])
        self.enc3 = MaskResnetDwn(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3])
        self.enc2 = MaskResnetDwn(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2])
        self.enc1 = MaskResnetDwn(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1])
        self.enc0 = MaskResnetDwn(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0])
        self.mid = AttnLayer(num_atten=con_attn_num,patchSize=patchSizes[0],channels=base_channels*channel_mult[0],hidden_size=attn_channels,
                                t_dim=time_emb_dims,num_heads=num_heads,mlp_ratio=mlp_ratio)
        self.dec0 = MaskResnetUp(base_channels*channel_mult[0],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0], no_mask =attn_layers[0])
        self.dec1 = MaskResnetUp(base_channels*channel_mult[1],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1], no_mask =attn_layers[1])
        self.dec2 = MaskResnetUp(base_channels*channel_mult[2],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2], no_mask =attn_layers[2])
        self.dec3 = MaskResnetUp(base_channels*channel_mult[3],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3], no_mask =attn_layers[3])
        self.dec4 = MaskResnetUp(base_channels*channel_mult[4],base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4], no_mask =attn_layers[4])
        self.dec5 = MaskResnetUp(base_channels*channel_mult[5],base_channels,dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5], no_mask =attn_layers[5])
        self.dconv6 = nn.Conv2d(base_channels,3,3,stride=1,padding=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.act_fn = nn.SiLU()
    
    def forward(self, x, t, m):
        t = self.t_embedder(t) 
        e6,m6 = self.econv6(x,m)
        e5,m5 = self.enc5(e6,t,m)
        e4,m4 = self.enc4(e5,t,m5)
        e3,m3 = self.enc3(e4,t,m4)
        e2,m2 = self.enc2(e3,t,m3)
        e1,m1 = self.enc1(e2,t,m2)
        e0,m0 = self.enc0(e1,t,m1)
        d0 = self.mid(e0,t)
        d1 = self.dec0(d0,e0,t,m0)
        d2 = self.dec1(d1,e1,t,m1)
        d3 = self.dec2(d2,e2,t,m2)
        d4 = self.dec3(d3,e3,t,m3)
        d5 = self.dec4(d4,e4,t,m4)
        out = self.dec5(d5,e5,t,m5)
        out = self.act_fn(self.normlize_1(out))
        out = self.dconv6(out)
        return out

class MaskDwn(nn.Module):
    def __init__(self, inCh, outCh, kernel, stride, padding=0, bias=True, num_groups=1):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=num_groups, num_channels=inCh)
        self.act_fn = nn.SiLU()
        self.mact_fn = LeakyHardTanh(min_val=0,max_val=1,slope=0.05)
        self.ienc = nn.Conv2d(inCh,outCh,kernel,stride=stride,padding=padding,bias=bias)
        self.menc = nn.Conv2d(inCh,outCh,kernel,stride=stride,padding=padding,bias=bias)

    def forward(self,x,m):
        x = self.act_fn(self.normlize_1(x))
        x = self.ienc(x)
        m = self.mact_fn(m)
        m = self.menc(m)
        return x,m

class MaskUp(nn.Module):
    def __init__(self, inCh, outCh, kernel, stride, padding=0, bias=True, num_groups=1):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=num_groups, num_channels=2*inCh)
        self.normlize_2 = nn.GroupNorm(num_groups=num_groups, num_channels=inCh)
        self.act_fn = nn.SiLU()
        self.mact_fn = LeakyHardTanh(min_val=0,max_val=1,slope=0.05)
        self.dec = nn.Conv2d(inCh+2,4*outCh,kernel,stride=stride,padding=padding,bias=bias)
        self.act_fn = nn.SiLU()
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, y, t,m=None):
        # tried with and without the below. The next two lines seem to help
        if not self.no_mask:
           x = x*m + y*(1-m) # first make sure all of x has "valid" data
        h = torch.concat([x,y],dim=1)
        h = self.act_fn(self.normlize_1(h))
        h = torch.concat([h,m,1-m],dim=1) # concat the mask so conv is mask aware
        h = self.dec(h)
        h = self.ps(h)
        return h


class PmaskModel(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512,):
        super().__init__()
        self.t_embedder = TimestepEmbedder(time_emb_dims)        
        self.enc6 = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5 = MaskDwn(base_channels,base_channels*channel_mult[5],3,stride=1,padding=1,num_groups=4)
        self.enc4 = MaskDwn(base_channels*channel_mult[5],base_channels*channel_mult[4],3,stride=1,padding=1,num_groups=4)
        self.enc3 = MaskDwn(base_channels*channel_mult[4],base_channels*channel_mult[3],3,stride=1,padding=1,num_groups=8)
        self.enc2 = MaskDwn(base_channels*channel_mult[3],base_channels*channel_mult[2],3,stride=1,padding=1,num_groups=8)
        self.MAT2 = MaskAttnLayer(num_atten=num_atten,patchSize=patchSizes,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
                                       num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout_rate)
        self.enc1 = MaskDwn(base_channels*channel_mult[2],base_channels*channel_mult[1],3,stride=1,padding=1,num_groups=8)
        self.enc0 = MaskDwn(base_channels*channel_mult[1],base_channels*channel_mult[0],3,stride=1,padding=1,num_groups=8)

class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device
        self.initialize()
 
    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
         
        self_sqrt_beta                       = torch.sqrt(self.beta)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
          
    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
    
    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor, noise2use=None):
        if noise2use is None:
            eps = torch.randn_like(x0)  # Noise
        else:
            eps = noise2use
        mean    = get(self.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
        std_dev = get(self.sqrt_one_minus_alpha_cumulative, t=timesteps) # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise

        return sample, eps  # return ... , gt noise --> model predicts this
