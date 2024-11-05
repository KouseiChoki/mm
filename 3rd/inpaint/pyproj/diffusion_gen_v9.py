import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_utils_v2 import Attention, Mlp, PatchEmbed, FFN, FFNM,FFN_v2,FFNM_v2,FFN2D,FFNM2D, Token2TV, TV2token
from diffusion_utils_v2 import MaskAttn2 as AttentionM
from diffusion_utils_v2 import Ptoken2TV,TV2pToken,Wtoken2TV,TV2wToken,unshuffle2B,shuffle2B
from Conv2dPartial import PConv2d,PConv2dE

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate2D(x, shift, scale):
    return x * (1 + scale.unsqueeze(2).unsqueeze(2)) + shift.unsqueeze(2).unsqueeze(2)

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
class AttnBlock2D(nn.Module):
    """
    attention block with patch size
    attention block with window size
    then FFN
    input and output are 2D
    FFN is 2D
    """
    def __init__(self, t_dim, hidden_size, num_heads, pSize, skip=True, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.skip = skip
        self.pSize = pSize
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attnP = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop,proj_drop=drop)
        self.attnW = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop,proj_drop=drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.FFN = FFN2D(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.cat_act = nn.SiLU()
        self.gateF = nn.Conv2d(t_dim,hidden_size,1)
        self.scaleF = nn.Conv2d(t_dim,hidden_size,1)
        self.shiftF = nn.Conv2d(t_dim,hidden_size,1)
        self.gateAp = nn.Conv2d(t_dim,hidden_size,1)
        self.scaleAp = nn.Conv2d(t_dim,hidden_size,1)
        self.shiftAp = nn.Conv2d(t_dim,hidden_size,1)
        self.gateAw = nn.Conv2d(t_dim,hidden_size,1)
        self.scaleAw = nn.Conv2d(t_dim,hidden_size,1)
        self.shiftAw = nn.Conv2d(t_dim,hidden_size,1)
        #self.adaLN_modulation = nn.Sequential(
        #    nn.SiLU(),
        #    nn.Linear(t_dim, 6 * hidden_size, bias=True)
        #)

    def forward(self, x, c):
        B,N,H,W = x.shape
        x_size = [H,W]
        shift_ap = self.shiftAp(c)
        scale_ap = self.scaleAp(c)
        gate_ap = self.gateAp(c)
        shift_aw = self.shiftAw(c)
        scale_aw = self.scaleAw(c)
        gate_aw = self.gateAw(c)
        shift_f = self.shiftF(c)
        scale_f = self.scaleF(c)
        gate_f  = self.gateF(c)
        a = modulate2D(self.norm1(x), shift_ap, scale_ap)
        a = TV2pToken(a,self.pSize)
        a = self.attnP(a)
        x = x + gate_ap.unsqueeze(2).unsqueeze(2) * Ptoken2TV(a,x_size,self.pSize)

        a = modulate2D(self.norm1(x), shift_aw, scale_aw)
        a = TV2wToken(a,self.pSize)
        a = self.attnW(a)
        x = x + gate_aw.unsqueeze(2).unsqueeze(2) * Wtoken2TV(a,x_size,self.pSize)

        x = x + gate_f.unsqueeze(2).unsqueeze(2) * self.FFN(modulate2D(self.norm2(x), shift_f, scale_f))
        return x
    
class AttnLayer2D(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.conv_gin = nn.Conv2d(channels,hidden_size,1) # essentially a bottleneck through the attention
        self.conv_gout = nn.Conv2d(hidden_size,channels,1)
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.ModuleList([
            AttnBlock2D(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,pSize=patchSize,
                      mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten)
            ])
        
    def forward(self, x,t):
        h = self.act_fn(self.normal_1(x))
        h = self.conv_gin(h)
        for block in self.attn:
            h = block(h,t) 
        h = self.conv_gout(h)
        return h    

class AttnBlockF(nn.Module):
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
        self.FFN = FFN(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, x_size, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.FFN(modulate(self.norm2(x), shift_mlp, scale_mlp),x_size)
        return x
    
class AttnBlockF2(nn.Module):
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
        self.FFN = FFN_v2(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, x_size, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.FFN(modulate(self.norm2(x), shift_mlp, scale_mlp),x_size)
        return x

class AttnLayer(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        p_groups = patchSize[0]*patchSize[1]
        if patchSize[0] == patchSize[1] == 1:
            self.conv_gin = nn.Conv2d(channels,hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            self.conv_gout = nn.Conv2d(hidden_size,channels,1,groups=p_groups)
            self.patch_en = False
        else:
            self.conv_pin = nn.Conv2d(channels*patchSize[0]*patchSize[1],hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            self.conv_pout = nn.Conv2d(hidden_size,channels*patchSize[0]*patchSize[1],1,groups=p_groups)
            self.patch_en = True
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
            h = unshuffle(h,self.patchSize)
            h = self.conv_pin(h)
        else:
            h = self.conv_gin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        for block in self.attn:
            h = block(h,t) 
        h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_pout(h)
            h = shuffle(h,self.patchSize)
        else:
            h = self.conv_gout(h)
        return h    
class AttnLayerF(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        p_groups = patchSize[0]*patchSize[1]
        if patchSize[0] == patchSize[1] == 1:
            self.conv_gin = nn.Conv2d(channels,hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            self.conv_gout = nn.Conv2d(hidden_size,channels,1,groups=p_groups)
            self.patch_en = False
        else:
            self.conv_pin = nn.Conv2d(channels*patchSize[0]*patchSize[1],hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            self.conv_pout = nn.Conv2d(hidden_size,channels*patchSize[0]*patchSize[1],1,groups=p_groups)
            self.patch_en = True
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.ModuleList([
            AttnBlockF(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                      mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten)
            ])
        self.patchSize = patchSize
    def forward(self, x,t):
        h = self.act_fn(self.normal_1(x))
        if self.patch_en:
            h = unshuffle(h,self.patchSize)
            h = self.conv_pin(h)
        else:
            h = self.conv_gin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        for block in self.attn:
            h = block(h,size_hw,t) 
        h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_pout(h)
            h = shuffle(h,self.patchSize)
        else:
            h = self.conv_gout(h)
        return h    
class AttnLayerF2(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        p_groups = patchSize[0]*patchSize[1]
        if patchSize[0] == patchSize[1] == 1:
            self.conv_gin = nn.Conv2d(channels,hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            self.conv_gout = nn.Conv2d(hidden_size,channels,1,groups=p_groups)
            self.patch_en = False
        else:
            self.conv_pin = nn.Conv2d(channels*patchSize[0]*patchSize[1],hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            self.conv_pout = nn.Conv2d(hidden_size,channels*patchSize[0]*patchSize[1],1,groups=p_groups)
            self.patch_en = True
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.ModuleList([
            AttnBlockF2(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                      mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten)
            ])
        self.patchSize = patchSize
    def forward(self, x,t):
        h = self.act_fn(self.normal_1(x))
        if self.patch_en:
            h = unshuffle(h,self.patchSize)
            h = self.conv_pin(h)
        else:
            h = self.conv_gin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        for block in self.attn:
            h = block(h,size_hw,t) 
        h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_pout(h)
            h = shuffle(h,self.patchSize)
        else:
            h = self.conv_gout(h)
        return h    

class AttnBlockM(nn.Module):
    """
    An attention block followed by MLP
    """
    def __init__(self, t_dim, hidden_size, num_heads, skip=True, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.skip = skip
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionM(hidden_size, num_heads=num_heads, qkv_bias=True,
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
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class AttnLayerM(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        #p_groups = patchSize[0]*patchSize[1]
        self.pSize = patchSize[0] * patchSize[1]
        self.conv_gin = nn.Conv2d(channels,hidden_size,1) # essentially a bottleneck through the attention
        self.conv_gout = nn.Conv2d(hidden_size,channels,1)
        if patchSize[0] == patchSize[1] == 1:
            #self.conv_gin = nn.Conv2d(channels,hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            #self.conv_gout = nn.Conv2d(hidden_size,channels,1,groups=p_groups)
            self.patch_en = False
        else:
            #self.conv_pin = nn.Conv2d(channels*patchSize[0]*patchSize[1],hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            #self.conv_pout = nn.Conv2d(hidden_size,channels*patchSize[0]*patchSize[1],1,groups=p_groups)
            self.patch_en = True
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.ModuleList([
            AttnBlockM(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                      mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten)
            ])
        self.patchSize = patchSize

    def forward(self, x,t=None,m=None):
        h = self.act_fn(self.normal_1(x))
        if self.patch_en:
            h = unshuffle2B(h,self.patchSize)
            #h = self.conv_pin(h)
            h = self.conv_gin(h)
            if m is not None:
                m = unshuffle2B(m,self.patchSize)
        else:
            h = self.conv_gin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        if m is not None:
            m = TV2token(m)
        # t is based on the batch, so we need to rearrange it to match the fact that we've increased the batch size by the patch size
        if t is not None: # attention requires a category input, so skip attention!
            pt = t.view(t.shape[0],1,t.shape[1]) # add a dimension
            pt = pt.expand(-1,self.pSize,-1) # expand by the patch size
            pt = pt.reshape(-1,pt.shape[-1]) # reshape to reduce the dimension
            for block in self.attn:
                h = block(h,pt,m) 
            h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_gout(h)
            h = shuffle2B(h,self.patchSize)
        else:
            h = self.conv_gout(h)
        return h    

class AttnBlockFM(nn.Module):
    """
    An attention block followed by MLP
    Does not modulate using category or time embedded
    """
    def __init__(self, t_dim, hidden_size, num_heads, skip=True, mlp_ratio=4.0, drop=0.1, catagory=True):
        super().__init__()
        self.skip = skip
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionM(hidden_size, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop,proj_drop=drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.FFNM = FFNM(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.cat = catagory
        if catagory:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_dim, 6 * hidden_size, bias=True)
            )

    def forward(self, x, x_size, c=None, mask=None):
        if self.cat and (c is not None):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
            x = x + gate_mlp.unsqueeze(1) * self.FFNM(modulate(self.norm2(x), shift_mlp, scale_mlp),x_size,mask)
        else:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.FFNM(self.norm2(x),x_size,mask)
        return x
class AttnBlockFM2(nn.Module):
    """
    An attention block followed by MLP
    Does not modulate using category or time embedded
    """
    def __init__(self, t_dim, hidden_size, num_heads, skip=True, mlp_ratio=4.0, drop=0.1, catagory=True):
        super().__init__()
        self.skip = skip
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionM(hidden_size, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop,proj_drop=drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.FFNM = FFNM_v2(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.cat = catagory
        if catagory:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_dim, 6 * hidden_size, bias=True)
            )

    def forward(self, x, x_size, c=None, mask=None):
        if self.cat and (c is not None):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask)
            x = x + gate_mlp.unsqueeze(1) * self.FFNM(modulate(self.norm2(x), shift_mlp, scale_mlp),x_size,mask)
        else:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.FFNM(self.norm2(x),x_size,mask)
        return x

class AttnLayerFM(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        #p_groups = patchSize[0]*patchSize[1]
        self.pSize = patchSize[0] * patchSize[1]
        self.conv_gin = nn.Conv2d(channels,hidden_size,1) # essentially a bottleneck through the attention
        self.conv_gout = nn.Conv2d(hidden_size,channels,1)
        if patchSize[0] == patchSize[1] == 1:
            #self.conv_gin = nn.Conv2d(channels,hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            #self.conv_gout = nn.Conv2d(hidden_size,channels,1,groups=p_groups)
            self.patch_en = False
        else:
            #self.conv_pin = nn.Conv2d(channels*patchSize[0]*patchSize[1],hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            #self.conv_pout = nn.Conv2d(hidden_size,channels*patchSize[0]*patchSize[1],1,groups=p_groups)
            self.patch_en = True
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.ModuleList([
            AttnBlockFM(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                      mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten)
            ])
        self.patchSize = patchSize

    def forward(self, x,t=None,m=None):
        h = self.act_fn(self.normal_1(x))
        if self.patch_en:
            h = unshuffle2B(h,self.patchSize)
            #h = self.conv_pin(h)
            h = self.conv_gin(h)
            if m is not None:
                m = unshuffle2B(m,self.patchSize)
        else:
            h = self.conv_gin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        if m is not None:
            m = TV2token(m)
        # t is based on the batch, so we need to rearrange it to match the fact that we've increased the batch size by the patch size
        if t is not None: # attention requires a category input, so skip attention!
            pt = t.view(t.shape[0],1,t.shape[1]) # add a dimension
            pt = pt.expand(-1,self.pSize,-1) # expand by the patch size
            pt = pt.reshape(-1,pt.shape[-1]) # reshape to reduce the dimension
        else:
            pt = t
        for block in self.attn:
                h = block(h,size_hw,pt,m) 
        h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_gout(h)
            h = shuffle2B(h,self.patchSize)
        else:
            h = self.conv_gout(h)
        return h    
class AttnLayerFM2(nn.Module):
    """
    a number of attention/MLP block surrounded by patch/unpatch
    """
    def __init__(self, num_atten, patchSize, channels, hidden_size, t_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        #p_groups = patchSize[0]*patchSize[1]
        self.pSize = patchSize[0] * patchSize[1]
        self.conv_gin = nn.Conv2d(channels,hidden_size,1) # essentially a bottleneck through the attention
        self.conv_gout = nn.Conv2d(hidden_size,channels,1)
        if patchSize[0] == patchSize[1] == 1:
            #self.conv_gin = nn.Conv2d(channels,hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            #self.conv_gout = nn.Conv2d(hidden_size,channels,1,groups=p_groups)
            self.patch_en = False
        else:
            #self.conv_pin = nn.Conv2d(channels*patchSize[0]*patchSize[1],hidden_size,1,groups=p_groups) # essentially a bottleneck through the attention
            #self.conv_pout = nn.Conv2d(hidden_size,channels*patchSize[0]*patchSize[1],1,groups=p_groups)
            self.patch_en = True
        self.act_fn = nn.SiLU()
        self.normal_1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.ModuleList([
            AttnBlockFM2(t_dim=t_dim,hidden_size=hidden_size,num_heads=num_heads,
                      mlp_ratio=mlp_ratio,drop=drop) for _ in range(num_atten)
            ])
        self.patchSize = patchSize

    def forward(self, x,t=None,m=None):
        h = self.act_fn(self.normal_1(x))
        if self.patch_en:
            h = unshuffle2B(h,self.patchSize)
            #h = self.conv_pin(h)
            h = self.conv_gin(h)
            if m is not None:
                m = unshuffle2B(m,self.patchSize)
        else:
            h = self.conv_gin(h)
        size_hw = h.shape[-2:]
        h = TV2token(h)
        if m is not None:
            m = TV2token(m)
        # t is based on the batch, so we need to rearrange it to match the fact that we've increased the batch size by the patch size
        if t is not None: # attention requires a category input, so skip attention!
            pt = t.view(t.shape[0],1,t.shape[1]) # add a dimension
            pt = pt.expand(-1,self.pSize,-1) # expand by the patch size
            pt = pt.reshape(-1,pt.shape[-1]) # reshape to reduce the dimension
        else:
            pt = t
        for block in self.attn:
                h = block(h,size_hw,pt,m) 
        h = Token2TV(h,size_hw)
        if self.patch_en:
            h = self.conv_gout(h)
            h = shuffle2B(h,self.patchSize)
        else:
            h = self.conv_gout(h)
        return h    

class ResnetBlock(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")
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
class ResnetBlockF(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = AttnLayerF(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
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
class ResnetBlockF2(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = AttnLayerF2(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
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
class ResnetBlock2D(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = AttnLayer2D(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
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

class ResnetBlockM(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding="same")
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = AttnLayerM(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
                                       num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout_rate)

    def forward(self, x, t=None, m=None):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        if m == None:
            h = self.conv_1(h)
        else:
            h = self.conv_1(h*m)
            mt = F.avg_pool2d(m,3,stride=1,padding=1)+1e-6
            h = h/mt

        # group 2
        # add in timestep embedding
        if t is not None:
            t_emb = self.dense_1(self.act_fn(t))[:, :, None, None]
            scale, shift = torch.chunk(t_emb, 2, dim=1)
            h = self.normlize_2(h) * (1 + scale) + shift
            
        # group 3
        h = self.act_fn(h)
        h = self.dropout(h)
        h = self.conv_2(h)
        if m == None:
            h = self.conv_2(h)
        else:
            h = self.conv_2(h*m)
            mt = F.avg_pool2d(m,3,stride=1,padding=1)+1e-6
            h = h/mt


        # Residual and attention
        h = h + x
        if self.apply_attention:
            if m == None:
                h = self.attention(h,t)
            else:
                h = self.attention(h,t,m)

        return h
class ResnetBlockFM(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        # use partial convolution and attention with feed forward network instead of MLP
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = PConv2d(inCh=self.channels, outCh=self.channels, kernel_size=3, stride=1, padding=1,norm=False)

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = PConv2d(inCh=self.channels, outCh=self.channels, kernel_size=3, stride=1, padding=1,norm=False)
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = AttnLayerFM(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
                                       num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout_rate)

    def forward(self, x, t=None, m=None):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h,mt = self.conv_1(h,m)

        # group 2
        # add in timestep embedding
        if t is not None:
            t_emb = self.dense_1(self.act_fn(t))[:, :, None, None]
            scale, shift = torch.chunk(t_emb, 2, dim=1)
            h = self.normlize_2(h) * (1 + scale) + shift
            
        # group 3
        h = self.act_fn(h)
        h = self.dropout(h)
        h, mt = self.conv_2(h,m)

        # Residual and attention
        h = h + x
        if self.apply_attention:
            h = self.attention(h,t,m)

        return h
class ResnetBlockFM2(nn.Module):
    def __init__(self, channels, num_atten=1,attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        # use partial convolution and attention with feed forward network instead of MLP
        self.channels = channels
        
        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.conv_1 = PConv2d(inCh=self.channels, outCh=self.channels, kernel_size=3, stride=1, padding=1,norm=False)

        # Group 2 time embedding
        #self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=2*self.channels)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = PConv2d(inCh=self.channels, outCh=self.channels, kernel_size=3, stride=1, padding=1,norm=False)
        self.apply_attention = apply_attention
        if apply_attention:
            self.attention = AttnLayerFM2(num_atten=num_atten,patchSize=patchSize,channels=channels,hidden_size=attn_channels, t_dim=time_emb_dims,
                                       num_heads=num_heads, mlp_ratio=mlp_ratio, drop=dropout_rate)

    def forward(self, x, t=None, m=None):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h,mt = self.conv_1(h,m)

        # group 2
        # add in timestep embedding
        if t is not None:
            t_emb = self.dense_1(self.act_fn(t))[:, :, None, None]
            scale, shift = torch.chunk(t_emb, 2, dim=1)
            h = self.normlize_2(h) * (1 + scale) + shift
            
        # group 3
        h = self.act_fn(h)
        h = self.dropout(h)
        h, mt = self.conv_2(h,m)

        # Residual and attention
        h = h + x
        if self.apply_attention:
            h = self.attention(h,t,m)

        return h

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
class ResnetDwnF(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.block1 = ResnetBlockF(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlockF(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t):
        h = self.act_fn(self.normlize_1(x))
        h = self.conv1(x)
        h = self.block1(h,t)
        h = self.block2(h,t)
        return h
class ResnetDwnF2(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.block1 = ResnetBlockF2(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlockF2(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t):
        h = self.act_fn(self.normlize_1(x))
        h = self.conv1(x)
        h = self.block1(h,t)
        h = self.block2(h,t)
        return h
class ResnetDwn2D(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.block1 = ResnetBlock2D(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlock2D(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t):
        h = self.act_fn(self.normlize_1(x))
        h = self.conv1(x)
        h = self.block1(h,t)
        h = self.block2(h,t)
        return h

class ResnetDwnM(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.block1 = ResnetBlockM(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlockM(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t, m=None):
        h = self.act_fn(self.normlize_1(x))
        if m == None:
            h = self.conv1(h)
            h = self.block1(h,t)
            h = self.block2(h,t)
            return h, m
        else:
            h = self.conv1(h*m)
            md = F.avg_pool2d(m,3,stride=2,padding=1)
            h = h/(md+1e-6)
            mm = F.max_pool2d(m,3,stride=2,padding=1)
            m = torch.minimum(3*md,mm)
            #m = torch.clamp(3*m,0,1)
            h = self.block1(h,t,m)
            h = self.block2(h,t,m)
            return h,m
class ResnetDwnFM(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = PConv2d(inCh=in_channels, outCh=out_channels, kernel_size=3, stride=2, padding=1,norm=False)
        self.block1 = ResnetBlockFM(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlockFM(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t, m=None):
        h = self.act_fn(self.normlize_1(x))
        h,m = self.conv1(h,m)
        h = self.block1(h,t,m)
        h = self.block2(h,t,m)
        return h, m
class ResnetDwnFM2(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act_fn = nn.SiLU()
        self.conv1 = PConv2d(inCh=in_channels, outCh=out_channels, kernel_size=3, stride=2, padding=1,norm=False)
        self.block1 = ResnetBlockFM2(out_channels,num_atten=num_atten,attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlockFM2(out_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=False)

    def forward(self, x, t, m=None):
        h = self.act_fn(self.normlize_1(x))
        h,m = self.conv1(h,m)
        h = self.block1(h,t,m)
        h = self.block2(h,t,m)
        return h, m

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
class ResnetUpF(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=2*in_channels)
        self.block1 = ResnetBlockF(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlockF(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
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
class ResnetUpF2(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=2*in_channels)
        self.block1 = ResnetBlockF2(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlockF2(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
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
class ResnetUp2D(nn.Module):
    def __init__(self, in_channels, out_channels,num_atten=2, attn_channels=1024,patchSize=2,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=2*in_channels)
        self.block1 = ResnetBlock2D(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
                            mlp_ratio=mlp_ratio,dropout_rate=dropout_rate, time_emb_dims=time_emb_dims, apply_attention=apply_attention)
        self.block2 = ResnetBlock2D(in_channels,num_atten=num_atten, attn_channels=attn_channels,patchSize=patchSize,num_heads=num_heads,
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

class CatGen(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512):
        super().__init__()

        self.econv6c = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5c = ResnetDwnM(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[5])
        self.enc4c = ResnetDwnM(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[4])
        self.enc3c = ResnetDwnM(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[3])
        self.enc2c = ResnetDwnM(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[2])
        self.enc1c = ResnetDwnM(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[1])
        self.enc0c = ResnetDwnM(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[0])
        self.cat_norm1 = nn.GroupNorm(num_groups=channel_mult[0],num_channels=base_channels*channel_mult[0])

    def forward(self, x, t, m, x0i, m0i, ec):
        t0 = self.t_embedder(t*0)
        if m0i == None: 
            es = self.econv6c(x0i)
            ms = m0i
        else:
            es = self.econv6c(x0i*m0i)
            md = F.avg_pool2d(m0i,3,stride=1,padding=1)
            es = es/(md+1e-6)
            mm6 = F.max_pool2d(m0i,3,stride=1,padding=1)
            ms = torch.minimum(3*md,mm6)
            #m6 = torch.clamp(3*m6,0,1)
        es,ms = self.enc5c(es,t0,ms)
        es,ms = self.enc4c(es,t0,ms)
        es,ms = self.enc3c(es,t0,ms)
        es,ms = self.enc2c(es,t0,ms)
        es,ms = self.enc1c(es,t0,ms)
        es,ms = self.enc0c(es,t0,ms)
        #
        es = self.cat_norm1(self.act_fn(es),)
        ec = self.aaPool(es*ms) # reduce to 2x2
        em = self.aaPool(ms)    # using weighted average
        ec = ec/(em+1e-6) 
        ec = self.fconv(ec*em)  # then all the way to 1x1
        em = F.avg_pool2d(em,4) # with weighted average
        ec = ec/(em+1e-6)
        ec = self.cat_norm2(self.act_fn(ec))
        ec = ec.squeeze(2).squeeze(2)
        ec = self.fLin(ec)
        return ec


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
    
class Model_C(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512,):
        super().__init__()

        self.econv6c = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5c = ResnetDwnM(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[5])
        self.enc4c = ResnetDwnM(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[4])
        self.enc3c = ResnetDwnM(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[3])
        self.enc2c = ResnetDwnM(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[2])
        self.enc1c = ResnetDwnM(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[1])
        self.enc0c = ResnetDwnM(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[0])
        self.cat_norm1 = nn.GroupNorm(num_groups=channel_mult[0],num_channels=base_channels*channel_mult[0])
        self.cat_norm2 = nn.GroupNorm(num_groups=32,num_channels=time_emb_dims*2)
        self.fLin = nn.Linear(time_emb_dims*2,time_emb_dims)
        self.aaPool = nn.AdaptiveAvgPool2d(4)
        self.fconv = nn.Conv2d(base_channels*channel_mult[0],time_emb_dims*2,4)

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
    
    def forward(self, x, t, x0i, m0i):
        t0 = self.t_embedder(t*0)
        if m0i == None: 
            es = self.econv6c(x0i)
            ms = m0i
        else:
            es = self.econv6c(x0i*m0i)
            md = F.avg_pool2d(m0i,3,stride=1,padding=1)
            es = es/(md+1e-6)
            mm6 = F.max_pool2d(m0i,3,stride=1,padding=1)
            ms = torch.minimum(3*md,mm6)
            #m6 = torch.clamp(3*m6,0,1)
        es,ms = self.enc5c(es,t0,ms)
        es,ms = self.enc4c(es,t0,ms)
        es,ms = self.enc3c(es,t0,ms)
        es,ms = self.enc2c(es,t0,ms)
        es,ms = self.enc1c(es,t0,ms)
        es,ms = self.enc0c(es,t0,ms)
        #
        es = self.cat_norm1(self.act_fn(es),)
        ec = self.aaPool(es*ms) # reduce to 2x2
        em = self.aaPool(ms)    # using weighted average
        ec = ec/(em+1e-6) 
        ec = self.fconv(ec*em)  # then all the way to 1x1
        em = F.avg_pool2d(em,4) # with weighted average
        ec = ec/(em+1e-6)
        ec = self.cat_norm2(self.act_fn(ec))
        ec = ec.squeeze(2).squeeze(2)
        ec = self.fLin(ec)

        t = self.t_embedder(t) 
        t = t + ec
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
class Model_TC(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512,):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.econv6c = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5c = ResnetDwnM(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[5])
        self.enc4c = ResnetDwnM(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[4])
        self.enc3c = ResnetDwnM(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[3])
        self.enc2c = ResnetDwnM(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[2])
        self.enc1c = ResnetDwnM(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[1])
        self.enc0c = ResnetDwnM(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[0])
        self.cat_norm1 = nn.GroupNorm(num_groups=channel_mult[0],num_channels=base_channels*channel_mult[0])
        self.cat_norm2a = nn.GroupNorm(num_groups=4,num_channels=time_emb_dims*2)
        self.fLin = nn.Linear(time_emb_dims*2,time_emb_dims)
        #self.aaPool = nn.AdaptiveAvgPool2d(4)
        self.amaxPool = nn.AdaptiveMaxPool2d(4)
        self.fconv = nn.Conv2d(base_channels*channel_mult[0],time_emb_dims*2,4)

        self.tcat_norm = nn.GroupNorm(num_groups=32,num_channels=time_emb_dims*2)
        self.tcLin = nn.Linear(time_emb_dims*2,time_emb_dims)
        
        self.t_Embedder = TimestepEmbedder(time_emb_dims)
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
    
    def forward(self, x, t, x0i, m0i, ec=None):
        t0 = None
        if ec is None:
            if m0i == None: 
                es = self.econv6c(x0i)
                ms = m0i
            else:
                es = self.econv6c(x0i*m0i)
                md = F.avg_pool2d(m0i,3,stride=1,padding=1)
                es = es/(md+1e-6)
                mm6 = F.max_pool2d(m0i,3,stride=1,padding=1)
                ms = torch.minimum(3*md,mm6)
                #m6 = torch.clamp(3*m6,0,1)
            es,ms = self.enc5c(es,t0,ms)
            es,ms = self.enc4c(es,t0,ms)
            es,ms = self.enc3c(es,t0,ms)
            es,ms = self.enc2c(es,t0,ms)
            es,ms = self.enc1c(es,t0,ms)
            es,ms = self.enc0c(es,t0,ms)
            #
            es = self.cat_norm1(self.act_fn(es))
            #Hes = 4*math.ceil(es.shape[-2]/4)
            #Wes = 4*math.ceil(es.shape[-1]/4)
            #es = F.interpolate(es,size=[Hes,Wes],mode='bilinear',antialias=True)
            #ms = F.interpolate(ms,size=[Hes,Wes],mode='bilinear',antialias=True)
            #ec = self.aaPool(es*ms) # reduce to 2x2
            #em = self.aaPool(ms)    # using weighted average
            ec = self.amaxPool(es*ms) # reduce to 2x2
            em = self.amaxPool(ms)    # using weighted average
            ec = ec/(em+1e-6) 
            ec = self.fconv(ec*em)  # then all the way to 1x1
            em = F.avg_pool2d(em,4) # with weighted average
            ec = ec/(em+1e-6)
            ec = self.cat_norm2a(self.act_fn(ec))
            ec = ec.squeeze(2).squeeze(2)
            ec = self.fLin(ec)

        t = self.t_Embedder(t) 
        #t = t + ec
        t = torch.concat([t,ec],dim=1)
        #t = self.act_fn(t)
        t = self.dropout(t)
        t = self.tcLin(t)
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
        return out, ec
class Model_F(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512,train_cat=True):
        super().__init__()
        
        self.train_cat = train_cat
        self.dropout = nn.Dropout(p=dropout_rate)
        self.econv6c = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        #self.econv6c = PConv2d(3,base_channels,3,stride=1,padding=1,norm=False)
        self.enc5c = ResnetDwnFM(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[5])
        self.enc4c = ResnetDwnFM(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[4])
        self.enc3c = ResnetDwnFM(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[3])
        self.enc2c = ResnetDwnFM(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[2])
        self.enc1c = ResnetDwnFM(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[1])
        self.enc0c = ResnetDwnFM(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[0])
        self.cat_norm1 = nn.GroupNorm(num_groups=channel_mult[0],num_channels=base_channels*channel_mult[0])
        self.cat_norm2a = nn.GroupNorm(num_groups=4,num_channels=time_emb_dims*2)
        self.fLin = nn.Linear(time_emb_dims*2,time_emb_dims)
        #self.aaPool = nn.AdaptiveAvgPool2d(4)
        self.amaxPool = nn.AdaptiveMaxPool2d(16)
        self.fconv = PConv2dE(base_channels*channel_mult[0],time_emb_dims*2,4,norm=False)

        self.tcat_norm = nn.GroupNorm(num_groups=32,num_channels=time_emb_dims*2)
        self.tcLin = nn.Linear(time_emb_dims*2,time_emb_dims)
        
        self.t_Embedder = TimestepEmbedder(time_emb_dims)
        self.econv6 = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5 = ResnetDwnF(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5])
        self.enc4 = ResnetDwnF(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4])
        self.enc3 = ResnetDwnF(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3])
        self.enc2 = ResnetDwnF(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2])
        self.enc1 = ResnetDwnF(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1])
        self.enc0 = ResnetDwnF(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0])
        self.mid = AttnLayerF(num_atten=con_attn_num,patchSize=patchSizes[0],channels=base_channels*channel_mult[0],hidden_size=attn_channels,
                                t_dim=time_emb_dims,num_heads=num_heads,mlp_ratio=mlp_ratio)
        self.dec0 = ResnetUpF(base_channels*channel_mult[0],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0])
        self.dec1 = ResnetUpF(base_channels*channel_mult[1],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1])
        self.dec2 = ResnetUpF(base_channels*channel_mult[2],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2])
        self.dec3 = ResnetUpF(base_channels*channel_mult[3],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3])
        self.dec4 = ResnetUpF(base_channels*channel_mult[4],base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4])
        self.dec5 = ResnetUpF(base_channels*channel_mult[5],base_channels,dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5])
        self.dconv6 = nn.Conv2d(base_channels,3,3,stride=1,padding=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.act_fn = nn.SiLU()
    
    def forward(self, x, t, x0i, m0i, ec=None):
        t0 = None
        if self.train_cat:   
            if ec is None:
                #es, ms = self.econv6c(x0i,m0i)
                #ms = m0i
                if m0i == None: 
                    es = self.econv6c(x0i)
                    ms = m0i
                else:
                    es = self.econv6c(x0i*m0i)
                    md = F.avg_pool2d(m0i,3,stride=1,padding=1)
                    es = es/(md+1e-6)
                    mm6 = F.max_pool2d(m0i,3,stride=1,padding=1)
                    ms = torch.minimum(3*md,mm6)
                #m6 = torch.clamp(3*m6,0,1)

                es,ms = self.enc5c(es,t0,ms)
                es,ms = self.enc4c(es,t0,ms)
                es,ms = self.enc3c(es,t0,ms)
                es,ms = self.enc2c(es,t0,ms) #16x16
                # code to get anysize image > 256x256 to a fixed size
                es = self.cat_norm1(self.act_fn(es))
                ec = self.amaxPool(es*ms) # reduce 
                ms = self.amaxPool(ms)    # using weighted average
                es = ec/(ms+1e-6)
                #
                es,ms = self.enc1c(es,t0,ms) #8x8
                es,ms = self.enc0c(es,t0,ms) #4x4
                ec, mt = self.fconv(es,ms)  # then all the way to 1x1
                ec = self.cat_norm2a(self.act_fn(ec))
                ec = ec.squeeze(2).squeeze(2)
                ec = self.fLin(ec)

        t = self.t_Embedder(t) 
        #t = t + ec
        if self.train_cat:
            t = torch.concat([t,ec],dim=1)
            #t = self.act_fn(t)
            t = self.dropout(t)
            t = self.tcLin(t)
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
        return out, ec
class Model_F2(nn.Module):
    def __init__(self, base_channels, channel_mult, attn_layers, num_atten, patchSizes, con_attn_num=2, attn_channels=1024,num_heads=4,
                 mlp_ratio=4.0,dropout_rate=0.1, time_emb_dims=512,train_cat=True):
        super().__init__()
        
        self.train_cat = train_cat
        self.dropout = nn.Dropout(p=dropout_rate)
        self.econv6c = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        #self.econv6c = PConv2d(3,base_channels,3,stride=1,padding=1,norm=False)
        self.enc5c = ResnetDwnFM2(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[5])
        self.enc4c = ResnetDwnFM2(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[4])
        self.enc3c = ResnetDwnFM2(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[3])
        self.enc2c = ResnetDwnFM2(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[2])
        self.enc1c = ResnetDwnFM2(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[1])
        self.enc0c = ResnetDwnFM2(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=False,num_atten=num_atten[0])
        self.cat_norm2c = nn.GroupNorm(num_groups=channel_mult[2],num_channels=base_channels*channel_mult[2])
        self.cat_norm2a = nn.GroupNorm(num_groups=4,num_channels=time_emb_dims*2)
        self.fLin = nn.Linear(time_emb_dims*2,time_emb_dims)
        #self.aaPool = nn.AdaptiveAvgPool2d(4)
        self.amaxPool = nn.AdaptiveMaxPool2d(16)
        self.fconv = PConv2dE(base_channels*channel_mult[0],time_emb_dims*2,4,norm=False)

        self.tcat_norm = nn.GroupNorm(num_groups=32,num_channels=time_emb_dims*2)
        self.tcLin = nn.Linear(time_emb_dims*2,time_emb_dims)
        
        self.t_Embedder = TimestepEmbedder(time_emb_dims)
        self.econv6 = nn.Conv2d(3,base_channels,3,stride=1,padding=1)
        self.enc5 = ResnetDwnF2(base_channels,base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5])
        self.enc4 = ResnetDwnF2(base_channels*channel_mult[5],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4])
        self.enc3 = ResnetDwnF2(base_channels*channel_mult[4],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3])
        self.enc2 = ResnetDwnF2(base_channels*channel_mult[3],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2])
        self.enc1 = ResnetDwnF2(base_channels*channel_mult[2],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1])
        self.enc0 = ResnetDwnF2(base_channels*channel_mult[1],base_channels*channel_mult[0],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0])
        self.mid = AttnLayerF2(num_atten=con_attn_num,patchSize=patchSizes[0],channels=base_channels*channel_mult[0],hidden_size=attn_channels,
                                t_dim=time_emb_dims,num_heads=num_heads,mlp_ratio=mlp_ratio)
        self.dec0 = ResnetUpF2(base_channels*channel_mult[0],base_channels*channel_mult[1],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[0],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[0],num_atten=num_atten[0])
        self.dec1 = ResnetUpF2(base_channels*channel_mult[1],base_channels*channel_mult[2],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[1],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[1],num_atten=num_atten[1])
        self.dec2 = ResnetUpF2(base_channels*channel_mult[2],base_channels*channel_mult[3],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[2],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[2],num_atten=num_atten[2])
        self.dec3 = ResnetUpF2(base_channels*channel_mult[3],base_channels*channel_mult[4],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[3],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[3],num_atten=num_atten[3])
        self.dec4 = ResnetUpF2(base_channels*channel_mult[4],base_channels*channel_mult[5],dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[4],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[4],num_atten=num_atten[4])
        self.dec5 = ResnetUpF2(base_channels*channel_mult[5],base_channels,dropout_rate=dropout_rate,time_emb_dims=time_emb_dims,
                              num_heads=num_heads,attn_channels=attn_channels,patchSize=patchSizes[5],
                              mlp_ratio=mlp_ratio,apply_attention=attn_layers[5],num_atten=num_atten[5])
        self.dconv6 = nn.Conv2d(base_channels,3,3,stride=1,padding=1)
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        self.act_fn = nn.SiLU()
    
    def forward(self, x, t, x0i, m0i, ec=None):
        t0 = None
        if self.train_cat:   
            if ec is None:
                #es, ms = self.econv6c(x0i,m0i)
                #ms = m0i
                if m0i == None: 
                    es = self.econv6c(x0i)
                    ms = m0i
                else:
                    es = self.econv6c(x0i*m0i)
                    md = F.avg_pool2d(m0i,3,stride=1,padding=1)
                    es = es/(md+1e-6)
                    mm6 = F.max_pool2d(m0i,3,stride=1,padding=1)
                    ms = torch.minimum(3*md,mm6)
                #m6 = torch.clamp(3*m6,0,1)

                es,ms = self.enc5c(es,t0,ms)
                es,ms = self.enc4c(es,t0,ms)
                es,ms = self.enc3c(es,t0,ms)
                es,ms = self.enc2c(es,t0,ms) #16x16
                # code to get anysize image > 256x256 to a fixed size
                es = self.cat_norm2c(self.act_fn(es))
                ec = self.amaxPool(es*ms) # reduce 
                ms = self.amaxPool(ms)    # using weighted average
                es = ec/(ms+1e-6)
                #
                es,ms = self.enc1c(es,t0,ms) #8x8
                es,ms = self.enc0c(es,t0,ms) #4x4
                ec, mt = self.fconv(es,ms)  # then all the way to 1x1
                ec = self.cat_norm2a(self.act_fn(ec))
                ec = ec.squeeze(2).squeeze(2)
                ec = self.fLin(ec)

        t = self.t_Embedder(t) 
        #t = t + ec
        if self.train_cat:
            t = torch.concat([t,ec],dim=1)
            #t = self.act_fn(t)
            t = self.dropout(t)
            t = self.tcLin(t)
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
        return out, ec
   
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
