import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from PIL import Image
import os
#import base64
#from IPython.display import display, HTML, clear_output
from Conv2dPartial import PConv2d

# version 2 changes all view() to reshape()

def NP2aWin(x,wSize): # numpy formated image to attention windows
    # take a B of numpy order tensors partition into attention windows
    B,H,W,C = x.shape
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,H//wSize[0],wSize[0],W//wSize[1],wSize[1],C)
        x = x.permute(0,1,3,2,4,5).reshape(-1,wSize[0],wSize[1],C).to(device)
    else:
        x = x.reshape(B,H//wSize[0],wSize[0],W//wSize[1],wSize[1],C)
        x = x.permute(0,1,3,2,4,5).reshape(-1,wSize[0],wSize[1],C)
    return x
def TV2aWin(x,wSize): # torch vision formated image to attention windows
    # take a B of numpy order tensors partition into attention windows
    B,C,H,W = x.shape
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,C,H//wSize[0],wSize[0],W//wSize[1],wSize[1])
        x = x.permute(0,2,4,1,3,5).reshape(-1,C,wSize[0],wSize[1]).to(device)
    else:
        x = x.reshape(B,C,H//wSize[0],wSize[0],W//wSize[1],wSize[1])
        x = x.permute(0,2,4,1,3,5).reshape(-1,C,wSize[0],wSize[1])
    return x
def TV2wToken(x,wSize): # torch vision formated image to attention windows
    # take a B of numpy order tensors partition into attention windows
    B,C,H,W = x.shape
    Hw = H//wSize[0]
    Ww = W//wSize[1]
    N = wSize[0]*wSize[1]
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,C,Hw,wSize[0],Ww,wSize[1])
        x = x.permute(0,2,4,3,5,1).reshape(-1,N,C).to(device)
    else:
        x = x.reshape(B,C,H//wSize[0],wSize[0],W//wSize[1],wSize[1])
        x = x.permute(0,2,4,3,5,1).reshape(-1,N,C)
    return x
def Wtoken2TV(x,x_size,wSize):
    Bw,N,C = x.shape # number of windows*batch, tokens, channels
    H,W = x_size
    Hw = H//wSize[0]
    Ww = W//wSize[1]
    B = Bw//Hw//Ww
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,Hw,Ww,wSize[0],wSize[1],C)
        x = x.permute(0,5,1,3,2,4).reshape(B,C,H,W).to(device) # tensor reshape order of the B
    else:
        x = x.reshape(B,Hw,Ww,wSize[0],wSize[1],C)
        x = x.permute(0,5,1,3,2,4).reshape(B,C,H,W) # tensor reshape order of the B
    return x
    
def TV2pToken(x,pSize): # torch vision formated image to attention windows
    # take a B of numpy order tensors partition into attention windows
    B,C,H,W = x.shape
    device = x.device
    Hp = H//pSize[0]
    Wp = W//pSize[1]
    N = Hp*Wp
    if(device =='mps'):
        x = x.to('cpu').reshape(B,C,Hp,pSize[0],Wp,pSize[1])
        x = x.permute(0,3,5,2,4,1).reshape(-1,N,C).to(device)
    else:
        x = x.reshape(B,C,Hp,pSize[0],Wp,pSize[1])
        x = x.permute(0,3,5,2,4,1).reshape(-1,N,C)
    return x
def Ptoken2TV(x,x_size,pSize):
    Bp,N,C = x.shape # number of windows*batch, tokens, channels
    H,W = x_size
    Hp = H//pSize[0]
    Wp = W//pSize[1]
    B = Bp//pSize[0]//pSize[1]
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,pSize[0],pSize[1],Hp,Wp,C)
        x = x.permute(0,5,3,1,4,2).reshape(B,C,H,W).to(device) # tensor reshape order of the B
    else:
        x = x.reshape(B,pSize[0],pSize[1],Hp,Wp,C)
        x = x.permute(0,5,3,1,4,2).reshape(B,C,H,W) # tensor reshape order of the B
    return x

def Ptoken2wToken(x,x_size,pSize):
    Bp,N,C = x.shape # number of windows*batch, tokens, channels
    H,W = x_size
    Hp = H//pSize[0]
    Wp = W//pSize[1]
    B = Bp//pSize[0]//pSize[1]
    Nw = pSize[0]*pSize[1]
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,pSize[0],pSize[1],Hp,Wp,C)
        x = x.permute(0,3,4,1,2,5).reshape(-1,Nw,C).to(device) # tensor reshape order of the B
    else:
        x = x.reshape(B,pSize[0],pSize[1],Hp,Wp,C)
        x = x.permute(0,5,3,1,4,2).reshape(-1,Nw,C) # tensor reshape order of the B
    return x

def AWin2np(x,wSize,H,W): # attention windows to numpy formated image
    # take a set of attention windows and create a B of numpy style images
    B = int(x.shape[0]/(H*W/wSize[0]/wSize[1]))
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,H//wSize[0],W//wSize[1],wSize[0],wSize[1],-1)
        x = x.permute(0,1,3,2,4,5).reshape(B,H,W,-1).to(device)
    else:
        x = x.reshape(B,H//wSize[0],W//wSize[1],wSize[0],wSize[1],-1)
        x = x.permute(0,1,3,2,4,5).reshape(B,H,W,-1)
    return x
def AWin2TV(x,wSize,H,W): # attention windows to torch vision formated image
    # take a set of attention windows and create a B of numpy style images
    B = int(x.shape[0]/(H*W/wSize[0]/wSize[1]))
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,H//wSize[0],W//wSize[1],-1,wSize[0],wSize[1])
        x = x.permute(0,3,2,4,3,5).reshape(B,-1,H,W).to(device)
    else:
        x = x.reshape(B,H//wSize[0],W//wSize[1],wSize[0],wSize[1],-1)
        x = x.permute(0,3,2,4,3,5).reshape(B,-1,H,W)
    return x

def Token2TV(x, x_size): # B of tokenized images to torch vision format
    B,N,C = x.shape # tokenized B
    H,W = x_size
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').permute(0,2,1).reshape(B,C,H,W).to(device) # tensor reshape order of the B
    else:
        x = x.permute(0,2,1).reshape(B,C,H,W)# tensor reshape order of the B
    return x

def TV2token(x): # B of torch vision images to tokenized format
    B,C,H,W = x.shape # tensor reshape order of the B
    device = x.device
    if(device =='mps'):
        x = x.to('cpu').reshape(B,C,-1).transpose(1,2).to(device) # tokenized B
    else:   
        x = x.reshape(B,C,-1).transpose(1,2) # tokenized B
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
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.reshape(-1, C, H*r[0], W*r[1])
    return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class FFN(nn.Module):
    """ FFN similar to others. Essentially a MLP with a 3x3 depthwise convolution in the middle
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        #self.norm = nn.LayerNorm(hidden_features)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.conv = nn.Conv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x, size):
        x = Token2TV(x,size)
        x = self.conv(x)
        x = TV2token(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class FFN_v2(nn.Module):
    """ FFN similar to others. Essentially a MLP with a 3x3 depthwise convolution in the middle
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = nn.Linear

        self.fc1a = linear_layer(in_features, hidden_features, bias=bias)
        self.fc1b = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        #self.norm = nn.LayerNorm(hidden_features)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.conva = nn.Conv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.convb = nn.Conv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x, size):
        x = Token2TV(x,size)
        y = self.convb(x)
        x = self.conva(x)
        x = TV2token(x)
        y = TV2token(y)
        y = self.fc1b(y)
        x = self.fc1a(x)
        x = self.act(x)
        x = x*y
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class FFN2D(nn.Module):
    """ FFN similar to others. Essentially a MLP with a 3x3 depthwise convolution in the middle
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = nn.Linear

        self.fc1a = linear_layer(in_features, hidden_features, bias=bias)
        self.fc1b = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        #self.norm = nn.LayerNorm(hidden_features)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.conva = nn.Conv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.convb = nn.Conv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        #x = Token2TV(x,size)
        x_size = x.shape[-2:]
        y = self.convb(x)
        x = self.conva(x)
        x = TV2token(x)
        y = TV2token(y)
        y = self.fc1b(y)
        x = self.fc1a(x)
        x = self.act(x)
        x = x*y
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = Token2TV(x,x_size)
        return x
    
class FFNM(nn.Module):
    """ FFN similar to others. Essentially a MLP with a 3x3 depthwise convolution in the middle
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.conv = PConv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False, norm=False)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x, size,m=None):
        if m is not None:
            m = Token2TV(m,size)
        x = Token2TV(x,size)
        x, _m = self.conv(x,m)
        x = TV2token(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class FFNM_v2(nn.Module):
    """ FFN similar to others. Essentially a MLP with a 3x3 depthwise convolution in the middle
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = nn.Linear

        self.fc1a = linear_layer(in_features, hidden_features, bias=bias)
        self.fc1b = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        #self.norm = nn.LayerNorm(hidden_features)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.conva = PConv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.convb = PConv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x, size,m=None):
        if m is not None:
            m = Token2TV(m,size)
        x = Token2TV(x,size)
        y, _m = self.convb(x,m)
        x, _m = self.conva(x,m)
        x = TV2token(x)
        y = TV2token(y)
        x = self.fc1a(x)
        y = self.fc1b(y)
        x = self.act(x)
        x = x*y
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class FFNM2D(nn.Module):
    """ FFN similar to others. Essentially a MLP with a 3x3 depthwise convolution in the middle
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = nn.Linear

        self.fc1a = linear_layer(in_features, hidden_features, bias=bias)
        self.fc1b = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        #self.norm = nn.LayerNorm(hidden_features)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.conva = PConv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.convb = PConv2d(in_features,in_features,3,padding=1,groups=in_features,bias=False)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x,m=None):
        #if m is not None:
        #    m = Token2TV(m,size)
        #x = Token2TV(x,size)
        x_size = x.shape[-2:]
        y, _m = self.convb(x,m)
        x, _m = self.conva(x,m)
        x = TV2token(x)
        y = TV2token(y)
        x = self.fc1a(x)
        y = self.fc1b(y)
        x = self.act(x)
        x = x*y
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = Token2TV(x,x_size)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: int = [32,32],
            patch_size: int = [2,2],
            in_chans: int = 3,
            embed_dim: int = 768,
            Normalize: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        if img_size is not None:
            self.img_size = img_size
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = nn.LayerNorm(embed_dim,bias=bias) if Normalize else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    
def get(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)

def setup_log_directory(config):
    '''Log and Model checkpoint directory Setup'''
    
    if os.path.isdir(config.root_log_dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(config.root_log_dir)]
        
        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = config.log_dir

    # Update the training config default directory 
    log_dir        = os.path.join(config.root_log_dir,        version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")
    
    return log_dir, checkpoint_dir

def frames2vid(images, save_path):

    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fourcc = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))

    # Appending the images to the video one by one
    for image in images:
        video.write(image)

    # Deallocating memories taken for window creation
    # cv2.destroyAllWindows()
    video.release()
    return 

#def display_gif(gif_path):
#    b64 = base64.b64encode(open(gif_path,'rb').read()).decode('ascii')
#    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))

class MaskAttn(nn.Module):
    def __init__(self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #self.linq = nn.Linear(dim, dim, bias=qkv_bias)
        #self.link = nn.Linear(dim, dim, bias=qkv_bias)
        #self.linv = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mWin=None): # x.shape = 
        B, N, C = x.shape # nw = 1, so just B,N,C
        device = x.device
        """
        if device == 'mps':
            normX = F.normalize(x,dim=-1).to('cpu')
            q = self.linq(normX).reshape(Bnw,N,self.nHeads,C//self.nHeads).permute(0,2,1,3).to(device)
            k = self.link(normX).reshape(Bnw,N,self.nHeads,C//self.nHeads).permute(0,2,3,1).to(device)
            v = self.linv(x).reshape(Bnw,N,self.nHeads,C//self.nHeads).permute(0,2,1,3).to(device)
        else:
            normX = F.normalize(x,dim=-1)
            q = self.linq(normX).reshape(Bnw,N,self.nHeads,C//self.nHeads).permute(0,2,1,3)
            k = self.link(normX).reshape(Bnw,N,self.nHeads,C//self.nHeads).permute(0,2,3,1)
            v = self.linv(x).reshape(Bnw,N,self.nHeads,C//self.nHeads).permute(0,2,1,3)
        """
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        if mWin is not None:
            attn_mWin = mWin.squeeze(-1).unsqueeze(1).unsqueeze(1)
            #attn = attn + attn_mWin.masked_fill(attn_mWin == 0, -100.).masked_fill(attn_mWin == 1, 0.)
            attn = attn + attn_mWin*100-100
        attn = self. softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MaskAttn2(nn.Module):
    def __init__(self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mWin=None): # x.shape = 
        B, N, C = x.shape # nw = 1, so just B,N,C
        device = x.device
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        if mWin is not None:
            m = torch.unsqueeze(mWin,1)
            q = q*m
            k = k*m
            v = v*m
            mattn = m @ m.transpose(-2,-1)
        attn = q @ k.transpose(-2, -1)
        
#        if mWin is not None:
#            attn_mWin = mWin.squeeze(-1).unsqueeze(1).unsqueeze(1)
#            #attn = attn + attn_mWin.masked_fill(attn_mWin == 0, -100.).masked_fill(attn_mWin == 1, 0.)
#            attn = attn + attn_mWin*100-100
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        if mWin is not None:
            mx = (mattn @ m).transpose(1,2).reshape(B,N,1)
            x = x/(mx+1e-6)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

