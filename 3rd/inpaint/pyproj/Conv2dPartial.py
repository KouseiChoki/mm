import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from torch import nn, cuda
from torch.autograd import Variable

#from torch.nn.utils.parametrizations import weight_norm as wnorm
from torch.nn.utils import weight_norm as wnorm

""" v1 is restricted to using an even kernel and S = 1 or 2
    Unlike the version that is described in a paper, this version allows the use of a weighted average
    for the denominator so that it can match the coefficients in the numerator"""
class PConv2dE(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride=1,padding=0,bias=False,norm=True,groups=1):
        super(PConv2dE, self).__init__()
        """the even kernel size version is used for downsampling. Wtih stride = 2, the effect is a LPF + 2x2 to 1 pixel 
            mapping in the down conversion. This will match an upconversion process where you use pixel shuffle which
            maps 1x1 to a 2x2 output"""
        # whether the mask is multi-channel or not
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.outCh = outCh
        self.inCh = inCh
        if norm:
            self.cf0 = wnorm(nn.Conv2d(inCh,outCh,kernel_size,stride=stride,padding=padding,bias=bias,groups=groups))
        else:
            self.cf0 = nn.Conv2d(inCh,outCh,kernel_size,stride=stride,padding=padding,bias=bias,groups=groups)
        #self.cf1 = nn.Conv2d(inCh,outCh,kernel_size,stride=stride,padding=padding,bias=bias,groups=groups)
        #self.cf1.weight = nn.Parameter(torch.ones_like(self.cf1.weight))
        self.wmp2D = nn.MaxPool2d(2,stride=stride,padding=0)
        
    def forward(self, input, mask=None):
        #if self.cf0.weight.type() != input.type():
        #    self.cf0.weight = self.cf0.weight.to(input)
        #if self.cf1.weight.type() != input.type():
        #    self.cf1.weight = self.cf1.weight.to(input)
        with torch.no_grad():
        #    self.cf1.weight = nn.Parameter(self.cf1.weight.abs())
            cf1_weight = self.cf0.weight.abs() # map the raw weights to be between 0-1
        if mask is not None:
            maxMask = self.wmp2D(mask)
            #update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), self.cf1.weight, bias=None, stride=self.stride, padding=self.padding,groups=self.groups)
            update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), cf1_weight, bias=None, stride=self.stride, padding=self.padding,groups=self.groups)
            denom = update_mask + 1e-6
            raw_numer = self.cf0(input*mask)
            output = raw_numer/denom
            return output, maxMask
        else:
            output = self.cf0(input)
            return output, mask


class PConv2d(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride=1,padding=0,bias=False,norm=True,groups=1):
        super(PConv2d, self).__init__()
        """The plain version is more like the version in the literature. It can use either an odd or even kernel. The
            window for the mask is based on the kernel size instead of being fixed at 2x2 """
        # whether the mask is multi-channel or not
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.outCh = outCh
        self.inCh = inCh
        if norm:
            self.cf0 = wnorm(nn.Conv2d(inCh,outCh,kernel_size,stride=stride,padding=padding,bias=bias,groups=groups))
        else:
            self.cf0 = nn.Conv2d(inCh,outCh,kernel_size,stride=stride,padding=padding,bias=bias,groups=groups)
        #self.cf1 = nn.Conv2d(inCh,outCh,kernel_size,stride=stride,padding=padding,bias=bias,groups=groups)
        #self.cf1.weight = nn.Parameter(torch.ones_like(self.cf1.weight))
        self.cmp2D = nn.MaxPool2d(kernel_size,stride=stride,padding=padding)

        
    def forward(self, input, mask=None):
        #if self.cf0.weight.type() != input.type():
        #    self.cf0.weight = self.cf0.weight.to(input)
        #if self.cf1.weight.type() != input.type():
        #    self.cf1.weight = self.cf1.weight.to(input)
        with torch.no_grad():
        #    self.cf1.weight = nn.Parameter(self.cf1.weight.abs())
            cf1_weight = self.cf0.weight.abs() # map the raw weights to be between 0-1
        if mask is not None:
            maxMask = self.cmp2D(mask)
            #update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), self.cf1.weight, bias=None, stride=self.stride, padding=self.padding,groups=self.groups)
            update_mask = F.conv2d(mask.expand(-1,self.inCh,-1,-1), cf1_weight, bias=None, stride=self.stride, padding=self.padding,groups=self.groups)
            denom = update_mask + 1e-6
            raw_numer = self.cf0(input*mask)
            output = raw_numer/denom
            return output, maxMask
        else:
            output = self.cf0(input)
            return output, mask
        