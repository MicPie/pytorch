## (Squeeze-Excite) SE-XResNet v0.1 compiled by Michael M. Pieler
# Based on the following resources:
# https://github.com/fastai/fastai_dev/blob/master/dev/11_layers.ipynb
# https://github.com/fastai/fastai_dev/blob/master/dev/11a_vision_models_xresnet.ipynb
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py#L85

from fastai.torch_core import *
import torch.nn as nn
import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial
from fastai.torch_core import Module
import torch.nn.functional as F


## Try self-attention module?
## Try better init?


# Unmodified from: https://github.com/fastai/fastai_dev/blob/master/dev/11_layers.ipynb
NormType = Enum('NormType', 'Batch BatchZero Weight Spectral')

def BatchNorm(nf, norm_type=NormType.Batch, ndim=2, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"BatchNorm{ndim}d")(nf, **kwargs)
    bn.bias.data.fill_(1e-3)
    bn.weight.data.fill_(0. if norm_type==NormType.BatchZero else 1.)
    return bn


# Unmodified from: https://github.com/fastai/fastai_dev/blob/master/dev/11_layers.ipynb
def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')
                   
defaults.activation=nn.ReLU

class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch,
                 act_cls=defaults.activation, transpose=False, init=nn.init.kaiming_normal_, xtra=None):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None: bias = not bn
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        if act_cls is not None: layers.append(act_cls())
        if bn: layers.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if xtra: layers.append(xtra)
        super().__init__(*layers)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full=False): self.full = full
    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)


# SE block based on: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py#L85
class SE_Module(Module): # change nn.Module to Module

    def __init__(self, channels, reduction=16):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# Based on: https://github.com/fastai/fastai_dev/blob/master/dev/11_layers.ipynb
class SE_ResBlock(nn.Module):
    "Resnet block from `ni` to `nh` with `stride`"
    def __init__(self, expansion, ni, nh, stride=1, norm_type=NormType.Batch, **kwargs):
        super().__init__()
        norm2 = NormType.BatchZero if norm_type==NormType.Batch else norm_type
        nf,ni = nh*expansion,ni*expansion
        layers  = [ConvLayer(ni, nh, 3, stride=stride, norm_type=norm_type, **kwargs),
                   ConvLayer(nh, nf, 3, norm_type=norm2, act_cls=None)
        ] if expansion == 1 else [
                   ConvLayer(ni, nh, 1, norm_type=norm_type, **kwargs),
                   ConvLayer(nh, nh, 3, stride=stride, norm_type=norm_type, **kwargs),
                   ConvLayer(nh, nf, 1, norm_type=norm2, act_cls=None, **kwargs)
        ]
        self.convs = nn.Sequential(*layers, SE_Module(nf))
        self.idconv = noop if ni==nf else ConvLayer(ni, nf, 1, act_cls=None, **kwargs)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
        #self.act = defaults.activation(inplace=True)
        self.act = defaults.activation() # does not work with Mish

    def forward(self, x): return self.act(self.convs(x) + self.idconv(self.pool(x)))


# Unmodified from: https://github.com/fastai/fastai_dev/blob/master/dev/11a_vision_models_xresnet.ipynb
def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)


# Based on: https://github.com/fastai/fastai_dev/blob/master/dev/11a_vision_models_xresnet.ipynb
class SE_XResNet(nn.Sequential):
    def __init__(self, expansion, layers, c_in=3, c_out=1000):
        
        stem = []
        sizes = [c_in,32,32,64]
        
        for i in range(3):
            stem.append(ConvLayer(sizes[i], sizes[i+1], stride=2 if i==0 else 1))

        block_szs = [64//expansion,64,128,256,512]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], l, 1 if i==0 else 2)
                  for i,l in enumerate(layers)]
        super().__init__(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            AdaptiveConcatPool2d(), Flatten(), # instead of nn.AdaptiveAvgPool2d(1)
            nn.Linear(block_szs[-1]*expansion*2, c_out) # multiply by 2 because we use AdaptiveConcatPool2d()!
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride):
        return nn.Sequential(
            *[SE_ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
              for i in range(blocks)])