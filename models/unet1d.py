import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# small helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.ConvTranspose1d(dim, default(dim_out, dim), 4, 2, 1)


def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=9):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel_size, padding=kernel_size // 2)  
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=9):
        super().__init__()
        self.block1 = Block(dim, dim_out, kernel_size=kernel_size)  
        self.block2 = Block(dim_out, dim_out, kernel_size=kernel_size)  
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class NoiseGating(nn.Module):
    def __init__(self, dim=1, dim_out=64, heads=4, hidden_dim=32):
        super().__init__()
        self.heads = heads
        hidden_dim = hidden_dim * heads    
        self.to_sn = nn.Conv1d(dim, hidden_dim * 2, kernel_size=7, padding= 7 // 2, bias = False)
        self.sigmoid = nn.Sigmoid()
        self.to_out = nn.Conv1d(hidden_dim, dim_out, 1)

    def forward(self, x):
        b, c, n = x.shape
        sn = self.to_sn(x).chunk(2, dim = 1)
        signal, noise = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), sn)        
        gate = self.sigmoid(noise)
        out = gate * signal 
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        out = self.to_out(out)
        return out


class CompetitiveGating(nn.Module):
    def __init__(self, dim, heads = 4):
        super().__init__()
        self.heads = heads
        hidden_dim = dim * heads
        self.to_sn = nn.Conv1d(dim, hidden_dim * 2, kernel_size=7, padding= 7 // 2, bias = False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        sn = self.to_sn(x).chunk(2, dim = 1)
        signal, noise = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), sn)
        gate = self.softmax(noise)
        out = gate * signal
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads) 
        out = self.to_out(out)
        return out 

# model
class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        channels = 1,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        init_dim = dim
        out_dim = channels
        
        input_channels = channels 
        
        self.init_conv = NoiseGating(input_channels, dim_out=64)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, kernel_size=9),
                Residual(CompetitiveGating(dim_in)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 9, padding=4)
            ]))
    
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, kernel_size=9)  
        self.mid_gate = Residual(CompetitiveGating(mid_dim))                   
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, kernel_size=9)                    

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, kernel_size=9),                    
                Residual(CompetitiveGating(dim_out)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 9, padding=4)  

            ]))

        self.final_res_block = ResnetBlock(dim * 2, dim, kernel_size=9)  
        self.final_conv = nn.Conv1d(dim, out_dim, 1)  

    def forward(self, x):
        x = self.init_conv(x)        
        r = x.clone()
        h = []

        for block, gate, downsample in self.downs:
            x = block(x)
            x = gate(x)
            h.append(x)
            
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_gate(x)       
        x = self.mid_block2(x)


        for block, gate, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block(x)
            x = gate(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x)    
        x = self.final_conv(x)

        return x

