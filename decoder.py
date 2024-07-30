import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_attentionBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)

    def forward(self,x):
        residual = x

        x = self.groupnorm(x)
        n,c,h,w = x.shape

        x = x.view((n,c,h*w))

        x= x.transpose(-1,-2)
        x = self.attention(x)
        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))
        x = x + residual
        return x
    



