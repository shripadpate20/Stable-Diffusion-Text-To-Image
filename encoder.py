import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_attentionblock, VAE_residualblock

class VAE_Enocder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            VAE_residualblock(128,128),
            VAE_residualblock(128,128),
            nn.Conv2d(28, 128, kernel_size=3, stride=2, padding=0),
            VAE_residualblock(128,256),
            VAE_residualblock(256,256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            VAE_residualblock(256,512),
            VAE_residualblock(512,512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            VAE_residualblock(512,512),
            VAE_residualblock(512,512),
            VAE_residualblock(512,512),
            VAE_residualblock(512),
            VAE_residualblock(512,512),
            nn.GroupNorm(32,512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
             nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )
    def forward(self,x,noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                 x = F.pad(x, (0, 1, 0, 1))

            x = module(x)
        mean , log_var = torch.chunk(x,2,din =1)
        log_var = torch.clamp(log_var,-30,20)
        var = log_var.exp()
        stdev = var.sqrt()
        x = mean + stdev * noise
        x *= 0.18215
        return x
