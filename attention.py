import torch
import torch.nn as nn
from torch.nn import functional as F
import math



class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, Dim: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(Dim, 3 * Dim, bias = in_proj_bias)

        self.out_proj = nn.Linear(Dim, Dim, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = Dim // self.n_heads

    def forward(self, x, causal_mask = False):
        # x ->(Batch_size, Seq_len, Dim)
        batch_size, seq_len , Dim = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        #(batch_size, seq_len, H, Dim/H) ->(batch_size, H, seq_len, Dim/H)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q,k.transpose(-1, -2))

        if causal_mask:
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -torch.inf)

        scores = scores / math.sqrt(self.d_head)

        scores = F.softmax(scores, dim = -1)

        output = torch.matmul(scores, v)
        #(batch_size, H, seq_len, Dim/H) -> (batch_size,seq_len,H, Dim/H) -> (batch_size,seq_len,Dim)
        output = output.transpose(1, 2).reshape(x.shape)

        output = self.out_proj(output)

        return output
    

class CrossAttention(nn.Module):

    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        
        weight = q @ k.transpose(-1, -2)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output