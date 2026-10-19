import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of causal self-attention """

    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout: float = 0.2):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        
        
        mask = torch.tril(torch.ones(block_size,block_size))
        
        # register the tril tensor on to the gpu so pytorch understand this is not a parameter to be learned
        self.register_buffer('tril', mask)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        
        # Compute Queries, Keys, and Values. Output shape for each: (B, S, H)
        k = self.key(x)   
        q = self.query(x) 
        v = self.value(x) 
        
        # compute affinities: (B, S, H) @ (B, H, S) -> (B, S, S)
        # Scaled by 1/sqrt(H) to prevent the softmax from saturating
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        
        # causal mask tokens can only communicate with past tokens, not future ones
        wei = wei.masked_fill(self.tril[:S, :S] == 0, float('-inf'))
        
        # apply softmax on the last dim
        wei = F.softmax(wei,dim=-1)
        
        # apply droupout to randomly mask out connections to prevent confidence on some tokens
        wei = self.dropout(wei)
        
        # aggregate values based on attention probabilities: (B, S, S) @ (B, S, H) -> (B, S, H)
        out = wei @ v
        
        return out
    
    
    
    
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """

    def __init__(self, num_heads: int, head_size: int, embed_dim: int, block_size: int, dropout: float = 0.2):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)])        
        
        self.proj = nn.Linear(num_heads * head_size,embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        out = [h(x) for h in self.heads]
        
        out = torch.cat(out , -1)
        
        out = self.proj(out)
        
        out = self.dropout(out)
        
        return out
    
    
class FeedForward(nn.Module):
    """ A simple linear network followed by a non-linearity """

    def __init__(self, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim,embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.2):
        super().__init__()
        head_size = embed_dim // num_heads
        
        # instantiate the modules
        self.sa = MultiHeadAttention(num_heads, head_size, embed_dim, block_size, dropout)
        self.ffwd = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # first residual skip and apply layer norm to the input
        x = x + self.sa(self.ln1(x))
        
        # second residual skip and apply layer norm to the input again
        x = x + self.ffwd(self.ln2(x))
        
        return x