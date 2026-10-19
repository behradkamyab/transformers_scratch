import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.attention import Block 

class GPTLanguageModel(nn.Module):
    """ Autoregressive Transformer Language Model """
    
    def __init__(self, vocab_size: int, embed_dim: int, block_size: int, num_heads: int, num_blocks: int, dropout: float = 0.2):
        super().__init__()
        self.block_size = block_size
        
        # token embeddings that maps integer token IDs to dense vectors
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        
        #  positional embeddings that injects sequence order information
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        
        # transformer Blocks that is the core computation engine
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_blocks)]
        )
        
        # final LayerNorm and Output Projection Head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, S = idx.shape
        device = idx.device
        
        # extract embeddings
        # tok_emb shape: (B, S, E)
        tok_emb = self.token_embedding_table(idx) 
        
        # pos_emb shape: (S, E) -> Broadcasts across the batch dimension to (B, S, E) during addition
        pos = torch.arange(S, dtype=torch.long, device=device)
        pos_emb = self.position_embedding_table(pos) 
        
        # Combine embeddings and pass through the network
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        
        # project back to vocabulary size
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            # Reshape for PyTorch's cross_entropy expectation
            B, S, V = logits.shape
            logits_reshaped = logits.view(B * S, V)
            targets_reshaped = targets.view(B * S)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """ Autoregressive text generation with Temperature, Top-K, and UNK masking """
        self.eval() 
        
        # In our WordTokenizer, <UNK> is always assigned index 0
        unk_token_id = 0 
        
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:] if idx.size(1) > self.block_size else idx
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus on the last time step: (B, V)
            logits_last_step = logits[:, -1, :]
            
            # UNK Masking
            # Force the model to never predict the <UNK> token by setting its score to -infinity
            logits_last_step[:, unk_token_id] = float('-inf')
            
            # Apply temperature scaling
            logits_last_step = logits_last_step / temperature
            
            # Apply Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits_last_step, min(top_k, logits_last_step.size(-1)))
                min_val = v[:, [-1]]
                logits_last_step[logits_last_step < min_val] = float('-inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(logits_last_step, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train() 
        return idx