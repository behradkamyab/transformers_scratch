import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token from a V x V lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass for the Bigram model.
        idx shape: (B, S) - the input context
        targets shape: (B, S) - the expected next tokens
        """
        # logits are the raw, unnormalized scores for the next word
        # Shape: (B, S, V)
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            # PyTorch's cross_entropy expects shape (B*S, V) for logits and (B*S) for targets
            B, S, V = logits.shape
            logits_reshaped = logits.view(B * S, V)
            targets_reshaped = targets.view(B * S)
            
            # calculate the Cross-Entropy Loss
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens autoregressively.
        idx is the current context of shape (B, S)
        """
        for _ in range(max_new_tokens):
            # Get the predictions (ignoring loss since we don't have targets)
            logits, _ = self(idx)
            
            # focus only on the very last time step in the sequence
            # Shape becomes (B, V)
            logits_last_step = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits_last_step, dim=-1)
            
            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # Shape: (B, 1)
            
            # append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # Shape: (B, S+1)
            
        return idx