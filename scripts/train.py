import os
import sys
import math
import torch

# Ensure Python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import create_dataloaders
from src.model.gpt import GPTLanguageModel

# Hyperparameters
batch_size = 32
seq_len = 64          # Context window (block_size)
max_vocab_size = 10000

# Transformer Architecture Configuration
embed_dim = 128        # Dimensionality of the token embeddings
num_heads = 8         # Number of attention heads (head_size = 64/8 = 8)
num_blocks = 4        # Number of sequential Transformer blocks
dropout = 0.2

# Training Configuration
learning_rate = 3e-4  
eval_iters = 100
max_iters = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MLOps Hyperparameters
warmup_steps = 100        # Linearly increase LR for the first 100 steps
min_lr = 3e-5             # 10% of the peak learning rate (3e-4)


def get_lr_multiplier(step):
        # linear Warmup
        if step < warmup_steps:
            return step / warmup_steps
        
        # cosine Decay
        progress = (step - warmup_steps) / (max_iters - warmup_steps)
        progress = min(1.0, progress) # Cap at 1.0 just in case
        
        # ranges from 1.0 down to min_lr_ratio
        min_lr_ratio = min_lr / learning_rate
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters):
    """Evaluates the model without tracking gradients to prevent memory leaks."""
    out = {}
    model.eval()
    for split_name, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        data_iter = iter(loader)
        for k in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out

def main():
    print(f"Using device: {device}")
    
    # load data
    train_loader, val_loader, tokenizer = create_dataloaders(
        file_path="../data/raw/input.txt",
        batch_size=batch_size,
        seq_len=seq_len,
        max_vocab_size=max_vocab_size
    )
    
    # initialize GPT Model
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=seq_len,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout
    )
    model = model.to(device)
    
    # Print parameter count (A standard MLOps logging metric)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f} M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    print(f"\nStarting training for {max_iters} iterations...")
    train_iter = iter(train_loader)
    
    # training Loop
    for iter_step in range(max_iters):
        if iter_step % 200 == 0 or iter_step == max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters)
            print(f"Step {iter_step:04d}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")
            
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
            
        xb, yb = xb.to(device), yb.to(device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    print("\nTraining complete. Generating sample text...")
    
    # inference
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=100 , temperature=0.8 , top_k=10)[0].tolist()
    print("\n--- Generated Output ---")
    print(tokenizer.decode(generated_indices))
    print("------------------------")

if __name__ == "__main__":
    main()