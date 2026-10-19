import os
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

class WordTokenizer:
    """
    Word-level Tokenizer supporting a maximum vocabulary size and <UNK> tokens.
    """
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.unk_token = "<UNK>"
        
    def tokenize(self, text: str) -> list[str]:
        """Splits text into words and individual punctuation marks."""
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def fit(self, text: str):
        """Builds the vocabulary mapping based on token frequency."""
        tokens = self.tokenize(text)
        word_counts = Counter(tokens)
        
        # sort desc and alphabetically
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        self.stoi = {self.unk_token: 0}
        self.itos = {0: self.unk_token}
        
        idx = 1
        for word, _ in sorted_words:
            if idx >= self.max_vocab_size:
                break
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1
            
        self.vocab_size = len(self.stoi)
        print(f"Vocabulary built: {self.vocab_size} unique tokens.")

    def encode(self, text: str) -> list[int]:
        """Converts a string into a list of integer token IDs."""
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi[self.unk_token]) for token in tokens]

    def decode(self, indices: list[int]) -> str:
        """Converts integer token IDs back into a string."""
        tokens = [self.itos.get(idx, self.unk_token) for idx in indices]
        return " ".join(tokens)


class TextDataset(Dataset):
    """
     Dataset yielding sequential batches for autoregressive training.
    """
    def __init__(self, data_tensor: torch.Tensor, seq_len: int):
        self.data_tensor = data_tensor
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data_tensor) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Context (x) and shifted target (y)
        x = self.data_tensor[idx : idx + self.seq_len]
        y = self.data_tensor[idx + 1 : idx + self.seq_len + 1]
        return x, y


def create_dataloaders(
    file_path: str = "../data/raw/input.txt", 
    batch_size: int = 32, 
    seq_len: int = 64, 
    max_vocab_size: int = 10000
) -> tuple[DataLoader, DataLoader, WordTokenizer]:
    """
    Loads raw text, builds the tokenizer, splits into Train/Val (90/10),
    and returns PyTorch DataLoaders
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run download script first.")

    print(f"Loading text from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = WordTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.fit(text)

    print("Encoding text into token IDs...")
    data_ids = tokenizer.encode(text)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)

    # Train/Validation Split
    n = int(0.9 * len(data_tensor))
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    train_dataset = TextDataset(train_data, seq_len)
    val_dataset = TextDataset(val_data, seq_len)

    # shuffle=True is critical for train_loader to ensure the model sees diverse context windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test the pipeline
    B, S, V_max = 32, 64, 5000
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=B, seq_len=S, max_vocab_size=V_max
    )
    
    # Fetch one batch
    x, y = next(iter(train_loader))
    print(f"\nBatch X shape: {tuple(x.shape)} -> (B, S)")
    print(f"Batch Y shape: {tuple(y.shape)} -> (B, S)")
    
    print("\nSample X decoded:")
    print(tokenizer.decode(x[0].tolist()))
    print("\nSample Y decoded (shifted right by 1):")
    print(tokenizer.decode(y[0].tolist()))