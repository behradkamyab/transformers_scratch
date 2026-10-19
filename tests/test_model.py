import sys
import os
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.attention import Head, MultiHeadAttention, Block
from src.model.gpt import GPTLanguageModel

# Dummy Configuration for testing
BATCH_SIZE = 4
SEQ_LEN = 16
EMBED_DIM = 32
NUM_HEADS = 4

@pytest.fixture
def dummy_input():
    # shape: (B, S, E) -> Used for Attention and Blocks
    return torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

@pytest.fixture
def dummy_indices():
    # shape: (B, S) -> Used for the GPT Language Model input
    return torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN))

def test_single_attention_head(dummy_input):
    head_size = EMBED_DIM // NUM_HEADS
    head = Head(embed_dim=EMBED_DIM, head_size=head_size, block_size=SEQ_LEN)
    
    out = head(dummy_input)
    
    # single head outputs (B, S, H)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, head_size), \
        f"Expected {(BATCH_SIZE, SEQ_LEN, head_size)}, got {out.shape}"

def test_multi_head_attention(dummy_input):
    head_size = EMBED_DIM // NUM_HEADS
    mha = MultiHeadAttention(num_heads=NUM_HEADS, head_size=head_size, 
                             embed_dim=EMBED_DIM, block_size=SEQ_LEN)
    
    out = mha(dummy_input)
    
    # MHA concatenates heads back to (B, S, E)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM), \
        f"Expected {(BATCH_SIZE, SEQ_LEN, EMBED_DIM)}, got {out.shape}"

def test_transformer_block(dummy_input):
    block = Block(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, block_size=SEQ_LEN)
    
    out = block(dummy_input)
    
    # block maintains the exact same shape (B, S, E)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM), \
        f"Expected {(BATCH_SIZE, SEQ_LEN, EMBED_DIM)}, got {out.shape}"

def test_gpt_forward_pass(dummy_indices):
    vocab_size = 100
    model = GPTLanguageModel(vocab_size=vocab_size, embed_dim=EMBED_DIM, 
                             block_size=SEQ_LEN, num_heads=NUM_HEADS, num_blocks=2)
    
    logits, loss = model(dummy_indices)
    
    # final output maps back to the vocabulary size: (B, S, V)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, vocab_size), \
        f"Expected {(BATCH_SIZE, SEQ_LEN, vocab_size)}, got {logits.shape}"
    
    # without targets provided, loss should be None
    assert loss is None