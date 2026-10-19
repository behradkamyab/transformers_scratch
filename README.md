# GPT From Scratch: Autoregressive Language Model

A pure PyTorch implementation of a Decoder-only Transformer built from the ground up.

Using high-level libraries like Hugging Face makes it easy to run models in a few lines of code, but it hides how the model actually works inside. I built this project to understand and code the core math of a Large Language Model (LLM) from scratch. It covers the full pipeline: loading raw text, causal masking, building multi-head attention, applying learning rate schedules, and controlling how text is generated.

## System Architecture and Tensor Shapes

To make sure all matrix multiplications work without shape errors, the code tracks five main dimensions during the forward pass:

* $B$ **(`Batch_Size`):** Number of text sequences processed at the same time (`32`)
* $S$ **(`Seq_Len`):** Maximum context window length (`64`)
* $E$ **(`Embed_Dim`):** Size of the embedding vector for each token (`128`)
* $H$ **(`Head_Dim`):** Size of each attention head ($H = E / \text{num\_heads} = 128 / 8 = 16$)
* $V$ **(`Vocab_Size`):** Total unique words in the vocabulary (`10,000`)

```text
Input Indices (B, S) 
  │
  ├─► Token Embedding Table (B, S, E) 
  │         + 
  └─► Positional Embedding Table (S, E) ──► Addition ──► x0 (B, S, E)
                                                              │
  ┌───────────────────────────────────────────────────────────┘
  ▼
[ Transformer Block x N ]
  │
  ├─► x = x + MultiHeadAttention(LayerNorm(x))   <-- Pre-Norm Residual 1 (Communication)
  │     ├─► Q, K, V Projections: (B, S, E) -> (B, S, H) per head
  │     ├─► Attention Scores: (Q @ K^T) / sqrt(H) -> (B, S, S)
  │     ├─► Causal Masking: Lower-triangular -inf mask -> Softmax
  │     └─► Output Projection: Concat(heads) @ W_O -> (B, S, E)
  │
  └─► x = x + FeedForward(LayerNorm(x))          <-- Pre-Norm Residual 2 (Computation)
        ├─► Linear Expansion: (B, S, E) -> (B, S, 4E) + ReLU
        └─► Linear Projection: (B, S, 4E) -> (B, S, E)
  │
  ▼
Final LayerNorm (B, S, E)
  │
  ▼
LM Projection Head (B, S, V) ──► Cross-Entropy Loss / Next-Word Sampling
```

## Hyperparameter Configuration

The final 3.37M parameter model was trained using the following configuration in `scripts/train.py`:

| Category | Hyperparameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Data & Context** | `batch_size` | `32` | Sequences processed per step |
| | `seq_len` | `64` | Context window length (`block_size`) |
| | `max_vocab_size` | `10000` | Top words kept in vocabulary |
| **Architecture** | `embed_dim` | `128` | Token and positional embedding dimension |
| | `num_heads` | `8` | Parallel attention heads (`head_size = 16`) |
| | `num_blocks` | `4` | Stacked Transformer layers |
| | `dropout` | `0.2` | Dropout probability for regularization |
| **Training & MLOps** | `learning_rate` | `3e-4` | Peak learning rate for AdamW |
| | `min_lr` | `3e-5` | Minimum learning rate after cosine decay |
| | `warmup_steps` | `100` | Linear warmup steps at start of training |
| | `max_iters` | `5000` | Total training iterations |
| | `eval_iters` | `100` | Batches used to estimate train/val loss |

## What Was Implemented

### 1. Data Pipeline (`src/dataset.py`)
* **Word-Level Tokenizer:** Built a regex-based tokenizer using the **WikiText-2** dataset. The vocabulary is limited to the top $10,000$ most common words. Any rare word outside this list is mapped to an `<UNK>` (unknown) token at index `0`.
* **Next-Token Shifting:** Built custom PyTorch `Dataset` and `DataLoader` classes that create `(B, S)` input batches and shift the target labels forward by one position ($t+1$) for next-word prediction.

### 2. Baseline Model (`src/model/baseline_model.py`)
* **Bigram Language Model ($M_0$):** Before building the Transformer, I created a simple Bigram model. It predicts the next word by looking only at the single current word using a $V \times V$ lookup table. This gave a starting validation loss of **9.022** to beat.

### 3. The Transformer Model (`src/model/attention.py` & `src/model/gpt.py`)
* **Scaled Dot-Product Attention:** Implemented $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$. A lower-triangular mask ($M$) fills future token positions with $-\infty$ before the softmax step so the model cannot cheat by looking ahead.
* **Multi-Head Attention:** Runs 8 attention heads (`head_size = 16`) in parallel using `nn.ModuleList`. This lets different heads focus on different parts of the sentence (like punctuation, grammar, or nearby nouns) before combining their outputs back to `embed_dim = 128`.
* **Pre-Norm Residual Connections:** Applies `LayerNorm` *before* the Attention and FeedForward layers ($x + f(\text{LN}(x))$), following modern designs like GPT-2 and Llama for much more stable training.

### 4. Training Pipeline and Testing (`scripts/train.py` & `tests/test_model.py`)
* **Warmup and Cosine Learning Rate Decay:** Training Transformers with a high learning rate right away can cause unstable gradients. I added a `LambdaLR` scheduler that slowly increases the learning rate from `0.0` to `3e-4` over the first 100 steps, and then lowers it using a cosine curve down to `3e-5`.
* **Memory-Safe Evaluation:** Uses `@torch.no_grad()` during validation checks so PyTorch does not store extra gradients in GPU memory.
* **Unit Tests:** Created a `pytest` suite (`tests/test_model.py`) that passes dummy tensors through `Head`, `MultiHeadAttention`, `Block`, and `GPTLanguageModel` to verify that output shapes are always exact.

### 5. Text Generation Controls (`GPTLanguageModel.generate`)
Picking random words straight from the raw output probabilities often produces bad text. I added three controls to improve generation:
* **`<UNK>` Masking:** Because `<UNK>` appears often in the training data, the model tries to generate it all the time. During inference, index `0` is set to $-\infty$ so the model is forced to pick real words.
* **Temperature Scaling:** Divides logits by a temperature value ($T = 0.8$) before softmax to make confident words even more likely.
* **Top-$K$ Sampling:** Limits the model to choosing only from the top $K = 10$ most likely words at each step, cutting off rare and random words.

## Experimental Results

All models were trained and tested on the same 90/10 train/validation split of WikiText-2 (~2.33M training tokens, ~259k validation tokens).

| Model Architecture | Trainable Parameters | Context Window ($S$) | Final Train Loss | Final Val Loss | Validation Perplexity ($e^{\text{Loss}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bigram Baseline ($M_0$)** | 100.00 M *(Lookup)* | 1 token | 6.8410 | 9.0220 | ~8,283.3 |
| **Custom GPT (`E=64`, 4-Head, 1k steps)** | 1.49 M | 64 tokens | 5.6897 | 5.9110 | ~369.1 |
| **Custom GPT (`E=128`, 8-Head, 5k steps + LR Decay)** | **3.37 M** | **64 tokens** | **4.6770** | **5.0778** | **~160.4** |

### What Perplexity Shows
Perplexity ($e^{\text{Loss}}$) measures how many words the model is choosing between when predicting the next word:
* The **Bigram Baseline** has a perplexity of **~8,283** out of 10,000 words, which means it is mostly guessing.
* The **3.37M GPT Model** lowers perplexity to **~160.4** while using **96.6% fewer parameters**, showing that the attention layers successfully learn patterns from the context window.

## Understanding the Generated Output

After training for 5,000 steps on a single GPU, generating 100 tokens gives output like this:

> `"= = = = = = History = = = = = = = = = = The first published in January 10 : 00 , the team also thegame ' s second quarter . The first season of the first time of the season in the season , and he was also won the second season . = = = = = On July 2008 , the first season , the first game ' s final season..."`

While the paragraph does not tell a logical story, it proves that the Transformer architecture is working properly:

1. **Learning Grammar and Formatting:** Without any hardcoded rules, the model learned how Wikipedia formats section headers (`= = = History = = =`), how timestamps are written (`January 10 : 00`), how apostrophes work (`game ' s`), and how to build grammatically valid phrases (`"On July 2008 , the first season ,"`).
2. **Why Small Models Repeat Words:** A 3.37M parameter model with a 64-token context window is too small to hold long-term meaning. When it generates a common word like `"season"`, that word fills up the short 64-token memory, making `"season"` the most likely word to predict again.
3. **Word-Level Tokenizer Limits:** Using whole words shows why modern LLMs use subword tokenizers (like Byte-Pair Encoding). Word-level tokenizers treat `"play"`, `"playing"`, and `"played"` as completely separate words and lose information whenever a word is replaced by `<UNK>`.

## Project Structure

```text
transformers_scratch/
├── data/
│   ├── processed/            # Processed data storage
│   └── raw/
│       └── input.txt         # WikiText-2 raw text corpus
├── docs/                     # Project documentation and notes
├── scripts/
│   ├── download_dataset.py   # Script to download WikiText-2
│   └── train.py              # Training loop, LR scheduler, and evaluation
├── src/
│   ├── model/
│   │   ├── attention.py      # Head, MultiHeadAttention, FeedForward, Block
│   │   ├── baseline_model.py # Bigram Language Model (M0 benchmark)
│   │   └── gpt.py            # GPTLanguageModel wrapper and generate() loop
│   └── dataset.py            # WordTokenizer, TextDataset, DataLoader setup
├── tests/
│   └── test_model.py         # Pytest unit tests for tensor shapes
├── .gitignore
├── README.md
└── requirements.txt          # Python dependencies
```

## How to Run

**1. Clone the repository and install dependencies**
```bash
git clone https://github.com/behradkamyab/transformers_scratch.git
cd transformers_scratch
pip install -r requirements.txt
```

**2. Download the dataset**
```bash
python scripts/download_dataset.py
```

**3. Run the unit tests**
Check that all model layers output the exact expected tensor shapes:
```bash
pytest tests/test_model.py
```

**4. Train the model**
Run the training loop and generate sample text at the end:
```bash
python scripts/train.py
```