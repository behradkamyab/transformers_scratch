# Autoregressive Transformer from Scratch: Project Checklist

## Notation Key
* **B** = `Batch_Size`
* **S** = `Seq_Len` (Context window)
* **E** = `Embed_Dim` (Hidden vector size)
* **H** = `Head_Dim` ($H = E / \text{num\_heads}$)
* **V** = `Vocab_Size` (Vocabulary size)

---

## Pillar 1: Manage & Setup
- [ ] Initialize Git repository (`git init`)
- [ ] Set up virtual environment and `requirements.txt`
- [ ] Create standardized modular folder structure (`src/`, `data/`, `scripts/`, `tests/`, `docs/`)
- [ ] Track Work Packages (WP1 to WP4) and milestones

---

## Pillar 2: Data Pipeline & Baseline (Data-Centric)
- [ ] **Data Ingestion:** Download raw `tinyshakespeare.txt` corpus into `data/raw/`
- [ ] **Tokenizer:** Build character-level encoder/decoder mapping ($V \approx 65$)
- [ ] **Dataset Partitions:** Implement a strict 90% train / 10% validation split
- [ ] **Batch Generator:** Construct `(B, S)` context/target tensors shifted by 1 time step
- [ ] **Baseline Model:** Implement a Bigram Language Model (`D0 + M0`) to establish the reference cross-entropy loss and validation perplexity

---

## Pillar 3: Model Development (Model-Centric)
- [ ] **Embedding Layer:** Token embedding `(B, S, E)` + Learned Positional Embedding `(S, E)`
- [ ] **Single Attention Head:** Implement `CausalSelfAttentionHead` with lower-triangular causal mask
- [ ] **Multi-Head Attention:** Implement vectorized 4D tensor attention `(B, num_heads, S, H)`
- [ ] **Transformer Block:** Wrap attention and 2-layer MLP with Pre-LayerNorm and residual connections
- [ ] **Autoregressive GPT Wrapper:** Assemble $N$ blocks with final `LayerNorm` and linear projection head (`lm_head`) to produce logits `(B, S, V)`
- [ ] **Sampling Engine:** Implement inference loop supporting Greedy, Temperature scaling, and Top-$k$ filtering

---

## Pillar 4: Evaluation & Repository Artifacts
- [ ] **Training Pipeline:** Implement CLI training script with AdamW optimizer, learning rate scheduling, and periodic validation loss logging
- [ ] **Experiment Logging:** Record comparison table (Bigram Baseline vs. Transformer variations)
- [ ] **Unit Tests:** Add shape assertions and forward-pass checks under `tests/`
- [ ] **Documentation:** Write a production README featuring a functional architecture diagram, tensor dimension walk-through, and quantitative results