# GPT-4 Mini: A Custom Transformer Language Model

## Project Overview
**GPT-4 Mini** is a simplified GPT-4-style language model implemented in PyTorch. This project demonstrates:

- Transformer decoder blocks with multi-head attention.
- Rotary positional embeddings for better sequence modeling.
- Byte-Pair Encoding (BPE) tokenizer using Hugging Face's tokenizers.
- Training on custom text datasets, suitable for large corpora like Wikipedia or Common Crawl.
- Text generation with temperature and top-k sampling.

This is ideal for learning, experimenting, and building custom LLMs on local or cloud GPUs.

---

## Features
- Subword tokenization using Byte-Level BPE.
- Pre-norm Transformer decoder blocks.
- Multi-head attention with causal masking.
- Feedforward networks with GELU and gated activations.
- Flexible text generation with temperature and top-k sampling.
- Supports interactive testing with user prompts.

---

## Requirements
- Python 3.10+
- PyTorch 2.x
- Hugging Face `tokenizers`
- `tqdm` for progress bars
- `matplotlib` (optional, for plotting loss)

Install dependencies:

```bash
pip install torch tqdm matplotlib tokenizers
