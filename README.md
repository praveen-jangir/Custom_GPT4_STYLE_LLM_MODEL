GPT-4 Mini: A Custom Transformer Language Model
🚀 Project Overview

GPT-4 Mini is a simplified GPT-4-style language model implemented in PyTorch. This project demonstrates:

Transformer decoder blocks with multi-head attention.

Rotary positional embeddings for better sequence modeling.

Byte-Pair Encoding (BPE) tokenizer using Hugging Face's tokenizers.

Training on custom text datasets, suitable for large corpora like Wikipedia or Common Crawl.

Text generation with temperature and top-k sampling.

This is ideal for learning, experimenting, and building custom LLMs on local or cloud GPUs.

🧩 Features

Subword tokenization using Byte-Level BPE.

Pre-norm Transformer decoder blocks.

Multi-head attention with causal masking.

Feedforward networks with GELU and gated activations.

Flexible text generation with temperature and top-k sampling.

Supports interactive testing with user prompts.

⚙️ Requirements

Python 3.10+

PyTorch 2.x

Hugging Face tokenizers

tqdm for progress bars

matplotlib (optional, for plotting loss)

pip install torch tqdm matplotlib tokenizers


Optional: If you have a GPU, PyTorch with CUDA is recommended for faster training.

🗂️ Project Structure
gpt4-mini/
│
├── main.py                # Main training and generation script
├── tokenizer/             # Directory to save BPE tokenizer files
├── data/
│   └── your_large_text_file.txt  # Training corpus
├── README.md
└── requirements.txt       # Required Python packages

🔧 Usage
1. Prepare Dataset

Place your training text files (e.g., Wikipedia dump) in the data/ folder. For example:

data/your_large_text_file.txt

2. Train Tokenizer

The script trains a BPE tokenizer automatically:

from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/your_large_text_file.txt"],
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
tokenizer.save_model("./tokenizer")

3. Train GPT-4 Mini

Run the training pipeline:

python main.py


The model is trained on sequences of length 64.

Default hyperparameters:

d_model=256, n_heads=8, n_layers=6

batch_size=16, epochs=5, seq_len=64

You can adjust these parameters for your GPU memory and dataset size.

4. Generate Text

The model supports interactive text generation:

prompt = "Artificial intelligence"
context = torch.tensor([tokenizer.encode(prompt).ids]).to(device)
generated = model.generate(context, max_length=100, temperature=0.8, top_k=10)
print(tokenizer.decode(generated[0].cpu().tolist()))


Temperature: Controls randomness (0.1 → deterministic, 2.0 → more creative)

Top-k: Limits selection to top k probable tokens for diversity.

5. Interactive Mode

Uncomment in main.py:

interactive_testing(model, tokenizer)


You can then input prompts in real-time to see generated text.

📖 Code Description
1. Tokenization

Uses Byte-Level BPE for subword tokenization.

Handles rare words, emojis, and large vocabularies.

Generates token IDs for model input.

2. Model Architecture

GPT4Mini: Stack of Transformer decoder blocks.

DecoderBlock:

Multi-head attention with causal masking

Feedforward network with GELU + gating

Layer normalization

Rotary positional embeddings for long-range context.

3. Training

Uses cross-entropy loss for next-token prediction.

Optimizer: AdamW

Scheduler: CosineAnnealingLR

Gradient clipping ensures stable training.

4. Generation

Supports temperature scaling and top-k sampling.

Iteratively predicts next tokens up to max length.

5. Testing

Script includes comprehensive tests:

Different prompts

Temperature variations

Top-k sampling effects

Edge-case handling

⚡ Tips for Best Results

Train on large datasets (Wikipedia, books, articles) for better generation.

Increase model size: d_model=512, n_layers=12 for richer language understanding.

Experiment with seq_len, top-k, and temperature for better output.

Use mixed-precision training with torch.cuda.amp for speed and memory efficiency.

📌 References

Vaswani et al., Attention is All You Need (2017)

Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers

OpenAI GPT: https://openai.com/research

⚙️ Future Improvements

Add gradient checkpointing for very deep models.

Integrate dataset streaming for massive corpora.

Implement top-p (nucleus) sampling for better diversity.

Save/load model checkpoints and tokenizer efficiently.

I can also create a ready-to-run zip package with:

main.py (full training + generation)

Pretrained BPE tokenizer

Example dataset

This way, you just download, run, and generate text.
