import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm import tqdm
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------
# Dataset class for tokenized data
# ----------------------
class TokenizedDataset(Dataset):
    def __init__(self, encodings, seq_len=128):
        self.encodings = encodings
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.encodings) - self.seq_len
    
    def __getitem__(self, idx):
        chunk = self.encodings[idx:idx+self.seq_len+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ----------------------
# GPT-4 style model components (same as previous)
# ----------------------
def apply_rotary_pos_emb(q, k):
    seq_len = q.size(2)
    freqs = torch.arange(0, q.size(-1), 2.0, device=q.device) / q.size(-1)
    freqs = 1.0 / (10000 ** freqs)
    pos = torch.arange(seq_len, device=q.device).float()
    sinusoid_inp = torch.einsum('i,j->ij', pos, freqs)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    q = torch.stack([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1).flatten(-2)
    k = torch.stack([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1).flatten(-2)
    return q, k

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', None)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.register_buffer('causal_mask', mask)
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        q, k = apply_rotary_pos_emb(q, k)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        mask = self.causal_mask[:seq_len,:seq_len].to(x.device)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gate = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x)) * torch.sigmoid(self.gate(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class GPT4Mini(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_model*4, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)
    
    def forward(self, x):
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
    
    def generate(self, context, max_length=100, temperature=1.0, top_k=None):
        self.eval()
        context = context.to(device)
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(context)[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_token], dim=1)
                if context.size(1) > self.max_seq_len:
                    context = context[:, -self.max_seq_len:]
        return context

# ----------------------
# Training loop
# ----------------------
def train_model(model, dataloader, epochs=10, lr=3e-4):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(dataloader))
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        losses.append(avg_loss)
    return losses

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # 1️⃣ Train tokenizer on dataset (Wikipedia/large text)
    paths = ["your_large_text_file.txt"]  # e.g., Wikipedia dump
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=30000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    
    # Save and reload tokenizer
    tokenizer.save_model("./tokenizer")
    tokenizer.enable_truncation(max_length=256)
    
    # Encode text
    encodings = tokenizer.encode_batch(open(paths[0]).readlines())
    encodings_flat = [token_id for enc in encodings for token_id in enc.ids]
    
    # Dataset & Dataloader
    dataset = TokenizedDataset(encodings_flat, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Model
    model = GPT4Mini(vocab_size=tokenizer.get_vocab_size(), d_model=256, n_heads=8, n_layers=6)
    
    # Train
    losses = train_model(model, dataloader, epochs=5, lr=3e-4)
    
    # Generate
    prompt = "Artificial intelligence"
    context = torch.tensor([tokenizer.encode(prompt).ids]).to(device)
    generated = model.generate(context, max_length=100, temperature=0.8, top_k=10)
    decoded_text = tokenizer.decode(generated[0].cpu().tolist())
    print("Generated text:", decoded_text)
