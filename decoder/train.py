from MiniGPT import MiniGPT
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

with open("input.txt", "r",encoding="shift_jis") as f:
    text_data = f.read()

chars = sorted(list(set(text_data)))
vocab_size = len(chars)

# 対応関係
idx_to_char = {i: ch for i, ch in enumerate(chars)}
char_to_idx = {ch: i for i, ch in enumerate(chars)}

def encode(text):
    return torch.tensor([char_to_idx[c] for c in text])

def decode(ids):
    return ''.join([idx_to_char[i] for i in ids])

encoded_data = encode(text_data)
n = int(0.9 * len(encoded_data))
train_data = encoded_data[:n]
val_data = encoded_data[n:]

block_size = 128   
batch_size = 32

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y 

# モデルの初期化
model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=512,
    num_heads=8,
    num_layers=6,
    max_seq_len=128
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


max_steps = 100

for step in range(max_steps):
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)
    
    logits = model(xb)
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        yb.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        with torch.no_grad():
            vb, vy = get_batch("val")
            v_logits = model(vb)
            v_loss = nn.functional.cross_entropy(
                v_logits.view(-1, vocab_size),
                vy.view(-1)
            )
        print(f"step {step}: train loss {loss.item():.4f}, val loss {v_loss.item():.4f}")

def generate(model, start_text="吾輩は", max_new_tokens=300):
    model.eval()
    ids = encode(start_text).to(device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if ids.size(1) > model.max_seq_len:
            ids = ids[:, -model.max_seq_len:]

        logits = model(ids)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return decode(ids[0].tolist())


print(generate(model))
