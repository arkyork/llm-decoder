import torch
from torch import nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    # query, key, value: (batch_size, num_heads, seq_len, head_dim)

    d_k = query.size(-1)

    print(d_k)

    score_qk = query @ key.transpose(-2, -1) / (d_k ** 0.5) #  (batch_size, num_heads, seq_len, seq_len)
    
    attention_score = F.softmax(score_qk, dim=-1)

    out = attention_score @ value

    return out


# Example usage
batch_size = 2
num_heads = 4
seq_len = 5
head_dim = 8
query = torch.randn(batch_size, num_heads, seq_len, head_dim)
key = torch.randn(batch_size, num_heads, seq_len, head_dim)
value = torch.randn(batch_size, num_heads, seq_len, head_dim)
output = scaled_dot_product_attention(query, key, value)
print(output)  # Expected output shape: (batch_size, num_heads, seq_len, head_dim)