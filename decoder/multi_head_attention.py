from scaled_dt_attention import scaled_dot_product_attention
import torch
from torch import nn
from typing import Union

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim # embedding dimension
        self.num_heads = num_heads # number of attention heads
        self.head_dim = embed_dim // num_heads # dimension per head

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(
            self, 
            x: torch.Tensor, 
            mask: Union[None | torch.Tensor] = None
        ) -> torch.Tensor:
        
        """Multi-Head Attention の実装 
        arguments:
        - x: (batch_size, seq_len, embed_dim)
        - mask: (1, 1, seq_len, seq_len) or None
        """

        B,S,C = x.size() # batch size, sequence length, embedding dimension

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # head splitting
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)

        attn = scaled_dot_product_attention(q, k, v, mask) # (B, num_heads, S, head_dim)

        attn = attn.transpose(1, 2).contiguous().view(B, S, C) # (B, S, embed_dim)
        out = self.out_proj(attn) # (B, S, embed_dim)
        return out

if __name__ == "__main__":
    # Example usage

    batch_size = 2
    seq_len = 5
    embed_dim = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, embed_dim)
    mha = MultiHeadAttention(embed_dim, num_heads)
    output = mha(x)
    print(output.shape)  # Expected output shape: (batch_size, seq_len, embed_dim)