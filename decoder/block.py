import torch
from torch import nn
from multi_head_attention import MultiHeadAttention
from typing import Union
class DecoderBlock(nn.Module):
    """Transformer デコーダーブロックの実装"""

    def __init__(self, embed_dim, num_heads):
        super(DecoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.relu = nn.ReLU()
        
    def forward(
            self, 
            x: torch.Tensor, 
            mask: Union[None | torch.Tensor]  = None
        ) -> torch.Tensor:
        """Decoder Block の順伝播
        arguments:
        - x: (batch_size, seq_len, embed_dim)
        - mask: (1, 1, seq_len, seq_len) or None
        """

        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.relu(x))
        return x