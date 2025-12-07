import torch
from torch import nn
from block import DecoderBlock

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(MiniGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """MiniGPT の順伝播
        arguments:
        - x: (batch_size, seq_len)
        """
        
        B, S = x.size()
        positions = torch.arange(0, S, device=x.device).unsqueeze(0).expand(B, S)
        x = self.token_embedding(x) + self.position_embedding(positions)

        causal_mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits