# app/core/nlp_module/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding
class CodificacionPosicional(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Multi-Head Attention
class AtencionMultiCabeza(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.linear_q(q).view(B, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.linear_k(k).view(B, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.linear_v(v).view(B, -1, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(B, -1, self.num_heads*self.d_k)
        return self.linear_out(out)

# FeedForward
class RedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# Encoder Layer
class CapaEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        super().__init__()
        self.att = AtencionMultiCabeza(d_model, num_heads)
        self.ff = RedFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x + self.dropout(self.att(x, x, x, mask)))
        x3 = self.norm2(x2 + self.dropout(self.ff(x2)))
        return x3

# Transformer completo con máscara triangular
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, N=3, num_heads=8, d_ff=1024, max_len=250):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = CodificacionPosicional(d_model, max_len)
        self.layers = nn.ModuleList([CapaEncoder(d_model, num_heads, d_ff) for _ in range(N)])
        self.out = nn.Linear(d_model, vocab_size)

    def generar_mascara_subsecuente(self, sz):
        """Máscara triangular para no ver tokens futuros"""
        mask = torch.triu(torch.ones(sz, sz), 1).bool()
        return mask

    def forward(self, x):
        mask = self.generar_mascara_subsecuente(x.size(1)).to(x.device)
        x = self.embedding(x)
        x = self.pe(x)
        for capa in self.layers:
            x = capa(x, mask)
        return self.out(x)
