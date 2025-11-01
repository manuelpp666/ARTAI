# ================================================================
# app/core/nlp_module/transformer_v3.py
# Versión mejorada — arquitectura pre-norm + GELU + dropout estable
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------
# Codificación Posicional
# ------------------------------
class CodificacionPosicional(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1).clamp(max=1e4)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ------------------------------
# Multi-Head Attention mejorada
# ------------------------------
class AtencionMultiCabeza(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.linear_q(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # -------------------------
        # 🔒 Atención estable numéricamente
        # -------------------------
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Evita valores infinitos por overflow
        scores = scores - scores.max(dim=-1, keepdim=True)[0]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)  # limpia NaN/Inf
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)

        out = self.linear_out(out)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)  # 🔒 salida limpia
        return out


# ------------------------------
# FeedForward con GELU + Dropout
# ------------------------------
class RedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ------------------------------
# Capa Encoder (Pre-Norm)
# ------------------------------
class CapaEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.att = AtencionMultiCabeza(d_model, num_heads, dropout)
        self.ff = RedFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.att(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ------------------------------
# Transformer completo
# ------------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=192, N=4, num_heads=6, d_ff=768, max_len=250, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = CodificacionPosicional(d_model, max_len)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([CapaEncoder(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm_final = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def generar_mascara_subsecuente(self, sz):
        """Máscara triangular (no ver tokens futuros)"""
        return torch.triu(torch.ones(sz, sz), 1).bool()

    def forward(self, x):
        mask = self.generar_mascara_subsecuente(x.size(1)).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        x = self.emb_dropout(self.pe(self.embedding(x)))
        for capa in self.layers:
            x = capa(x, mask)
        x = self.norm_final(x)
        return self.out(x)
