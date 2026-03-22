"""
Two micro-scale language models sharing the same transformer backbone.
The only difference is the embedding strategy:

  - MicroLLM_HTP:      HTP deterministic embedding (0 params) + small learned (24-dim)
  - MicroLLM_Standard:  Standard nn.Embedding (learned, dim params per token)

Both use: RMSNorm, RoPE, SwiGLU, causal attention, 2 layers, 4 heads, dim=80.

Reference: arXiv:2511.20665 (Schmitz, 2025) — Harmonic Token Projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from htp_embedding import HTPWordEmbedding


# ─── Shared transformer components ──────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_hidden):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_hidden)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── Base class with shared logic ───────────────────────────────


class _MicroLLMBase(nn.Module):
    def _init_weights(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_transformer(self, dim, n_layers, n_heads, ffn_hidden, vocab_size):
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ffn_hidden) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def _transformer_forward(self, x, idx_device):
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=idx_device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.output(x)

    def forward(self, idx, targets=None):
        x = self._embed(idx)
        logits = self._transformer_forward(x, idx.device)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Model A: HTP embedding ────────────────────────────────────


class MicroLLM_HTP(_MicroLLMBase):
    """HTP deterministic embedding + small learned embedding."""

    def __init__(self, vocab_size=600, dim=80, n_layers=2, n_heads=4,
                 ffn_hidden=176, max_seq_len=256, learned_emb_dim=24):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.htp = HTPWordEmbedding(dim=dim)
        self.learned_emb = nn.Embedding(vocab_size, learned_emb_dim)
        self.input_proj = nn.Linear(dim + learned_emb_dim, dim, bias=False)
        self._build_transformer(dim, n_layers, n_heads, ffn_hidden, vocab_size)
        self._init_weights()

    def _embed(self, idx):
        htp_emb = self.htp(idx)
        learned = self.learned_emb(idx)
        x = torch.cat([htp_emb, learned], dim=-1)
        return self.input_proj(x)


# ─── Model B: Standard learned embedding ────────────────────────


class MicroLLM_Standard(_MicroLLMBase):
    """Standard nn.Embedding (baseline)."""

    def __init__(self, vocab_size=600, dim=80, n_layers=2, n_heads=4,
                 ffn_hidden=176, max_seq_len=256):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, dim)
        self._build_transformer(dim, n_layers, n_heads, ffn_hidden, vocab_size)
        self._init_weights()

    def _embed(self, idx):
        return self.embedding(idx)
