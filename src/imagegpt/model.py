import math

import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dims))
        self.bias = nn.Parameter(torch.ones(dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_size)
        # keep dimensions for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # normalization: norm = (x - mean) / (std + eps)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))


class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.sos_token = nn.Parameter(torch.zeros(d_model))
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def add_sos_token(self, x: torch.Tensor) -> torch.Tensor:
        # pop the last token and add the sos token in the beginnig
        emb = self.embeddings(x)  # (batch_size, seq_len, d_model)
        sos = self.sos_token.view(1, 1, -1).expand(x.shape[0], 1, -1)  # (batch_size, 1, d_model)
        return torch.cat((sos, emb[:, :-1]), dim=1)  # (bs, deq_len, d_model)

    def add_position_embedding(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, seq_len, d_model)
        positions = torch.arange(x.shape[1]).to(x.device)
        pos_embs = self.pos_embeddings(positions).unsqueeze(0)
        return x + pos_embs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_sos_token(x)
        x = self.add_position_embedding(x)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        h: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> tuple[torch.Tensor, ...]:
        d_k = query.shape[-1]
        # (bs, h, seq_len, d_k) @ (bs, h, d_k, seq_len) -> (bs, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        # combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
