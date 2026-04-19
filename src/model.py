from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f'Sequence length {seq_len} exceeds max positional size {self.max_len}')
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


class TinyTransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = 'relu',
        position_type: str = 'sinusoidal',
        tie_weights: bool = True,
        max_position_embeddings: int = 128,
    ) -> None:
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)

        if position_type == 'sinusoidal':
            self.src_positional = SinusoidalPositionalEncoding(d_model, dropout=dropout, max_len=max_position_embeddings)
            self.tgt_positional = SinusoidalPositionalEncoding(d_model, dropout=dropout, max_len=max_position_embeddings)
        elif position_type == 'learned':
            self.src_positional = LearnedPositionalEncoding(d_model, dropout=dropout, max_len=max_position_embeddings)
            self.tgt_positional = LearnedPositionalEncoding(d_model, dropout=dropout, max_len=max_position_embeddings)
        else:
            raise ValueError(f'Unsupported position_type: {position_type}')

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        if tie_weights:
            self.output_projection.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        with torch.no_grad():
            self.src_embedding.weight[self.src_pad_id].fill_(0.0)
            self.tgt_embedding.weight[self.tgt_pad_id].fill_(0.0)

    def generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = src.eq(self.src_pad_id)
        src_embeddings = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embeddings = self.src_positional(src_embeddings)
        memory = self.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_key_padding_mask = tgt_in.eq(self.tgt_pad_id)
        tgt_embeddings = self.tgt_embedding(tgt_in) * math.sqrt(self.d_model)
        tgt_embeddings = self.tgt_positional(tgt_embeddings)
        tgt_mask = self.generate_square_subsequent_mask(tgt_in.size(1), tgt_in.device)
        decoded = self.decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_projection(decoded)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_key_padding_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_key_padding_mask)
