"""
The three architectures compared in Phase 2. See documentation.txt Part 6
for the full rationale behind each one and why the hyperparameters are
sized the way they are; the short version is repeated in each class
docstring below.

All three share the same call signature -- forward(x, pad_mask) -- where
x is (batch, T, F) and pad_mask is (batch, T) boolean with True meaning
"this position is padding, ignore it". Models A and C return just logits;
Model B returns (logits, attention_weights) so train.py/evaluate.py handle
both uniformly by checking `isinstance(out, tuple)`.
"""

import math

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """Model A -- baseline. Bidirectional LSTM, classification from the
    final timestep's hidden state. No temporal explanation output; exists
    to give the other two models a floor to beat and to show, in the
    paper's ablation table, what attention/self-attention actually buys
    you over a plain recurrent baseline.

    build_sequences.py always LEFT-pads (padding first, real data last,
    so every sequence's real data ends exactly at index T-1 regardless of
    how many real timesteps it has -- see SEQ_LEN handling in
    build_sequences.build_sequences). That means the correct "last real
    timestep" is always out[:, -1, :]; no masking/gather is needed or
    correct here. An earlier version of this method tried to gather
    index (n_real - 1), which assumes RIGHT-padding (real data first,
    padding after) -- for a heavily-padded sequence like a CISS event
    (often only 25-50 of 160 positions are real) that gather pulled the
    hidden state from deep inside the zero-padded region and produced
    near-random predictions. See documentation.txt Part 9.1.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, pad_mask=None):
        out, _ = self.lstm(x)  # (batch, T, hidden*2)
        out = out[:, -1, :]    # left-padded => real data always ends at T-1
        return self.classifier(self.dropout(out))


class LSTMAttentionClassifier(nn.Module):
    """Model B -- BiLSTM + additive (Bahdanau-style) attention. Adds one
    learned scalar score per timestep on top of the same LSTM backbone as
    Model A. The resulting attention_weights (batch, T) are a cheap,
    directly-interpretable temporal explanation -- which seconds before
    impact did the model weight most heavily -- and serve as a sanity
    check against the Transformer's self-attention in Model C.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_score = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, pad_mask=None):
        lstm_out, _ = self.lstm(x)  # (batch, T, hidden*2)
        scores = self.attn_score(lstm_out).squeeze(-1)  # (batch, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)
        context = (attn_weights.unsqueeze(-1) * lstm_out).sum(dim=1)
        logits = self.classifier(self.dropout(context))
        return logits, attn_weights


class PositionalEncoding(nn.Module):
    """Standard sinusoidal position encoding (Vaswani et al. 2017). A
    Transformer has no inherent notion of timestep order -- self-attention
    is permutation-invariant -- so position has to be injected explicitly.
    """

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerCrashClassifier(nn.Module):
    """Model C -- the primary/contribution model. A small Transformer
    encoder: every timestep can attend directly to every other timestep
    regardless of distance, which matters here because the critical
    pre-crash moment is not always in the final second -- SynSHRP2 windows
    run up to 20s and near-crash events in particular can show their most
    informative signal (a swerve, a late brake) well before impact, which
    an LSTM has to propagate step-by-step to "remember" that far.

    d_model=64 is deliberately small: the input has only 11 channels and
    the training pool is ~6,000 sequences (see documentation.txt Part 4.6)
    -- a 256- or 512-dim Transformer would have far more parameters than
    the data can constrain and would overfit long before it generalizes.
    nhead=4 divides 64 evenly; num_layers=3 is a standard depth for
    non-text Transformer encoders at this scale.
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3,
                 num_classes=5, dim_feedforward=256, dropout=0.1, max_len=200):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask=None):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        if pad_mask is not None:
            valid = (~pad_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return self.classifier(self.dropout(h))


def build_model(name, input_dim, num_classes=5):
    if name == "bilstm":
        return BiLSTMClassifier(input_dim, num_classes=num_classes)
    if name == "lstm_attn":
        return LSTMAttentionClassifier(input_dim, num_classes=num_classes)
    if name == "transformer":
        return TransformerCrashClassifier(input_dim, num_classes=num_classes)
    raise ValueError(f"unknown model name: {name}")
