import torch
import torch.nn as nn
import math


# Positional Encoding (VERY IMPORTANT for Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=12
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.positional_encoding = PositionalEncoding(d_model)

        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: [batch, context, features]

        x = self.input_projection(x)          # → [batch, context, d_model]
        x = self.positional_encoding(x)       # add time info
        x = self.transformer(x)               # transformer encoder
        x = x[:, -1, :]                       # last timestep
        out = self.fc(x)                      # → [batch, horizon]

        return out