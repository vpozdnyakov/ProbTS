import math
import torch
import torch.nn as nn
from probts.model.forecaster import Forecaster
import torch.nn.functional as F
from probts.utils import repeat
from typing import Literal, Union
from probts.data.data_utils.data_scaler import (
    StandardScaler,
    TemporalScaler,
    BinScaler,
    BinaryQuantizer,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class TimeSeriesTransformerDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, num_layers, dim_feedforward=128, dropout=0.1, max_len=500
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    def generate_causal_mask(self, size):
        """
        Creates a mask to prevent attention to future positions.
        Shape: (size, size) where mask[i, j] == -inf if j > i
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x):
        """
        x: Tensor of shape (b, c, d)
        returns: Tensor of shape (b, c, d) where each vector is a one-step forecast
        """
        b, c, d = x.shape
        x_pos = self.pos_encoder(x)
        tgt = x_pos
        causal_mask = self.generate_causal_mask(c).to(x.device)
        out = self.transformer_decoder(tgt=tgt, memory=x_pos, tgt_mask=causal_mask)
        return out 


def sliding_window_batch(x, L, H):
    if len(x.shape) == 4:
        x = x.squeeze()
    """
    x: Tensor of shape (B, L+H, C)
    Returns: Tensor of shape (B, H, L, C)
    """
    B, total_len, C = x.shape
    assert total_len >= L + H, "Not enough sequence length for given L and H"

    windows = [
        x[:, h : h + L, :].unsqueeze(1) for h in range(H)
    ]  # list of (B, 1, L, C)
    return torch.cat(windows, dim=1)  # (B, H, L, C)


def get_sequence_from_prob(p: torch.Tensor, is_sample: bool, eps: float = 1e-6):
    """
    p: Tensor of shape (B, D) with probabilities
    Returns:
        best_sequences: Tensor of shape (B, D) with the most probable [1...1, 0...0] sequence
        best_probs: Tensor of shape (B,) with normalized probability of the best sequence
    """
    B, D = p.shape

    # Clamp p to avoid log(0) or log(1) instability
    p_clamped = p.clamp(min=eps, max=1 - eps)

    # Use log domain to compute cumulative products
    log_p = torch.log(p_clamped)
    log_1_minus_p = torch.log(1 - p_clamped)

    log_success = torch.cumsum(log_p, dim=1)  # shape (B, D)
    log_fail = torch.cumsum(log_1_minus_p.flip(dims=[1]), dim=1).flip(
        dims=[1]
    )  # shape (B, D)

    # Pad with log(1) = 0 to align indexing
    zero = torch.zeros((B, 1), dtype=p.dtype, device=p.device)
    log_success = torch.cat([zero, log_success], dim=1)  # shape (B, D+1)
    log_fail = torch.cat([log_fail, zero], dim=1)  # shape (B, D+1)

    # Sum log-probs for each possible cutoff (index k: first 0 after all 1s)
    log_probs = log_success + log_fail  # shape (B, D+1)
    log_probs_max = torch.max(log_probs, dim=1, keepdim=True)[0]
    probs_normalized = torch.exp(log_probs - log_probs_max)
    probs_normalized = probs_normalized / probs_normalized.sum(dim=1, keepdim=True)

    # Sample or take the most probable index
    if is_sample:
        k = torch.multinomial(probs_normalized, num_samples=1)
    else:
        k = torch.argmax(probs_normalized, dim=1, keepdim=True)

    # Create the monotonic sequence [1,...,1,0,...,0]
    arange = torch.arange(D, device=p.device).unsqueeze(0)
    best_sequences = (arange < k).to(p.dtype)  # shape (B, D)

    best_probs = torch.gather(probs_normalized, dim=1, index=k).squeeze(1)

    return best_sequences, best_probs


class BinFormer(Forecaster):
    def __init__(
        self,
        context_length: int,
        is_prob_forecast: bool,
        num_bins: int,
        min_bin_value=-10.0,
        max_bin_value=10.0,
        dropout=0.2,
        n_heads=4,
        n_layers=3,
        f_hidden_size=40,
        attn_dropout=0.,
        scaler_type: Union[Literal["standard", "temporal", "None"], None] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the model with parameters.
        """
        super().__init__(context_length=context_length, **kwargs)
        # Initialize model parameters here
        self.context_length = context_length
        self.num_bins = num_bins
        self.is_prob_forecast = is_prob_forecast
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.f_hidden_size = f_hidden_size
        self.attn_dropout = attn_dropout

        if scaler_type is None:
            self.scalers = None
        elif scaler_type == "standard":
            self.scalers = [
                BinScaler(
                    StandardScaler(var_specific=True),
                    BinaryQuantizer(
                        num_bins=num_bins, min_val=min_bin_value, max_val=max_bin_value
                    ),
                )
                for _ in range(self.target_dim)
            ]
        elif scaler_type == "temporal":
            self.scalers = [
                BinScaler(
                    TemporalScaler(time_first=False),
                    BinaryQuantizer(
                        num_bins=num_bins, min_val=min_bin_value, max_val=max_bin_value
                    ),
                )
                for _ in range(self.target_dim)
            ]
        else:
            assert False, f"The scaler type {scaler_type} is not supported"
        self.dropout = nn.Dropout(dropout)

        layers = []
        for i in range(kwargs["target_dim"]):
            layers.append(
                TimeSeriesTransformerDecoder(
                    d_model=num_bins,
                    nhead=n_heads,
                    num_layers=n_layers,
                    dim_feedforward=f_hidden_size,
                    dropout=attn_dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x, layer_id=0):
        if len(x.shape) == 4:
            x = x.squeeze()
        x = x.float()
        # x: (batch_size, context_length, num_bins)
        out = self.layers[layer_id](x)
        return out

    def loss(self, batch_data):
        """
        Compute the loss for the given batch data.

        Parameters:
        batch_data [dict]: Dictionary containing input data and possibly target data.

        Returns:
        Tensor: Computed loss.
        """

        inputs = self.get_inputs(batch_data, "all")
        losses = []
        if len(inputs.shape) == 4:
            iter_c = -2
        else:
            iter_c = -1
        for c in range(inputs.shape[iter_c]):
            if self.scalers is not None:
                self.scalers[c].fit(
                    inputs[:, :, c : c + 1].reshape(-1)[: -self.prediction_length]
                )
                c_inputs = self.scalers[c].transform(inputs[:, :, c : c + 1])
            else:
                c_inputs = inputs[:, :, c : c + 1]
            target = c_inputs[:, -self.prediction_length :, :]
            outputs = self(c_inputs)[:, -self.prediction_length:, :]
            
            c_loss = F.binary_cross_entropy_with_logits(input=outputs, target=target)
            losses.append(c_loss)
        loss = torch.stack(losses).mean()
        return loss

    def forecast(self, batch_data, num_samples=None):
        do_sample = (
            num_samples is not None and num_samples > 1 and self.is_prob_forecast
        )
        inputs = self.get_inputs(batch_data, "encode")
        forecasts_list = []
        for c in range(inputs.shape[2]):
            if self.scalers is not None:
                self.scalers[c].fit(inputs[:, :, c : c + 1].reshape(-1))
                c_inputs = self.scalers[c].transform(inputs[:, :, c : c + 1])
            else:
                c_inputs = inputs[:, :, c : c + 1]

            if do_sample:
                c_inputs = repeat(
                    c_inputs.unsqueeze(1), num_samples, 1
                )  # (B, NS, T, D)
                batch_size = c_inputs.shape[0]
                c_inputs = c_inputs.view(-1, *c_inputs.shape[2:])
            current_context = c_inputs.clone()
            c_forecasts = []
            for _ in range(self.prediction_length):
                pred = F.sigmoid(self(current_context))  # (B, D)
                pred = pred[:, -1, :]
                pred, _ = get_sequence_from_prob(pred, do_sample)
                pred = pred.int()
                c_forecasts.append(pred.unsqueeze(1))  # (B, 1, D)
                next_input = pred.unsqueeze(1)

                if len(current_context.shape) == 4:
                    next_input = next_input.unsqueeze(1)
                current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

            c_forecasts = torch.cat(c_forecasts, dim=1)
            if self.scalers is not None:
                c_forecasts = self.scalers[c].inverse_transform(c_forecasts)
            if do_sample:
                c_forecasts = c_forecasts.view(
                    batch_size, num_samples, *c_forecasts.shape[1:]
                )
            else:
                c_forecasts = c_forecasts.unsqueeze(1)  # (B, 1,  T, D)
            if inputs.shape[2] > 1:
                c_forecasts = c_forecasts.unsqueeze(-2)  # (B, 1, T, D, num_bins)
            forecasts_list.append(c_forecasts)
        forecasts = torch.concat(forecasts_list, dim=-2)  # was 2
        return forecasts
