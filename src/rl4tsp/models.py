from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4tsp.tsp import compute_tour_length


def _validated_fixed_start(
    fixed_start: torch.Tensor | None,
    *,
    batch_size: int,
    n_cities: int,
    device: torch.device,
) -> torch.Tensor | None:
    if fixed_start is None:
        return None
    if fixed_start.shape != (batch_size,):
        raise ValueError(f"fixed_start must have shape ({batch_size},)")
    fixed_start = fixed_start.to(device).long()
    if torch.any(fixed_start < 0).item() or torch.any(fixed_start >= n_cities).item():
        raise ValueError(f"fixed_start entries must be in [0, {n_cities - 1}]")
    return fixed_start


class PointerNet(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(input_dim + 2 * hidden_dim, 2 * hidden_dim)
        self.start_input = nn.Parameter(torch.zeros(input_dim + 2 * hidden_dim))
        self.w1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=False)
        self.v = nn.Linear(2 * hidden_dim, 1, bias=False)

    def forward(
        self,
        coords: torch.Tensor,
        decode_mode: str = "sample",
        fixed_start: torch.Tensor | None = None,
        return_probabilities: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if decode_mode not in {"sample", "greedy"}:
            raise ValueError("decode_mode must be 'sample' or 'greedy'")
        batch_size, n_cities, _ = coords.shape
        fixed_start = _validated_fixed_start(
            fixed_start,
            batch_size=batch_size,
            n_cities=n_cities,
            device=coords.device,
        )
        batch_index = torch.arange(batch_size, device=coords.device)
        enc_out, (h_enc, c_enc) = self.encoder(coords)
        h_dec = torch.cat([h_enc[-2], h_enc[-1]], dim=-1)
        c_dec = torch.cat([c_enc[-2], c_enc[-1]], dim=-1)
        mask = torch.ones(batch_size, n_cities, device=coords.device, dtype=torch.bool)
        previous_idx = None
        actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []

        for step in range(n_cities):
            if previous_idx is None:
                dec_input = self.start_input.unsqueeze(0).expand(batch_size, -1)
            else:
                dec_input = torch.cat([coords[batch_index, previous_idx], enc_out[batch_index, previous_idx]], dim=-1)
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            scores = self.v(torch.tanh(self.w1(enc_out) + self.w2(h_dec).unsqueeze(1))).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            probs = F.softmax(scores, dim=-1)
            if return_probabilities:
                probabilities.append(probs.detach().cpu())

            if step == 0 and fixed_start is not None:
                action = fixed_start
            elif decode_mode == "greedy":
                action = torch.argmax(probs, dim=-1)
            else:
                distribution = torch.distributions.Categorical(probs)
                action = distribution.sample()
                log_probs.append(distribution.log_prob(action))
            actions.append(action)
            mask[batch_index, action] = False
            previous_idx = action

        action_tensor = torch.stack(actions, dim=1)
        lengths = compute_tour_length(coords, action_tensor)
        log_prob_tensor = torch.stack(log_probs, dim=1) if log_probs else None
        if return_probabilities:
            return action_tensor, lengths, log_prob_tensor, probabilities
        return action_tensor, lengths, log_prob_tensor


class AttentionModel(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, num_layers: int = 3, tanh_clipping: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.tanh_clipping = tanh_clipping
        self.input_proj = nn.Linear(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.start_token = nn.Parameter(torch.randn(1, d_model))

    def forward(
        self,
        coords: torch.Tensor,
        decode_mode: str = "sample",
        fixed_start: torch.Tensor | None = None,
        return_probabilities: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if decode_mode not in {"sample", "greedy"}:
            raise ValueError("decode_mode must be 'sample' or 'greedy'")
        batch_size, n_cities, _ = coords.shape
        fixed_start = _validated_fixed_start(
            fixed_start,
            batch_size=batch_size,
            n_cities=n_cities,
            device=coords.device,
        )

        batch_index = torch.arange(batch_size, device=coords.device)
        node_emb = self.encoder(self.input_proj(coords))
        keys = self.w_k(node_emb)
        mask = torch.ones(batch_size, n_cities, device=coords.device, dtype=torch.bool)
        current_idx: torch.Tensor | None = None
        actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []

        for step in range(n_cities):
            if current_idx is None:
                current_emb = self.start_token.expand(batch_size, -1)
            else:
                current_emb = node_emb[batch_index, current_idx]
            query = self.w_q(current_emb).unsqueeze(1)
            scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
            if self.tanh_clipping > 0:
                scores = self.tanh_clipping * torch.tanh(scores)
            scores = scores.masked_fill(~mask, -1e9)
            probs = F.softmax(scores, dim=-1)
            if return_probabilities:
                probabilities.append(probs.detach().cpu())

            if step == 0 and fixed_start is not None:
                action = fixed_start
            elif decode_mode == "greedy":
                action = torch.argmax(probs, dim=-1)
            else:
                distribution = torch.distributions.Categorical(probs)
                action = distribution.sample()
                log_probs.append(distribution.log_prob(action))
            actions.append(action)
            mask[batch_index, action] = False
            current_idx = action

        action_tensor = torch.stack(actions, dim=1)
        lengths = compute_tour_length(coords, action_tensor)
        log_prob_tensor = torch.stack(log_probs, dim=1) if log_probs else None
        if return_probabilities:
            return action_tensor, lengths, log_prob_tensor, probabilities
        return action_tensor, lengths, log_prob_tensor
