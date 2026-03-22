from __future__ import annotations

import random
from typing import List

import torch
from torch.utils.data import DataLoader

from src.dpo_data import _build_labels_from_inputs


def get_logprobs(logits: torch.Tensor, labels: torch.Tensor, pad_token: int) -> torch.Tensor:
    """Compute per-sequence log-probability from token logits and labels.

    Inputs:
    - logits: Tensor of shape (B, T, V).
      B=batch size, T=sequence length, V=vocabulary size.
    - labels: Target token ids, shape (B, T).
    - pad_token: Padding token id to ignore.

    Processing:
    - Applies log_softmax over vocabulary dimension.
    - Gathers log-probability of each target token.
    - Masks pad positions and sums over sequence length.

    Output:
    - Tensor of shape (B,), one summed log-probability per sequence.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    mask = labels != pad_token
    return (per_token_logps * mask).sum(-1)


def compute_dpo_loss_from_tensors(
    model,
    ref_model,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    pad_token: int,
    beta: float,
) -> torch.Tensor:
    """Compute Direct Preference Optimization loss for one winner/loser batch.

    Inputs:
    - model: Trainable policy model returning dict with key "logits".
    - ref_model: Frozen reference model with same output interface.
    - x_w: Winner inputs, shape (B, T).
    - y_w: Winner labels, shape (B, T).
    - x_l: Loser inputs, shape (B, T).
    - y_l: Loser labels, shape (B, T).
    - pad_token: Padding token id.
    - beta: DPO reward scaling.

    Processing:
    - Computes sequence log-probabilities for winner and loser under policy and reference.
    - Builds rewards as beta * (policy_logp - reference_logp).
    - Uses -logsigmoid(reward_w - reward_l), averaged over batch.

    Output:
    - Scalar tensor: mean DPO loss.
    """
    logits_w = model(x_w)["logits"]
    logits_l = model(x_l)["logits"]
    logps_w = get_logprobs(logits_w, y_w, pad_token)
    logps_l = get_logprobs(logits_l, y_l, pad_token)

    with torch.no_grad():
        ref_logits_w = ref_model(x_w)["logits"]
        ref_logits_l = ref_model(x_l)["logits"]
        ref_logps_w = get_logprobs(ref_logits_w, y_w, pad_token)
        ref_logps_l = get_logprobs(ref_logits_l, y_l, pad_token)

    reward_w = beta * (logps_w - ref_logps_w)
    reward_l = beta * (logps_l - ref_logps_l)
    return -torch.nn.functional.logsigmoid(reward_w - reward_l).mean()


def compute_random_batch_nll(
    model,
    data_list: List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    n_batches: int,
    batch_size: int,
) -> float:
    """Estimate mean token NLL from random mini-batches sampled with replacement.

    Inputs:
    - model: Model returning logits of shape (B, T, V).
    - data_list: List of encoded sequences, each shape (T,).
    - pad_token: Padding token id.
    - device: Target device.
    - n_batches: Number of sampled batches.
    - batch_size: Number of samples per batch.

    Output:
    - Python float: aggregated token NLL over sampled batches.
    """
    total_nll, total_tokens = 0.0, 0
    for _ in range(n_batches):
        idxs = random.choices(range(len(data_list)), k=batch_size)
        x = torch.stack([data_list[i] for i in idxs], dim=0).to(device)
        y = _build_labels_from_inputs(x, pad_token)

        logits = model(x)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token = torch.gather(log_probs, 2, y.unsqueeze(2)).squeeze(2)

        mask = y != pad_token
        total_nll += -(per_token * mask).sum().item()
        total_tokens += mask.sum().item()

    return total_nll / max(total_tokens, 1)


def compute_full_dataset_nll(
    model,
    data_list: List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> float:
    """Compute exact mean token NLL over a full dataset.

    Inputs:
    - model: Model returning logits of shape (B, T, V).
    - data_list: List of encoded sequences, each shape (T,).
    - pad_token: Padding token id.
    - device: Target device.
    - batch_size: Chunk size for iteration.

    Output:
    - Python float: exact token NLL for the full dataset.
    - Returns nan if data_list is empty.
    """
    total_nll, total_tokens = 0.0, 0
    if len(data_list) == 0:
        return float("nan")

    for start in range(0, len(data_list), batch_size):
        chunk = data_list[start : start + batch_size]
        x = torch.stack(chunk, dim=0).to(device)
        y = _build_labels_from_inputs(x, pad_token)

        logits = model(x)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token = torch.gather(log_probs, 2, y.unsqueeze(2)).squeeze(2)

        mask = y != pad_token
        total_nll += -(per_token * mask).sum().item()
        total_tokens += mask.sum().item()

    return total_nll / max(total_tokens, 1)


def compute_full_dataset_dpo_loss_from_loader(
    model,
    ref_model,
    loader: DataLoader,
    pad_token: int,
    device: torch.device,
    beta: float,
) -> float:
    """Compute mean DPO loss over all pairs from a DataLoader.

    Inputs:
    - model: Trainable policy model.
    - ref_model: Frozen reference model.
    - loader: Yields tuples (x_w, x_l), each tensor shape (B, T).
    - pad_token: Padding token id.
    - device: Target device.
    - beta: DPO scaling factor.

    Output:
    - Python float: mean DPO loss over all pairs.
    """
    total_loss, total_pairs = 0.0, 0
    for x_w, x_l in loader:
        x_w = x_w.to(device)
        x_l = x_l.to(device)
        y_w = _build_labels_from_inputs(x_w, pad_token)
        y_l = _build_labels_from_inputs(x_l, pad_token)

        logits_w = model(x_w)["logits"]
        logits_l = model(x_l)["logits"]
        logps_w = get_logprobs(logits_w, y_w, pad_token)
        logps_l = get_logprobs(logits_l, y_l, pad_token)

        ref_logits_w = ref_model(x_w)["logits"]
        ref_logits_l = ref_model(x_l)["logits"]
        ref_logps_w = get_logprobs(ref_logits_w, y_w, pad_token)
        ref_logps_l = get_logprobs(ref_logits_l, y_l, pad_token)

        reward_w = beta * (logps_w - ref_logps_w)
        reward_l = beta * (logps_l - ref_logps_l)
        loss_per_pair = -torch.nn.functional.logsigmoid(reward_w - reward_l)

        total_loss += loss_per_pair.sum().item()
        total_pairs += loss_per_pair.numel()

    return total_loss / max(total_pairs, 1)


def compute_mean_token_likelihood(
    model,
    data_list: List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> float:
    """Compute mean token likelihood by exponentiating mean log-likelihood.

    Inputs:
    - model: Model returning logits of shape (B, T, V).
    - data_list: List of encoded sequences, each shape (T,).
    - pad_token: Padding token id.
    - device: Target device.
    - batch_size: Chunk size for iteration.

    Output:
    - Python float: mean token likelihood.
    - Returns nan if input list is empty.
    """
    total_logp, total_tokens = 0.0, 0
    if len(data_list) == 0:
        return float("nan")

    for start in range(0, len(data_list), batch_size):
        chunk = data_list[start : start + batch_size]
        x = torch.stack(chunk, dim=0).to(device)
        y = _build_labels_from_inputs(x, pad_token)

        logits = model(x)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token_logp = torch.gather(log_probs, 2, y.unsqueeze(2)).squeeze(2)
        mask = y != pad_token

        total_logp += (per_token_logp * mask).sum().item()
        total_tokens += mask.sum().item()

    mean_log_likelihood = total_logp / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(mean_log_likelihood)).item())


def compute_sequence_nll(
    model,
    data_list: List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Compute one normalized NLL value per sequence.

    Inputs:
    - model: Model returning logits of shape (B, T, V).
    - data_list: List of encoded sequences, each shape (T,).
    - pad_token: Padding token id.
    - device: Target device.
    - batch_size: Chunk size for iteration.

    Output:
    - Tensor of shape (N,), where N=len(data_list).
    - Returns empty tensor if input list is empty.
    """
    seq_nlls: List[torch.Tensor] = []
    if len(data_list) == 0:
        return torch.empty(0, dtype=torch.float32)

    for start in range(0, len(data_list), batch_size):
        chunk = data_list[start : start + batch_size]
        x = torch.stack(chunk, dim=0).to(device)
        y = _build_labels_from_inputs(x, pad_token)

        logits = model(x)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token = torch.gather(log_probs, 2, y.unsqueeze(2)).squeeze(2)

        mask = y != pad_token
        token_counts = mask.sum(dim=1).clamp(min=1)
        seq_nll = -((per_token * mask).sum(dim=1) / token_counts)
        seq_nlls.append(seq_nll.detach().cpu())

    return torch.cat(seq_nlls, dim=0)


def compute_val_separation_correlation(
    model,
    good_data: List[torch.Tensor],
    bad_data: List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, List[float], List[float]]:
    """Compute Pearson correlation between sequence NLL and binary class labels.

    Inputs:
    - model: Model used to score sequences.
    - good_data: List of good encoded sequences, each shape (T,).
    - bad_data: List of bad encoded sequences, each shape (T,).
    - pad_token: Padding token id.
    - device: Target device.
    - batch_size: Chunk size for scoring.

    Processing:
    - Computes per-sequence NLL for good and bad groups.
    - Builds labels: good=0, bad=1.
    - Computes Pearson r between NLL values and labels.

    Output:
    - Tuple (pearson_r, good_nll_list, bad_nll_list).
    - good_nll_list length equals len(good_data).
    - bad_nll_list length equals len(bad_data).
    - pearson_r is nan if any group is empty or denominator is zero.
    """
    good_nll = compute_sequence_nll(model, good_data, pad_token, device, batch_size=batch_size)
    bad_nll = compute_sequence_nll(model, bad_data, pad_token, device, batch_size=batch_size)

    if good_nll.numel() == 0 or bad_nll.numel() == 0:
        pearson_r = float("nan")
    else:
        nll_values = torch.cat([good_nll.float(), bad_nll.float()], dim=0)
        labels = torch.cat(
            [
                torch.zeros(good_nll.numel(), dtype=torch.float32),
                torch.ones(bad_nll.numel(), dtype=torch.float32),
            ],
            dim=0,
        )

        nll_centered = nll_values - nll_values.mean()
        labels_centered = labels - labels.mean()
        denom = torch.sqrt((nll_centered.pow(2).sum()) * (labels_centered.pow(2).sum()))

        if denom.item() == 0.0:
            pearson_r = float("nan")
        else:
            pearson_r = float((nll_centered * labels_centered).sum().item() / denom.item())

    return pearson_r, good_nll.tolist(), bad_nll.tolist()
