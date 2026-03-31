from __future__ import annotations

import random
from typing import List

import torch
from torch.utils.data import DataLoader

from src_precomputed.dpo_data import _build_labels_from_inputs


def compute_pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation between two 1D tensors.

    Inputs:
    - x: Tensor of shape (N,).
    - y: Tensor of shape (N,).

    Output:
    - Pearson r as Python float.
    - Returns nan if inputs are empty, have different lengths, or zero variance.
    """
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return float("nan")

    x = x.float()
    y = y.float()

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum()))

    if denom.item() == 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum().item() / denom.item())


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


def _compute_paired_sequence_logprobs(
    model,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    pad_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward for winner+loser inputs and split sequence log-probs."""
    batch_size = x_w.size(0)
    x_cat = torch.cat([x_w, x_l], dim=0)
    y_cat = torch.cat([y_w, y_l], dim=0)
    logps_cat = get_logprobs(model(x_cat)["logits"], y_cat, pad_token)
    return logps_cat[:batch_size], logps_cat[batch_size:]


def _slice_data_batch(
    data: torch.Tensor | List[torch.Tensor],
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one contiguous dataset slice on the target device."""
    if isinstance(data, torch.Tensor):
        return data[start:end].to(device)
    return torch.stack(data[start:end], dim=0).to(device)


def _slice_optional_labels_batch(
    labels: torch.Tensor | List[torch.Tensor] | None,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a label slice if provided."""
    if labels is None:
        return None
    return _slice_data_batch(labels, start, end, device)


def _index_data_batch(
    data: torch.Tensor | List[torch.Tensor],
    indices: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Return an indexed mini-batch on the target device."""
    if isinstance(data, torch.Tensor):
        return data[indices].to(device)
    return torch.stack([data[i] for i in indices], dim=0).to(device)


def _index_optional_labels_batch(
    labels: torch.Tensor | List[torch.Tensor] | None,
    indices: List[int],
    device: torch.device,
) -> torch.Tensor | None:
    """Return indexed labels if provided."""
    if labels is None:
        return None
    return _index_data_batch(labels, indices, device)


def compute_dpo_loss_from_tensors(
    model,
    ref_model,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    pad_token: int,
    beta: float,
    ref_logps_w: torch.Tensor | None = None,
    ref_logps_l: torch.Tensor | None = None,
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
    logps_w, logps_l = _compute_paired_sequence_logprobs(
        model, x_w, y_w, x_l, y_l, pad_token
    )

    if ref_logps_w is None or ref_logps_l is None:
        with torch.no_grad():
            ref_logps_w, ref_logps_l = _compute_paired_sequence_logprobs(
                ref_model, x_w, y_w, x_l, y_l, pad_token
            )

    reward_w = beta * (logps_w - ref_logps_w)
    reward_l = beta * (logps_l - ref_logps_l)
    return -torch.nn.functional.logsigmoid(reward_w - reward_l).mean()


def compute_reint_loss_from_tensors(
    model,
    ref_model,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    pad_token: int,
    beta: float,
    lambda_reint: float,
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
    - lambda_reint: Reint reward scaling.
    Processing:
    - Computes sequence log-probabilities for winner and loser under policy and reference.
    - Builds rewards as beta * (policy_logp - reference_logp) for winner. And lambda_reint * (policy_logp - reference_logp) for loser.
    - Uses -(reward_w - reward_l), averaged over batch.

    Output:
    - Scalar tensor: mean DPO loss.
    """
    logps_w, logps_l = _compute_paired_sequence_logprobs(
        model, x_w, y_w, x_l, y_l, pad_token
    )

    # with torch.no_grad():
    #     ref_logits_w = ref_model(x_w)["logits"]
    #     ref_logits_l = ref_model(x_l)["logits"]
    #     ref_logps_w = get_logprobs(ref_logits_w, y_w, pad_token)
    #     ref_logps_l = get_logprobs(ref_logits_l, y_l, pad_token)

    reward_w = beta * (logps_w) # - ref_logps_w)
    reward_l = beta * lambda_reint *(logps_l) # - ref_logps_l)
    return -(reward_w - reward_l).mean()


def compute_token_level_kd_loss_from_tensors(
    model,
    ref_model,
    x: torch.Tensor,
    y: torch.Tensor,
    pad_token: int,
) -> torch.Tensor:
    """Compute token-level knowledge distillation loss on one batch.

    Inputs:
    - model: Trainable student model returning dict with key "logits".
    - ref_model: Frozen teacher/reference model with same output interface.
    - x: Input tokens, shape (B, T).
    - y: Target labels used only to mask pad positions, shape (B, T).
    - pad_token: Padding token id to ignore.

    Processing:
    - Computes student and teacher logits on the same inputs.
    - Converts them to log-probabilities/probabilities over the vocabulary.
    - Computes per-token ``D_KL(teacher || student)``.
    - Masks pad positions using ``y`` and averages over non-pad tokens.

    Output:
    - Scalar tensor: mean token-level KD loss.
    """
    logits = model(x)["logits"].float()
    student_log_probs = torch.log_softmax(logits, dim=-1) # shape (B, T, V)

    with torch.no_grad():
        ref_logits = ref_model(x)["logits"].float()
        teacher_log_probs = torch.log_softmax(ref_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp() # shape (B, T, V)

    per_token_kd = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1) # shape (B, T)
    mask = (y != pad_token).to(per_token_kd.dtype)

    return (per_token_kd * mask).sum() / mask.sum().clamp(min=1.0) # avoid division by zero if all tokens are pad, shape: scalar tensor


def compute_batched_token_level_kd_loss_from_tensors(
    model,
    ref_model,
    x: torch.Tensor,
    y: torch.Tensor,
    pad_token: int,
    batch_size: int,
) -> torch.Tensor:
    """Compute token-level KD loss over a tensor batch using smaller chunks.

    Inputs:
    - model: Trainable student model returning dict with key "logits".
    - ref_model: Frozen teacher/reference model with same output interface.
    - x: Input tokens, shape (B, T).
    - y: Target labels used only to mask pad positions, shape (B, T).
    - pad_token: Padding token id to ignore.
    - batch_size: Chunk size used to split ``x`` and ``y`` along batch dimension.

    Output:
    - Scalar tensor: mean token-level KD loss over all non-pad tokens in ``x``.
    """
    total_kd = torch.zeros((), device=x.device)
    total_tokens = torch.zeros((), device=x.device)

    for start in range(0, x.size(0), batch_size):
        end = start + batch_size
        x_batch = x[start:end]
        y_batch = y[start:end]
        kd_loss_batch = compute_token_level_kd_loss_from_tensors(
            model,
            ref_model,
            x_batch,
            y_batch,
            pad_token=pad_token,
        )
        valid_tokens_batch = (y_batch != pad_token).sum().to(kd_loss_batch.dtype)
        total_kd = total_kd + kd_loss_batch * valid_tokens_batch
        total_tokens = total_tokens + valid_tokens_batch

    return total_kd / total_tokens.clamp(min=1.0)


def compute_random_batch_nll(
    model,
    data_list: torch.Tensor | List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    n_batches: int,
    batch_size: int,
    labels: torch.Tensor | List[torch.Tensor] | None = None,
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
        x = _index_data_batch(data_list, idxs, device)
        y = _index_optional_labels_batch(labels, idxs, device)
        if y is None:
            y = _build_labels_from_inputs(x, pad_token)

        logits = model(x)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token = torch.gather(log_probs, 2, y.unsqueeze(2)).squeeze(2)

        mask = y != pad_token
        total_nll += -(per_token * mask).sum().item()
        total_tokens += mask.sum().item()

    return total_nll / max(total_tokens, 1)


def compute_sequence_logprobs(
    model,
    data_list: torch.Tensor | List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
    labels: torch.Tensor | List[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute one summed log-probability per sequence over a full dataset.

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
    seq_logps: List[torch.Tensor] = []
    if len(data_list) == 0:
        return torch.empty(0, dtype=torch.float32)

    with torch.inference_mode():
        for start in range(0, len(data_list), batch_size):
            x = _slice_data_batch(data_list, start, start + batch_size, device)
            y = _slice_optional_labels_batch(labels, start, start + batch_size, device)
            if y is None:
                y = _build_labels_from_inputs(x, pad_token)
            logits = model(x)["logits"]
            seq_logps.append(get_logprobs(logits, y, pad_token).detach().cpu())

    return torch.cat(seq_logps, dim=0)


def compute_full_dataset_nll(
    model,
    data_list: torch.Tensor | List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
    labels: torch.Tensor | List[torch.Tensor] | None = None,
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
        x = _slice_data_batch(data_list, start, start + batch_size, device)
        y = _slice_optional_labels_batch(labels, start, start + batch_size, device)
        if y is None:
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
    for batch in loader:
        if len(batch) >= 6:
            x_w, x_l, y_w, y_l, ref_logps_w, ref_logps_l = batch[:6]
            ref_logps_w = ref_logps_w.to(device)
            ref_logps_l = ref_logps_l.to(device)
        elif len(batch) >= 4:
            x_w, x_l, y_w, y_l = batch[:4]
            ref_logps_w = None
            ref_logps_l = None
        else:
            x_w, x_l = batch
            y_w = None
            y_l = None
            ref_logps_w = None
            ref_logps_l = None
        x_w = x_w.to(device)
        x_l = x_l.to(device)
        if y_w is None or y_l is None:
            y_w = _build_labels_from_inputs(x_w, pad_token)
            y_l = _build_labels_from_inputs(x_l, pad_token)
        else:
            y_w = y_w.to(device)
            y_l = y_l.to(device)

        logps_w, logps_l = _compute_paired_sequence_logprobs(
            model, x_w, y_w, x_l, y_l, pad_token
        )

        if ref_logps_w is None or ref_logps_l is None:
            ref_logps_w, ref_logps_l = _compute_paired_sequence_logprobs(
                ref_model, x_w, y_w, x_l, y_l, pad_token
            )

        reward_w = beta * (logps_w - ref_logps_w)
        reward_l = beta * (logps_l - ref_logps_l)
        loss_per_pair = -torch.nn.functional.logsigmoid(reward_w - reward_l)

        total_loss += loss_per_pair.sum().item()
        total_pairs += loss_per_pair.numel()

    return total_loss / max(total_pairs, 1)


def compute_full_dataset_reint_loss_from_loader(
    model,
    ref_model,
    loader: DataLoader,
    pad_token: int,
    device: torch.device,
    beta: float,
    lambda_reint: float,
) -> float:
    """Compute mean DPO loss over all pairs from a DataLoader.

    Inputs:
    - model: Trainable policy model.
    - ref_model: Frozen reference model.
    - loader: Yields tuples (x_w, x_l), each tensor shape (B, T).
    - pad_token: Padding token id.
    - device: Target device.
    - beta: DPO scaling factor.
    - lambda_reint: Reint reward scaling factor.

    Output:
    - Python float: mean DPO loss over all pairs.
    """
    total_loss, total_pairs = 0.0, 0
    for batch in loader:
        if len(batch) >= 4:
            x_w, x_l, y_w, y_l = batch[:4]
        else:
            x_w, x_l = batch[:2]
            y_w = None
            y_l = None
        x_w = x_w.to(device)
        x_l = x_l.to(device)
        if y_w is None or y_l is None:
            y_w = _build_labels_from_inputs(x_w, pad_token)
            y_l = _build_labels_from_inputs(x_l, pad_token)
        else:
            y_w = y_w.to(device)
            y_l = y_l.to(device)

        logps_w, logps_l = _compute_paired_sequence_logprobs(
            model, x_w, y_w, x_l, y_l, pad_token
        )

        # ref_logits_w = ref_model(x_w)["logits"]
        # ref_logits_l = ref_model(x_l)["logits"]
        # ref_logps_w = get_logprobs(ref_logits_w, y_w, pad_token)
        # ref_logps_l = get_logprobs(ref_logits_l, y_l, pad_token)

        reward_w = beta * (logps_w) # - ref_logps_w)
        reward_l = beta * lambda_reint * (logps_l) # - ref_logps_l)
        loss_per_pair = -(reward_w - reward_l)

        total_loss += loss_per_pair.sum().item()
        total_pairs += loss_per_pair.numel()

    return total_loss / max(total_pairs, 1)


def compute_mean_token_likelihood(
    model,
    data_list: torch.Tensor | List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
    labels: torch.Tensor | List[torch.Tensor] | None = None,
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
        x = _slice_data_batch(data_list, start, start + batch_size, device)
        y = _slice_optional_labels_batch(labels, start, start + batch_size, device)
        if y is None:
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
    data_list: torch.Tensor | List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
    labels: torch.Tensor | List[torch.Tensor] | None = None,
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
        x = _slice_data_batch(data_list, start, start + batch_size, device)
        y = _slice_optional_labels_batch(labels, start, start + batch_size, device)
        if y is None:
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
    good_data: torch.Tensor | List[torch.Tensor],
    bad_data: torch.Tensor | List[torch.Tensor],
    pad_token: int,
    device: torch.device,
    batch_size: int,
    good_labels: torch.Tensor | List[torch.Tensor] | None = None,
    bad_labels: torch.Tensor | List[torch.Tensor] | None = None,
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
    good_nll = compute_sequence_nll(
        model, good_data, pad_token, device, batch_size=batch_size, labels=good_labels
    )
    bad_nll = compute_sequence_nll(
        model, bad_data, pad_token, device, batch_size=batch_size, labels=bad_labels
    )

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
        pearson_r = compute_pearson_correlation(nll_values, labels)

    return pearson_r, good_nll.tolist(), bad_nll.tolist()
