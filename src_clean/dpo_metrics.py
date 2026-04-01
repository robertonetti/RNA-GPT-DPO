from __future__ import annotations

from typing import List

import torch


def _safe_trapezoid(x_values: List[float], y_values: List[float]) -> float:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return float("nan")
    area = 0.0
    for idx in range(1, len(x_values)):
        dx = float(x_values[idx]) - float(x_values[idx - 1])
        area += dx * (float(y_values[idx]) + float(y_values[idx - 1])) * 0.5
    return float(area)


def compute_auroc_from_good_bad_nll(good_nll: List[float], bad_nll: List[float]) -> float:
    labels: List[int] = []
    scores: List[float] = []

    for value in good_nll:
        numeric = float(value)
        if numeric == numeric:
            labels.append(1)
            scores.append(-numeric)
    for value in bad_nll:
        numeric = float(value)
        if numeric == numeric:
            labels.append(0)
            scores.append(-numeric)

    if len(labels) == 0 or len(labels) != len(scores):
        return float("nan")

    positives = sum(1 for label in labels if label == 1)
    negatives = sum(1 for label in labels if label == 0)
    if positives == 0 or negatives == 0:
        return float("nan")

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tpr: List[float] = [0.0]
    fpr: List[float] = [0.0]
    tp = 0
    fp = 0
    index = 0

    while index < len(ranked):
        score_value = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == score_value:
            if ranked[index][1] == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        tpr.append(tp / positives)
        fpr.append(fp / negatives)

    return _safe_trapezoid(fpr, tpr)


def get_logprobs(logits: torch.Tensor, labels: torch.Tensor, pad_token: int) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    mask = labels != pad_token
    return (token_logps * mask).sum(dim=-1)


def _paired_logprobs(
    model,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    pad_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x_w.size(0)
    x_cat = torch.cat([x_w, x_l], dim=0)
    y_cat = torch.cat([y_w, y_l], dim=0)
    logps = get_logprobs(model(x_cat)["logits"], y_cat, pad_token)
    return logps[:batch_size], logps[batch_size:]


def _paired_logprobs_separate(
    model,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    pad_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        get_logprobs(model(x_w)["logits"], y_w, pad_token),
        get_logprobs(model(x_l)["logits"], y_l, pad_token),
    )


def compute_preference_loss(
    model,
    ref_model,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    x_l: torch.Tensor,
    y_l: torch.Tensor,
    pad_token: int,
    beta: float,
    reint: bool,
    lambda_reint: float,
    ref_logps_w: torch.Tensor | None = None,
    ref_logps_l: torch.Tensor | None = None,
) -> torch.Tensor:
    if reint:
        if torch.is_grad_enabled():
            logps_w, logps_l = _paired_logprobs_separate(model, x_w, y_w, x_l, y_l, pad_token)
        else:
            logps_w, logps_l = _paired_logprobs(model, x_w, y_w, x_l, y_l, pad_token)
        reward_w = beta * logps_w
        reward_l = beta * lambda_reint * logps_l
        return -(reward_w - reward_l).mean()

    logps_w, logps_l = _paired_logprobs(model, x_w, y_w, x_l, y_l, pad_token)
    if ref_logps_w is None or ref_logps_l is None:
        with torch.no_grad():
            ref_logps_w, ref_logps_l = _paired_logprobs(
                ref_model, x_w, y_w, x_l, y_l, pad_token
            )
    reward_w = beta * (logps_w - ref_logps_w)
    reward_l = beta * (logps_l - ref_logps_l)
    return -torch.nn.functional.logsigmoid(reward_w - reward_l).mean()


def compute_batched_kl_loss(
    model,
    ref_model,
    x: torch.Tensor,
    y: torch.Tensor,
    pad_token: int,
    batch_size: int,
) -> torch.Tensor:
    total = torch.zeros((), device=x.device)
    total_tokens = torch.zeros((), device=x.device)

    for start in range(0, x.size(0), batch_size):
        x_batch = x[start : start + batch_size]
        y_batch = y[start : start + batch_size]

        student_log_probs = torch.log_softmax(model(x_batch)["logits"].float(), dim=-1)
        with torch.no_grad():
            teacher_logits = ref_model(x_batch)["logits"].float()
            teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()

        per_token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        mask = (y_batch != pad_token).to(per_token_kl.dtype)
        total = total + (per_token_kl * mask).sum()
        total_tokens = total_tokens + mask.sum()

    return total / total_tokens.clamp(min=1.0)


def compute_full_preference_loss_from_loader(
    model,
    ref_model,
    loader,
    pad_token: int,
    device: torch.device,
    beta: float,
    reint: bool,
    lambda_reint: float,
) -> float:
    total_loss = 0.0
    total_pairs = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                x_w, x_l, y_w, y_l, ref_logps_w, ref_logps_l = batch
                ref_logps_w = ref_logps_w.to(device)
                ref_logps_l = ref_logps_l.to(device)
            else:
                x_w, x_l, y_w, y_l = batch
                ref_logps_w = None
                ref_logps_l = None

            x_w = x_w.to(device)
            x_l = x_l.to(device)
            y_w = y_w.to(device)
            y_l = y_l.to(device)

            loss = compute_preference_loss(
                model,
                ref_model,
                x_w,
                y_w,
                x_l,
                y_l,
                pad_token=pad_token,
                beta=beta,
                reint=reint,
                lambda_reint=lambda_reint,
                ref_logps_w=ref_logps_w,
                ref_logps_l=ref_logps_l,
            )
            batch_pairs = x_w.size(0)
            total_loss += float(loss.item()) * batch_pairs
            total_pairs += batch_pairs

    return total_loss / max(total_pairs, 1)


def compute_preference_loss_from_batch_in_chunks(
    model,
    ref_model,
    batch: tuple,
    pad_token: int,
    device: torch.device,
    beta: float,
    reint: bool,
    lambda_reint: float,
    chunk_size: int,
) -> float:
    if len(batch) == 6:
        x_w, x_l, y_w, y_l, ref_logps_w, ref_logps_l = batch
    else:
        x_w, x_l, y_w, y_l = batch
        ref_logps_w = None
        ref_logps_l = None

    total_loss = 0.0
    total_pairs = 0

    with torch.no_grad():
        for start in range(0, x_w.size(0), chunk_size):
            end = start + chunk_size
            x_w_chunk = x_w[start:end].to(device)
            x_l_chunk = x_l[start:end].to(device)
            y_w_chunk = y_w[start:end].to(device)
            y_l_chunk = y_l[start:end].to(device)

            ref_logps_w_chunk = (
                ref_logps_w[start:end].to(device) if ref_logps_w is not None else None
            )
            ref_logps_l_chunk = (
                ref_logps_l[start:end].to(device) if ref_logps_l is not None else None
            )

            loss = compute_preference_loss(
                model,
                ref_model,
                x_w_chunk,
                y_w_chunk,
                x_l_chunk,
                y_l_chunk,
                pad_token=pad_token,
                beta=beta,
                reint=reint,
                lambda_reint=lambda_reint,
                ref_logps_w=ref_logps_w_chunk,
                ref_logps_l=ref_logps_l_chunk,
            )
            batch_pairs = x_w_chunk.size(0)
            total_loss += float(loss.item()) * batch_pairs
            total_pairs += batch_pairs

    return total_loss / max(total_pairs, 1)


def compute_sequence_logprobs(
    model,
    data: torch.Tensor,
    labels: torch.Tensor,
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    if len(data) == 0:
        return torch.empty(0, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x = data[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].to(device)
            outputs.append(get_logprobs(model(x)["logits"], y, pad_token).cpu())

    return torch.cat(outputs, dim=0)


def compute_sequence_nll(
    model,
    data: torch.Tensor,
    labels: torch.Tensor,
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    if len(data) == 0:
        return torch.empty(0, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x = data[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].to(device)
            logits = model(x)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logps = torch.gather(log_probs, dim=2, index=y.unsqueeze(2)).squeeze(2)
            mask = y != pad_token
            token_count = mask.sum(dim=1).clamp(min=1)
            outputs.append((-((token_logps * mask).sum(dim=1) / token_count)).cpu())

    return torch.cat(outputs, dim=0)


def compute_mean_token_likelihood(
    model,
    data: torch.Tensor,
    labels: torch.Tensor,
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> float:
    total_logp = 0.0
    total_tokens = 0
    if len(data) == 0:
        return float("nan")

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x = data[start : start + batch_size].to(device)
            y = labels[start : start + batch_size].to(device)
            logits = model(x)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logps = torch.gather(log_probs, dim=2, index=y.unsqueeze(2)).squeeze(2)
            mask = y != pad_token
            total_logp += (token_logps * mask).sum().item()
            total_tokens += mask.sum().item()

    mean_logp = total_logp / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(mean_logp)).item())


def attach_ref_logprobs(
    dataset: dict | None,
    ref_model,
    pad_token: int,
    device: torch.device,
    batch_size: int,
) -> None:
    if dataset is None:
        return
    dataset["ref_good_logps"] = compute_sequence_logprobs(
        ref_model,
        dataset["good_data"],
        dataset["good_labels"],
        pad_token,
        device,
        batch_size,
    )
    dataset["ref_bad_logps"] = compute_sequence_logprobs(
        ref_model,
        dataset["bad_data"],
        dataset["bad_labels"],
        pad_token,
        device,
        batch_size,
    )
