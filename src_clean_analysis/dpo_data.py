from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from torch.utils.data import Dataset


def resolve_path(project_dir: Path, path_value: str | None) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def has_path(path_value: str | None) -> bool:
    return path_value is not None and str(path_value).strip() != ""


def read_fasta(path: Path) -> List[str]:
    sequences: List[str] = []
    current: List[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            continue
        current.append(line)

    if current:
        sequences.append("".join(current))
    return sequences


def build_tokenizer(dn_path: Path) -> tuple[dict[str, int], int]:
    text = dn_path.read_text(encoding="utf-8").splitlines()[1::2]
    chars = sorted(set("\n".join(text)))
    pad_symbol = "?"
    if pad_symbol not in chars:
        chars.append(pad_symbol)
    stoi = {char: idx for idx, char in enumerate(chars)}
    return stoi, stoi[pad_symbol]


def pad_encode(
    sequence: str,
    block_size: int,
    pad_token: int,
    encode_fn: Callable[[str], List[int]],
) -> torch.Tensor:
    ids = encode_fn(f"\n{sequence}\n")
    if len(ids) < block_size:
        ids = ids + [pad_token] * (block_size - len(ids))
    else:
        ids = ids[:block_size]
    return torch.tensor(ids, dtype=torch.long)


def build_labels_from_inputs(x: torch.Tensor, pad_token: int) -> torch.Tensor:
    pad_column = torch.full((x.size(0), 1), pad_token, dtype=x.dtype, device=x.device)
    return torch.cat([x[:, 1:], pad_column], dim=1)


def encode_sequences(
    sequences: List[str],
    block_size: int,
    pad_token: int,
    encode_fn: Callable[[str], List[int]],
) -> torch.Tensor:
    if not sequences:
        return torch.empty((0, block_size), dtype=torch.long)
    return torch.stack(
        [pad_encode(seq, block_size, pad_token, encode_fn) for seq in sequences],
        dim=0,
    )


def load_group_dataset(
    good_fasta_path: Path,
    bad_fasta_path: Path,
    block_size: int,
    pad_token: int,
    encode_fn: Callable[[str], List[int]],
) -> Dict[str, torch.Tensor]:
    good_data = encode_sequences(read_fasta(good_fasta_path), block_size, pad_token, encode_fn)
    bad_data = encode_sequences(read_fasta(bad_fasta_path), block_size, pad_token, encode_fn)
    return {
        "good_data": good_data,
        "bad_data": bad_data,
        "good_labels": build_labels_from_inputs(good_data, pad_token),
        "bad_labels": build_labels_from_inputs(bad_data, pad_token),
    }


def load_preference_dataset(
    good_fasta_path: Path,
    bad_fasta_path: Path,
    csv_path: Path,
    block_size: int,
    pad_token: int,
    encode_fn: Callable[[str], List[int]],
) -> Dict[str, Any]:
    good_data = encode_sequences(read_fasta(good_fasta_path), block_size, pad_token, encode_fn)
    bad_data = encode_sequences(read_fasta(bad_fasta_path), block_size, pad_token, encode_fn)

    bad_mapping: List[List[int]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            bad_mapping.append([int(value) for value in row if value.strip()])

    if len(good_data) != len(bad_mapping):
        raise ValueError(
            f"Mismatch between good sequences ({len(good_data)}) and CSV rows ({len(bad_mapping)})."
        )

    return {
        "good_data": good_data,
        "bad_data": bad_data,
        "good_labels": build_labels_from_inputs(good_data, pad_token),
        "bad_labels": build_labels_from_inputs(bad_data, pad_token),
        "bad_mapping": bad_mapping,
        "ref_good_logps": None,
        "ref_bad_logps": None,
    }


def sample_indices(n_items: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    if n_items <= 0:
        raise ValueError("Cannot sample from an empty dataset.")
    if batch_size <= n_items:
        return torch.randperm(n_items, generator=generator)[:batch_size]
    return torch.randint(0, n_items, (batch_size,), generator=generator)


class PreferencePairDataset(Dataset):
    def __init__(self, dataset: Dict[str, Any]):
        self.good_data = dataset["good_data"]
        self.bad_data = dataset["bad_data"]
        self.good_labels = dataset["good_labels"]
        self.bad_labels = dataset["bad_labels"]
        self.ref_good_logps = dataset.get("ref_good_logps")
        self.ref_bad_logps = dataset.get("ref_bad_logps")

        pair_good_idx: List[int] = []
        pair_bad_idx: List[int] = []
        for good_idx, bad_indices in enumerate(dataset["bad_mapping"]):
            for bad_idx in bad_indices:
                pair_good_idx.append(good_idx)
                pair_bad_idx.append(bad_idx)

        self.pair_good_idx = torch.tensor(pair_good_idx, dtype=torch.long)
        self.pair_bad_idx = torch.tensor(pair_bad_idx, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.pair_good_idx.numel())

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return int(self.pair_good_idx[idx]), int(self.pair_bad_idx[idx])

    def _batch_from_indices(self, good_idx: torch.Tensor, bad_idx: torch.Tensor):
        outputs = [
            self.good_data.index_select(0, good_idx),
            self.bad_data.index_select(0, bad_idx),
            self.good_labels.index_select(0, good_idx),
            self.bad_labels.index_select(0, bad_idx),
        ]
        if self.ref_good_logps is not None and self.ref_bad_logps is not None:
            outputs.extend(
                [
                    self.ref_good_logps.index_select(0, good_idx),
                    self.ref_bad_logps.index_select(0, bad_idx),
                ]
            )
        return tuple(outputs)

    def collate_fn(self, batch: List[tuple[int, int]]):
        good_idx = torch.tensor([good for good, _ in batch], dtype=torch.long)
        bad_idx = torch.tensor([bad for _, bad in batch], dtype=torch.long)
        return self._batch_from_indices(good_idx, bad_idx)

    def sample_batch(self, batch_size: int, generator: torch.Generator):
        pair_idx = sample_indices(len(self), batch_size, generator)
        good_idx = self.pair_good_idx.index_select(0, pair_idx)
        bad_idx = self.pair_bad_idx.index_select(0, pair_idx)
        return self._batch_from_indices(good_idx, bad_idx)
