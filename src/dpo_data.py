from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, List, Dict, Any

import torch
from torch.utils.data import Dataset


def resolve_path(script_dir: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (script_dir / path).resolve()


def read_fasta(path: Path) -> List[str]:
    """Read a FASTA file and return one sequence string per record."""
    seqs: List[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    curr_seq: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if curr_seq:
                seqs.append("".join(curr_seq))
                curr_seq = []
        else:
            curr_seq.append(line)

    if curr_seq:
        seqs.append("".join(curr_seq))
    return seqs


def pad_encode(
    seq: str,
    block_size: int,
    pad_token: int,
    encode_fn: Callable[[str], List[int]],
) -> torch.Tensor:
    """Encode one sequence and force fixed length through truncation/padding."""
    txt = f"\n{seq}\n"
    ids = encode_fn(txt)
    if len(ids) < block_size:
        ids += [pad_token] * (block_size - len(ids))
    else:
        ids = ids[:block_size]
    return torch.tensor(ids, dtype=torch.long)


def _build_labels_from_inputs(x: torch.Tensor, pad_token: int) -> torch.Tensor:
    """Build next-token labels from input tokens by shifting left and padding tail."""
    pad_col = torch.full((x.size(0), 1), pad_token, dtype=x.dtype, device=x.device)
    return torch.cat([x[:, 1:], pad_col], dim=1)


def load_dpo_dataset(
    good_fasta_path: Path,
    bad_fasta_path: Path,
    csv_path: Path,
    block_size: int,
    pad_token: int,
    encode_fn: Callable[[str], List[int]],
) -> Dict[str, Any]:
    """Load preferred/dispreferred FASTA files and the good->bad index mapping CSV."""
    good_seqs = read_fasta(good_fasta_path)
    bad_seqs = read_fasta(bad_fasta_path)

    good_data = [pad_encode(s, block_size, pad_token, encode_fn) for s in good_seqs]
    bad_data = [pad_encode(s, block_size, pad_token, encode_fn) for s in bad_seqs]

    weights: List[float] = []
    bad_indices_mapping: List[List[int]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            bad_idxs = [int(x) for x in row if x.strip()]
            bad_indices_mapping.append(bad_idxs)
            weights.append(float(len(bad_idxs)))

    assert len(good_data) == len(weights), (
        f"Mismatch: {len(good_data)} good sequences but {len(weights)} CSV rows."
    )

    return {
        "good_data": good_data,
        "bad_data": bad_data,
        "weight_tensor": torch.tensor(weights, dtype=torch.float),
        "bad_mapping": bad_indices_mapping,
    }


class PreferencePairDataset(Dataset):
    """Dataset that expands good->bad mapping into explicit (good, bad) pair samples."""

    def __init__(self, dataset: Dict[str, Any]):
        self.good_data = dataset["good_data"]
        self.bad_data = dataset["bad_data"]
        self.all_pairs: List[tuple[int, int]] = []
        for good_idx, bad_idxs in enumerate(dataset["bad_mapping"]):
            for bad_idx in bad_idxs:
                self.all_pairs.append((good_idx, bad_idx))

    def __len__(self) -> int:
        return len(self.all_pairs)

    def __getitem__(self, idx: int):
        g, b = self.all_pairs[idx]
        return self.good_data[g], self.bad_data[b]
