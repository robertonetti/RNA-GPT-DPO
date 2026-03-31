from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from torch.utils.data import Dataset


def resolve_path(script_dir: Path, p: str) -> Path:
    """Resolve a filesystem path string into an absolute Path.

    Inputs:
    - script_dir: Base directory used to resolve relative paths.
    - p: Absolute or relative path string.

    Output:
    - Absolute Path pointing to the target location.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return (script_dir / path).resolve()


def read_fasta(path: Path) -> List[str]:
    """Read a FASTA file and return one concatenated sequence per record.

    Inputs:
    - path: FASTA file path.

    Output:
    - List of length N_records.
    - Each item is a plain string sequence (wrapped FASTA lines merged).
    """
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
    """Encode a sequence and force fixed length.

    Inputs:
    - seq: Raw sequence string.
    - block_size: Target token length T.
    - pad_token: Integer id used for right-padding.
    - encode_fn: String-to-token-id function.

    Processing:
    - Wraps sequence with boundary markers "\n".
    - Encodes to token ids.
    - Right-pads or truncates to exactly block_size.

    Output:
    - Tensor of shape (T,) and dtype torch.long.
    """
    txt = f"\n{seq}\n"
    ids = encode_fn(txt)
    if len(ids) < block_size:
        ids += [pad_token] * (block_size - len(ids))
    else:
        ids = ids[:block_size]
    return torch.tensor(ids, dtype=torch.long)


def _build_labels_from_inputs(x: torch.Tensor, pad_token: int) -> torch.Tensor:
    """Create next-token labels by left-shifting each sequence.

    Inputs:
    - x: Input token tensor of shape (B, T).
    - pad_token: Token id appended in the last label position.

    Output:
    - Label tensor of shape (B, T) where y[:, t] = x[:, t+1], and the last
      column is pad_token.
    """
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
    """Load encoded good/bad sequences and their preference mapping.

    Inputs:
    - good_fasta_path: FASTA with preferred sequences.
    - bad_fasta_path: FASTA with dispreferred sequences.
    - csv_path: CSV where each row lists bad indices paired to one good sample.
    - block_size: Target encoded length T.
    - pad_token: Padding token id.
    - encode_fn: Sequence encoder.

    Output dictionary:
    - good_data: tensor of shape (N_good, T).
    - bad_data: tensor of shape (N_bad, T).
    - good_labels: tensor of shape (N_good, T).
    - bad_labels: tensor of shape (N_bad, T).
    - weight_tensor: float tensor of shape (N_good,).
    - bad_mapping: list of length N_good; each item is a list of bad indices.
    - ref_good_logps/ref_bad_logps: optional tensors filled later after ref-model precompute.
    """
    good_seqs = read_fasta(good_fasta_path)
    bad_seqs = read_fasta(bad_fasta_path)

    good_data_list = [pad_encode(s, block_size, pad_token, encode_fn) for s in good_seqs]
    bad_data_list = [pad_encode(s, block_size, pad_token, encode_fn) for s in bad_seqs]
    good_data = (
        torch.stack(good_data_list, dim=0)
        if len(good_data_list) > 0
        else torch.empty((0, block_size), dtype=torch.long)
    )
    bad_data = (
        torch.stack(bad_data_list, dim=0)
        if len(bad_data_list) > 0
        else torch.empty((0, block_size), dtype=torch.long)
    )
    good_labels = _build_labels_from_inputs(good_data, pad_token)
    bad_labels = _build_labels_from_inputs(bad_data, pad_token)

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
        "good_labels": good_labels,
        "bad_labels": bad_labels,
        "weight_tensor": torch.tensor(weights, dtype=torch.float),
        "bad_mapping": bad_indices_mapping,
        "ref_good_logps": None,
        "ref_bad_logps": None,
    }


class PreferencePairDataset(Dataset):
    """Expand good->bad mapping into explicit (good, bad) training pairs."""

    def __init__(self, dataset: Dict[str, Any]):
        """Flatten one-to-many mapping into a pair index list.

        Input:
        - dataset: Dictionary returned by load_dpo_dataset.

        Internal representation:
        - all_pairs has length N_pairs; each element is (good_idx, bad_idx).
        """
        self.good_data = dataset["good_data"]
        self.bad_data = dataset["bad_data"]
        self.good_labels = dataset["good_labels"]
        self.bad_labels = dataset["bad_labels"]
        self.ref_good_logps = dataset.get("ref_good_logps")
        self.ref_bad_logps = dataset.get("ref_bad_logps")
        pair_good_idx: List[int] = []
        pair_bad_idx: List[int] = []
        for good_idx, bad_idxs in enumerate(dataset["bad_mapping"]):
            for bad_idx in bad_idxs:
                pair_good_idx.append(good_idx)
                pair_bad_idx.append(bad_idx)
        self.pair_good_idx = torch.tensor(pair_good_idx, dtype=torch.long)
        self.pair_bad_idx = torch.tensor(pair_bad_idx, dtype=torch.long)

    def __len__(self) -> int:
        """Return number of explicit preference pairs.

        Output:
        - Integer N_pairs.
        """
        return int(self.pair_good_idx.numel())

    def __getitem__(self, idx: int):
        """Return one pair as integer indices.

        Input:
        - idx: Pair index in [0, N_pairs).

        Output:
        - Tuple (good_idx, bad_idx).
        """
        return int(self.pair_good_idx[idx]), int(self.pair_bad_idx[idx])

    def collate_fn(self, batch: List[tuple[int, int]]):
        """Gather one mini-batch with vectorized index_select operations."""
        if len(batch) == 0:
            raise ValueError("Empty batch is not supported.")

        good_idx = torch.tensor([g for g, _ in batch], dtype=torch.long)
        bad_idx = torch.tensor([b for _, b in batch], dtype=torch.long)

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
