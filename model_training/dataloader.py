"""
Utilities to build autoregressive-ready DataLoaders.

This module converts train/val/test datasets into padded batch tensors with
shape (batch, sequence_length, features), along with length and mask tensors
to handle variable-length sequences for RNNs/Transformers.

Supported inputs:
- A pandas DataFrame with an ID column, a time/visit column, and feature columns
- A pre-built list of sequences (numpy arrays or torch tensors) with shape (T, F)

Outputs per batch:
- x: float tensor of shape (B, S, F) with zero padding by default
- lengths: long tensor of shape (B,) with original sequence lengths
- mask: float or bool tensor of shape (B, S); 1.0 for valid, 0.0 for padding
- y (optional): when target_mode="shifted", returns next-step targets aligned
  with x for autoregressive prediction; same shape as x, with padded tail

Example usage (DataFrame input):
    df_train, df_val, df_test = ...
    dl = build_dataloaders(
        train=df_train,
        val=df_val,
        test=df_test,
        id_col="subject_id",
        time_col="visit_month",
        feature_cols=["feat1", "feat2", ...],
        batch_size=64,
        target_mode="shifted",
    )
    for batch in dl["train"]:
        x, lengths, mask, y = batch["x"], batch["lengths"], batch["mask"], batch.get("y")

Example usage (prebuilt sequences):
    train_seqs = [np.array([[...], ...], dtype=float), torch.tensor(...), ...]
    dl = build_dataloaders(
        train=train_seqs,
        val=val_seqs,
        test=test_seqs,
        batch_size=32,
    )

Notes:
- Padding value defaults to 0.0.
- Mask uses 1.0 for valid steps and 0.0 for padding by default (float mask);
  set mask_dtype=bool to return a boolean mask instead.
"""

import math

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception as e:  # pragma: no cover - allows static usage without torch installed
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import numpy as np


# -----------------------------
# Data preparation helpers
# -----------------------------

def sequences_from_dataframe(
    df,
    *,
    id_col,
    time_col=None,
    feature_cols=None,
    sort_time=True,
    max_seq_len=None,
):
    """
    Convert a long-form DataFrame into a list of per-entity sequences.

    Each sequence has shape (T, F) where T is the number of visits for the entity
    and F is len(feature_cols). If time_col is provided and sort_time is True,
    visits are sorted by time ascending within each entity.

    Args:
        df: Input DataFrame containing id, time, and feature columns.
        id_col: Column identifying entities (e.g., patient/subject).
        time_col: Column identifying visit order or timestamp. Optional.
        feature_cols: Columns to include as features. If None, uses all columns
            except id_col and time_col.
        sort_time: Whether to sort by time_col within each entity.
        max_seq_len: Optional maximum per-entity sequence length (truncate).

    Returns:
        List of torch.FloatTensor sequences with shape (T, F).
    """
    if pd is None:
        raise ImportError("pandas is required for DataFrame inputs")
    if torch is None:
        raise ImportError("torch is required to build tensors from DataFrame")

    if feature_cols is None:
        drop_cols = [id_col] + ([time_col] if time_col is not None else [])
        feature_cols = [c for c in df.columns if c not in drop_cols]

    # Ensure deterministic ordering: by id, then time (if provided)
    if sort_time and time_col is not None:
        df_sorted = df.sort_values([id_col, time_col])
    else:
        df_sorted = df.sort_values([id_col])

    sequences = []
    for _, g in df_sorted.groupby(id_col, sort=False):
        arr = g[list(feature_cols)].to_numpy(dtype=float, copy=False)
        if max_seq_len is not None:
            arr = arr[: max(0, int(max_seq_len))]
        if arr.shape[0] == 0:
            continue  # skip empty sequences
        sequences.append(torch.tensor(arr, dtype=torch.float32))
    return sequences


# -----------------------------
# Dataset and collator
# -----------------------------

class AutoregressiveSequenceDataset(Dataset):
    """A simple Dataset over a list of variable-length sequences (T, F)."""

    def __init__(self, sequences):
        if torch is None:
            raise ImportError("torch is required for AutoregressiveSequenceDataset")
        self._seqs = []
        for s in sequences:
            if isinstance(s, np.ndarray):
                t = torch.tensor(s, dtype=torch.float32)
            elif isinstance(s, torch.Tensor):
                t = s.to(dtype=torch.float32)
            else:
                raise TypeError("Each sequence must be a numpy array or torch tensor")
            if t.dim() != 2:
                raise ValueError(f"Each sequence must have shape (T, F), got {tuple(t.shape)}")
            if t.shape[0] == 0:
                continue  # skip empty sequences
            self._seqs.append(t)

        if len(self._seqs) == 0:
            raise ValueError("No non-empty sequences provided")

        # Verify consistent feature dimension across sequences
        f0 = self._seqs[0].shape[1]
        for i, s in enumerate(self._seqs):
            if s.shape[1] != f0:
                raise ValueError(
                    f"Inconsistent feature dimension at index {i}: {s.shape[1]} != {f0}"
                )

        self.feature_dim = f0

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]

class CollateConfig:
    def __init__(self, pad_value=0.0, mask_dtype="float", target_mode="none", target_indices=None):
        # pad_value: float; mask_dtype: "float" or "bool"; target_mode: "none" or "shifted"
        self.pad_value = pad_value
        self.mask_dtype = mask_dtype
        self.target_mode = target_mode
        # Optional subset of feature indices to return as targets when target_mode is enabled
        self.target_indices = target_indices


class SequenceBatchCollator:
    """
    Collate function to pad variable-length sequences to a batch tensor.

    Returns a dict with keys:
      - x: Tensor[B, S, F]
      - lengths: Tensor[B]
      - mask: Tensor[B, S] where 1.0 (or True) indicates valid (non-pad)
      - y: Optional[Tensor[B, S, F]] present when target_mode="shifted"
    """

    def __init__(self, cfg=None):
        if torch is None:
            raise ImportError("torch is required for SequenceBatchCollator")
        self.cfg = cfg or CollateConfig()

    def __call__(self, batch):
        if len(batch) == 0:
            raise ValueError("Empty batch")

        # Compute per-sequence lengths and padding target length
        lengths = torch.tensor([int(seq.shape[0]) for seq in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        feature_dim = int(batch[0].shape[1])
        batch_size = len(batch)

        # Allocate padded input tensor (B, S, F)
        x = torch.full(
            (batch_size, max_len, feature_dim),
            fill_value=self.cfg.pad_value,
            dtype=torch.float32,
        )

        if self.cfg.mask_dtype == "bool":
            mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        else:
            mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

        # Copy each variable-length sequence into the padded tensor and build mask
        for i, seq in enumerate(batch):
            seq_len = int(seq.shape[0])
            x[i, :seq_len, :] = seq
            if self.cfg.mask_dtype == "bool":
                mask[i, :seq_len] = True
            else:
                mask[i, :seq_len] = 1.0

        out = {"x": x, "lengths": lengths, "mask": mask}

        if self.cfg.target_mode == "shifted":
            # Determine output target dimension: all features or a subset
            if self.cfg.target_indices is not None:
                target_dim = len(self.cfg.target_indices)
            else:
                target_dim = feature_dim

            # Allocate padded target tensor (B, S, Fy)
            y = torch.full(
                (batch_size, max_len, target_dim),
                fill_value=self.cfg.pad_value,
                dtype=torch.float32,
            )

            # For each sequence, produce next-step targets: y[t] = seq[t+1]
            # The final timestep has no next-step target and remains padding.
            for i, seq in enumerate(batch):
                seq_len = int(seq.shape[0])
                if seq_len > 1:
                    if self.cfg.target_indices is None:
                        y[i, : seq_len - 1, :] = seq[1:, :]
                    else:
                        y[i, : seq_len - 1, :] = seq[1:, self.cfg.target_indices]
                # y[i, seq_len-1:, :] left as pad_value
            out["y"] = y

        return out


# -----------------------------
# DataLoader builders
# -----------------------------

def _ensure_torch():
    if torch is None:
        raise ImportError("torch is required for building DataLoaders")


def make_dataloader_from_sequences(
    sequences,
    *,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    pad_value=0.0,
    mask_dtype="float",
    target_mode="none",
    target_indices=None,
):
    """
    Build a DataLoader from a list of sequences (T, F).
    """
    _ensure_torch()
    dataset = AutoregressiveSequenceDataset(sequences)
    collate_fn = SequenceBatchCollator(
        CollateConfig(
            pad_value=pad_value,
            mask_dtype=mask_dtype,
            target_mode=target_mode,
            target_indices=target_indices,
        )
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def make_dataloader_from_dataframe(
    df,
    *,
    id_col,
    time_col=None,
    feature_cols=None,
    target_cols=None,
    sort_time=True,
    max_seq_len=None,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    pad_value=0.0,
    mask_dtype="float",
    target_mode="none",
):
    """
    Build a DataLoader from a long-form DataFrame.
    """
    seqs = sequences_from_dataframe(
        df,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        sort_time=sort_time,
        max_seq_len=max_seq_len,
    )
    # Map target_cols to indices within feature_cols if provided
    target_indices = None
    if target_cols is not None:
        if feature_cols is None:
            # If feature_cols was None, infer it the same way as sequences_from_dataframe did
            drop_cols = [id_col] + ([time_col] if time_col is not None else [])
            feature_cols = [c for c in df.columns if c not in drop_cols]
        feat_index = {c: i for i, c in enumerate(feature_cols)}
        missing = [c for c in target_cols if c not in feat_index]
        if len(missing) > 0:
            raise ValueError(f"Some target_cols are not in feature_cols: {missing}")
        target_indices = [feat_index[c] for c in target_cols]
    return make_dataloader_from_sequences(
        seqs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        pad_value=pad_value,
        mask_dtype=mask_dtype,
        target_mode=target_mode,
        target_indices=target_indices,
    )


def build_dataloaders(
    *,
    train,
    val=None,
    test=None,
    # DataFrame-specific args
    id_col=None,
    time_col=None,
    feature_cols=None,
    target_cols=None,
    sort_time=True,
    max_seq_len=None,
    # Loader args
    batch_size=64,
    shuffle_train=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    pad_value=0.0,
    mask_dtype="float",
    target_mode="none",
):
    """
    Create train/val/test DataLoaders from either DataFrames or sequences.

    If any of train/val/test is a DataFrame, id_col must be provided, and
    feature_cols should identify the feature set for sequences.
    """
    _ensure_torch()

    def _mk(obj, shuffle):
        if obj is None:
            return None
        if pd is not None and isinstance(obj, pd.DataFrame):
            if id_col is None:
                raise ValueError("id_col must be provided for DataFrame inputs")
            return make_dataloader_from_dataframe(
                obj,
                id_col=id_col,
                time_col=time_col,
                feature_cols=feature_cols,
                target_cols=target_cols,
                sort_time=sort_time,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                pad_value=pad_value,
                mask_dtype=mask_dtype,
                target_mode=target_mode,
            )
        # Otherwise treat as sequences
        return make_dataloader_from_sequences(
            obj,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            pad_value=pad_value,
            mask_dtype=mask_dtype,
            target_mode=target_mode,
        )

    loaders = {}
    tr = _mk(train, shuffle_train)
    if tr is not None:
        loaders["train"] = tr
    va = _mk(val, False)
    if va is not None:
        loaders["val"] = va
    te = _mk(test, False)
    if te is not None:
        loaders["test"] = te
    return loaders


# -----------------------------
# Schema-aware convenience
# -----------------------------

def build_dataloaders_from_schema(
    *,
    train,
    val,
    test,
    schema_path,
    id_col="subject_id",
    time_col="months_since_bl",
    batch_size=64,
    shuffle_train=True,
    pad_value=0.0,
    mask_dtype="float",
    target_mode="shifted",
    num_workers=0,
    pin_memory=False,
    drop_last=False,
):
    """
    Build DataLoaders using schema-defined column order.

    - Features (x): schema['model_input_cols']
    - Targets (y): schema['observation_cols'] (when target_mode != 'none')
    - Validates that every observation column appears in model_input_cols
      because targets are selected as a subset of the feature tensor (shifted).
    """
    import json
    with open(schema_path, "r") as f:
        schema = json.load(f)
    model_input_cols = schema.get("model_input_cols", [])
    observation_cols = schema.get("observation_cols", [])
    if not model_input_cols:
        raise ValueError("Schema missing 'model_input_cols'")
    if target_mode != "none" and not observation_cols:
        raise ValueError("Schema missing 'observation_cols' required for targets")

    # Targets are selected as a subset of the feature tensor; ensure schema matches
    if target_mode != "none":
        not_in_features = [c for c in observation_cols if c not in model_input_cols]
        if not_in_features:
            raise ValueError(
                "Every observation_cols item must also appear in model_input_cols; missing in features: "
                + ", ".join(not_in_features)
            )

    # Validate that all required feature columns exist in provided DataFrames
    def _validate_df(df, df_name):
        if df is None:
            return
        missing = [c for c in model_input_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns in {df_name}: {missing}. "
                f"Ensure your CSVs contain all schema model_input_cols from {schema_path}."
            )

    _validate_df(train, "train")
    _validate_df(val, "val")
    _validate_df(test, "test")

    return build_dataloaders(
        train=train,
        val=val,
        test=test,
        id_col=id_col,
        time_col=time_col,
        feature_cols=model_input_cols,
        target_cols=observation_cols if target_mode != "none" else None,
        sort_time=True,
        max_seq_len=None,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        pad_value=pad_value,
        mask_dtype=mask_dtype,
        target_mode=target_mode,
    )


# -----------------------------
# Convenience: minimal sanity check
# -----------------------------

def _sanity_check_example():  # pragma: no cover - illustrative only
    if torch is None:
        return
    # Create three sequences with lengths 3, 2, 4; feature dim 2
    seqs = [
        torch.tensor([[1.0, 1.1], [2.0, 2.2], [3.0, 3.3]]),
        torch.tensor([[10.0, 10.1], [20.0, 20.2]]),
        torch.tensor([[7.0, 7.1], [8.0, 8.1], [9.0, 9.1], [10.0, 10.1]]),
    ]
    dl = make_dataloader_from_sequences(seqs, batch_size=2, target_mode="shifted")
    batch = next(iter(dl))
    x, lengths, mask, y = batch["x"], batch["lengths"], batch["mask"], batch.get("y")
    assert x.shape == (2, 3, 2)
    assert lengths.tolist() == [3, 2]
    assert mask.shape == (2, 3)
    if y is not None:
        assert y.shape == x.shape
