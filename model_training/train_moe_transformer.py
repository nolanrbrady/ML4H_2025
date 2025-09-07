"""
Decoder-only Transformer training with schema-aware dataloaders and checkpoints.

- DataLoaders from columns_schema.json:
  - Features: schema['model_input_cols']
  - Targets: schema['observation_cols'] shifted by one timestep (next-state)
  - Batch keys: x, lengths, mask (bool), y (shifted targets)

- Model: Decoder-only Transformer with dual heads
  - Continuous/ordinal head -> MSE
  - Binary (one-hot) head -> BCEWithLogits

- Training: single function entrypoint + simple __main__ harness
  - Per-epoch train/val logging, best checkpoint tracking, optional per-epoch saves
"""

from pathlib import Path
import os
import sys
import json
import pandas as pd
import torch

# Ensure the parent package directory (autoreg) is importable when running this file directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from dataloader import build_dataloaders_from_schema
from models.moe_transformer import TransformerWithMoE
from preprocessing import ORDINAL_VARS

torch.manual_seed(42)

# -----------------------------
# Device
# -----------------------------
_device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {_device}")


def build_transformer_dataloaders(
    *,
    train_df,
    val_df,
    test_df,
    schema_path,
    batch_size=64,
    pad_value=0.0,
):
    """
    Build train/val/test DataLoaders using columns_schema.json.

    - Inputs (features): schema['model_input_cols']
    - Targets (y): schema['observation_cols'] shifted by one step
    - Padding value: pad_value
    """
    loaders = build_dataloaders_from_schema(
        train=train_df,
        val=val_df,
        test=test_df,
        schema_path=schema_path,
        id_col="subject_id",
        time_col="months_since_bl",
        batch_size=batch_size,
        shuffle_train=True,
        pad_value=pad_value,
        mask_dtype="bool",
        target_mode="shifted",
    )
    return loaders


def make_key_padding_mask(batch):
    """Return (B, S) bool mask with True at padding positions for nn.Transformer."""
    # batch['mask'] is True for valid steps; invert to mark padding
    return ~batch["mask"].bool()


def make_causal_attn_mask(seq_len, device=None):
    """Upper-triangular (S, S) bool mask with True where attention is disallowed (future)."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def mask_loss(loss_per_step, mask):
    """Average loss over True positions in `mask` (B, S[, ...])."""
    if loss_per_step.dim() > 2:
        loss_per_step = loss_per_step.view(loss_per_step.shape[0], loss_per_step.shape[1], -1).mean(-1)
    mask = mask.to(dtype=loss_per_step.dtype)
    total = (loss_per_step * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return total / denom


def make_shifted_target_mask(batch):
    """Mask (B, S) float where True/1.0 only for timesteps that have a next-step target."""
    # Valid steps mask (True for valid input steps)
    valid = batch["mask"].bool()  # (B, S)
    lengths = batch["lengths"].long()  # (B,)
    B, S = valid.shape
    # For each sequence, valid target positions are indices < (length-1)
    positions = torch.arange(S, device=lengths.device).unsqueeze(0).expand(B, S)
    max_valid_idx = (lengths - 1).clamp_min(0).unsqueeze(1)
    has_target = positions < max_valid_idx  # (B, S) bool
    out = (valid & has_target).to(dtype=torch.float32)
    return out


# -----------------------------
# Schema helpers
# -----------------------------

def _load_schema(schema_path):
    with open(schema_path, "r") as f:
        return json.load(f)


def _target_indices_from_schema(schema):
    """
    Map y_cont_cols and y_bin_cols to indices within observation_cols.

    Returns (cont_idx, bin_idx, observation_cols)
    """
    observation_cols = schema.get("observation_cols", [])
    y_cont_cols = schema.get("y_cont_cols", [])
    y_bin_cols = schema.get("y_bin_cols", [])
    if not observation_cols:
        raise ValueError("Schema missing 'observation_cols'")
    feat_index = {c: i for i, c in enumerate(observation_cols)}
    cont_idx = [feat_index[c] for c in y_cont_cols if c in feat_index]
    bin_idx = [feat_index[c] for c in y_bin_cols if c in feat_index]
    return cont_idx, bin_idx, observation_cols


def _ordinal_indices_within_cont(schema):
    """
    Ordinal indices within y_cont_cols to support rounding in inference.
    """
    y_cont_cols = schema.get("y_cont_cols", [])
    ord_names = set(ORDINAL_VARS)
    return [i for i, c in enumerate(y_cont_cols) if c in ord_names]


def build_model_from_schema(schema_path, *, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1,
                            num_experts=4, num_experts_per_tok=2):
    schema = _load_schema(schema_path)
    model_input_cols = schema.get("model_input_cols", [])
    if not model_input_cols:
        raise ValueError("Schema missing 'model_input_cols'")
    cont_idx, bin_idx, observation_cols = _target_indices_from_schema(schema)

    model = TransformerWithMoE(
        input_dim=len(model_input_cols),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        out_cont_dim=len(cont_idx),
        out_bin_dim=len(bin_idx),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    aux = {
        "cont_idx": cont_idx,
        "bin_idx": bin_idx,
        "ordinal_within_cont": _ordinal_indices_within_cont(schema),
        "schema": schema,
    }
    return model, aux


def _ordinal_indices_within_cont(schema):
    """
    Ordinal indices within y_cont_cols to support rounding in inference.
    """
    y_cont_cols = schema.get("y_cont_cols", [])
    ord_names = set(ORDINAL_VARS)
    return [i for i, c in enumerate(y_cont_cols) if c in ord_names]


# -----------------------------
# Model builder and training
# -----------------------------

def _save_checkpoint(*, model, aux, epoch, val_loss, output_dir, tag):
    """Save model state_dict + minimal metadata."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "aux": aux,
    }
    fname = f"epoch_{epoch:03d}_{tag}.pt" if val_loss is None else f"epoch_{epoch:03d}_val{val_loss:.4f}_{tag}.pt"
    out_path = os.path.join(output_dir, fname)
    torch.save(ckpt, out_path)
    return out_path

def _save_best_checkpoint(*, model, aux, epoch, val_loss, path):
    """Overwrite best checkpoint to a fixed path."""
    ckpt = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "aux": aux,
    }
    torch.save(ckpt, path)
    return path


def train_decoder_only_transformer(
    *,
    train_df,
    val_df,
    test_df,
    schema_path,
    device=None,
    # Hyperparameters
    epochs=50,
    batch_size=32,
    lr=3e-4,
    d_model=256,
    nhead=4,
    num_layers=3,
    dim_feedforward=768,
    dropout=0.3,
    num_experts=4,
    num_experts_per_tok=1,
    aux_loss_weight=0.005,
    weight_decay=0.01,
    lr_scheduler_patience=5,
    lr_scheduler_factor=0.5,
    lr_min=1e-5,
    # Checkpointing / early stopping
    output_dir="autoreg/model_training/moe_transformer_checkpoints",
    save_all_epochs=False,
    best_model_path="./best_moe_transformer_model.pt",
    patience=20,
):
    device = torch.device(device) if isinstance(device, str) else (device or _device)

    # Data
    loaders = build_transformer_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        schema_path=schema_path,
        batch_size=batch_size,
        pad_value=0.0,
    )

    # Model
    model, aux = build_model_from_schema(
        schema_path,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    model.to(device)

    cont_idx = aux["cont_idx"]
    bin_idx = aux["bin_idx"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_scheduler_factor, patience=lr_scheduler_patience,
        min_lr=lr_min, verbose=True
    )
    mse_loss_fn = torch.nn.MSELoss(reduction="none")
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def run_epoch(loader, train):
        model.train(train)
        total_mse = total_bce = total = 0.0
        total_aux = 0.0
        steps = 0
        for batch in loader:
            x = batch["x"].to(device)
            y_full = batch["y"].to(device)  # observation_cols order
            # Masks
            loss_mask = make_shifted_target_mask(batch).to(device)  # (B, S) float
            key_padding_mask = make_key_padding_mask(batch).to(device)  # (B, S) bool
            attn_mask = make_causal_attn_mask(x.size(1), device=device)  # (S, S) bool

            # Forward
            pred_cont, pred_bin = model(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

            loss_mse = torch.tensor(0.0, device=device)
            loss_bce = torch.tensor(0.0, device=device)

            if len(cont_idx) > 0 and pred_cont is not None:
                y_cont = y_full[..., cont_idx]
                per_step_mse = mse_loss_fn(pred_cont, y_cont).mean(-1)  # (B, S)
                loss_mse = mask_loss(per_step_mse, loss_mask)

            if len(bin_idx) > 0 and pred_bin is not None:
                y_bin = y_full[..., bin_idx]
                per_step_bce = bce_loss_fn(pred_bin, y_bin).mean(-1)  # (B, S)
                loss_bce = mask_loss(per_step_bce, loss_mask)

            # Auxiliary MoE load balancing loss (if available)
            aux_dict = getattr(model, "get_auxiliary_losses", lambda: {})()
            aux_loss = aux_dict.get("load_balancing_loss", None)
            if aux_loss is None:
                aux_loss_val = 0.0
                aux_term = torch.tensor(0.0, device=device)
            else:
                aux_term = aux_loss
                aux_loss_val = float(aux_term.detach().cpu())

            # Combine losses; weight auxiliary term
            loss = loss_mse + loss_bce + aux_loss_weight * aux_term

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_mse += float(loss_mse.detach().cpu())
            total_bce += float(loss_bce.detach().cpu())
            total += float(loss.detach().cpu())
            total_aux += aux_loss_val
            steps += 1
        if steps == 0:
            return 0.0, 0.0, 0.0, 0.0
        return total / steps, total_mse / steps, total_bce / steps, total_aux / steps

    history = {}
    best_val = None
    best_path = None
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        # Train
        train_total, train_mse, train_bce, train_aux = run_epoch(loaders["train"], train=True)
        history[f"train/total_epoch_{epoch}"] = train_total
        history[f"train/aux_epoch_{epoch}"] = train_aux

        # Validation (and logging)
        val_total = None
        if "val" in loaders:
            val_total, val_mse, val_bce, val_aux = run_epoch(loaders["val"], train=False)
            history[f"val/total_epoch_{epoch}"] = val_total
            history[f"val/aux_epoch_{epoch}"] = val_aux
            # Step LR scheduler on validation loss
            scheduler.step(val_total)
            # Report current LR
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d}/{epochs:03d} - Train: {train_total:.4f} - Val: {val_total:.4f} - Aux(train): {train_aux:.4f} - LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch:03d}/{epochs:03d} - Train: {train_total:.4f} - Aux(train): {train_aux:.4f}")

        # Save checkpoint for this epoch
        if save_all_epochs:
            _save_checkpoint(model=model, aux=aux, epoch=epoch, val_loss=val_total, output_dir=output_dir, tag="ckpt")

        # Track best by validation total loss (and early stopping)
        if val_total is not None:
            if (best_val is None) or (val_total < best_val - 1e-12):
                prev_best = best_val
                best_val = val_total
                best_path = _save_best_checkpoint(model=model, aux=aux, epoch=epoch, val_loss=val_total, path=best_model_path)
                prev_str = "inf" if prev_best is None else f"{prev_best:.4f}"
                print(f"  Validation improved: {prev_str} -> {best_val:.4f} (saved {best_path})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered (patience={patience}). Best Val: {best_val:.4f}")
                    break

    # Load best checkpoint and test
    if best_path is not None and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])  # type: ignore[index]
        history["checkpoint/best"] = best_path

    if "test" in loaders:
        test_total, _, _, test_aux = run_epoch(loaders["test"], train=False)
        print(f"Test: {test_total:.4f}")
        history["test/total"] = test_total
        history["test/aux"] = test_aux

    return model, history


def round_ordinals_inplace(pred_cont, schema_path):
    """
    Round ordinal outputs in-place using ORDINAL_VARS ∩ y_cont_cols.
    Expects pred_cont shaped (B, S, Cc).
    """
    schema = _load_schema(schema_path)
    y_cont_cols = schema.get("y_cont_cols", [])
    ord_names = set(ORDINAL_VARS)
    ordinal_idx = [i for i, c in enumerate(y_cont_cols) if c in ord_names]
    if not ordinal_idx:
        return
    with torch.no_grad():
        pred_cont[..., ordinal_idx] = torch.round(pred_cont[..., ordinal_idx])


if __name__ == "__main__":
    import numpy as np

    # =============================
    # Easy-to-edit configuration
    # =============================
    # Data
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../clinician_policy/columns_schema.json")
    TRAIN_CSV = "./X_train.csv"
    VAL_CSV = "./X_val.csv"
    TEST_CSV = "./X_test.csv"
    # Model/optim (tuned for ~1k training examples)
    EPOCHS = 200
    BATCH_SIZE = 32
    LR = 3e-4
    D_MODEL = 256
    NHEAD = 4
    NUM_LAYERS = 3
    DIM_FEEDFORWARD = 768
    DROPOUT = 0.3
    NUM_EXPERTS = 8
    NUM_EXPERTS_PER_TOK = 1
    AUX_LOSS_WEIGHT = 0.005
    WEIGHT_DECAY = 0.01
    LR_SCHED_PATIENCE = 5
    LR_SCHED_FACTOR = 0.5
    LR_MIN = 1e-5
    # Checkpointing / early stopping
    BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_moe_transformer_model.pt")
    PATIENCE = 50

    # Load DataFrames
    def _read_csv_safe(p):
        return pd.read_csv(p) if (p is not None and os.path.exists(p)) else None

    train_df = _read_csv_safe(TRAIN_CSV)
    val_df = _read_csv_safe(VAL_CSV)
    test_df = _read_csv_safe(TEST_CSV)

    if train_df is None or val_df is None or test_df is None:
        raise SystemExit("Please set TRAIN_CSV, VAL_CSV, and TEST_CSV to valid file paths.")

    # Kick off training
    model, history = train_decoder_only_transformer(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        schema_path=SCHEMA_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        aux_loss_weight=AUX_LOSS_WEIGHT,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_patience=LR_SCHED_PATIENCE,
        lr_scheduler_factor=LR_SCHED_FACTOR,
        lr_min=LR_MIN,
        best_model_path=BEST_MODEL_PATH,
        patience=PATIENCE,
    )
    print("Training complete.")
    if "checkpoint/best" in history:
        print(f"Best checkpoint: {history['checkpoint/best']}")
