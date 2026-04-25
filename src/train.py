"""
Shared training utilities.

Public API
----------
FocalLoss(gamma)
train_one_epoch(model, loader, criterion, optimizer, device) -> (loss, acc)
evaluate(model, loader, criterion, device)               -> (loss, acc)
run_training(model, train_loader, val_loader, optimizer, criterion,
             scheduler, num_epochs, device, ckpt_path, classes) -> history
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Parameters
    ----------
    gamma : focusing parameter (default 2.0)
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns
    -------
    (mean_loss, accuracy)
    """
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in tqdm(loader, leave=False, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(imgs)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one evaluation pass (no gradient updates).

    Returns
    -------
    (mean_loss, accuracy)
    """
    model.eval()
    total_loss = correct = total = 0

    with torch.inference_mode():
        for imgs, labels in tqdm(loader, leave=False, desc="eval"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * len(imgs)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(imgs)

    return total_loss / total, correct / total


def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    num_epochs: int,
    device: torch.device,
    ckpt_path: str | Path | None = None,
    classes: list[str] | None = None,
) -> list[dict]:
    """Full training loop with best-checkpoint saving.

    Parameters
    ----------
    ckpt_path : if provided, saves the best val-loss checkpoint here
    classes   : stored inside the checkpoint for later loading

    Returns
    -------
    history : list of dicts with keys epoch, train_loss, train_acc,
              val_loss, val_acc, elapsed_sec
    """
    best_val_loss = float("inf")
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        history.append(dict(
            epoch=epoch,
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=vl_loss,   val_acc=vl_acc,
            elapsed_sec=elapsed,
        ))

        print(
            f"Epoch {epoch:>3}/{num_epochs} | "
            f"train {tr_loss:.4f}/{tr_acc:.4f}  "
            f"val {vl_loss:.4f}/{vl_acc:.4f}  "
            f"({elapsed:.0f}s)"
        )

        if ckpt_path is not None and vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    vl_loss,
                    "val_acc":     vl_acc,
                    "classes":     classes,
                },
                ckpt_path,
            )

    total_min = (time.time() - t_start) / 60
    print(f"\nTraining complete — {total_min:.1f} min total")
    return history
