"""Per-epoch Val MCC injection for GraphBAN's Trainer (instance-level patch)."""
from __future__ import annotations

import torch

from .evaluation import collect_predictions, optimize_threshold_on_validation


def patch_trainer_for_mcc_logging(
    trainer_obj,
    val_gen: torch.utils.data.DataLoader,
    device: torch.device,
    n_class: int,
) -> None:
    """Inject a Val MCC line after GraphBAN's own AUROC/AUPRC output per epoch.

    Monkey-patches ``trainer_obj.test`` at the instance level, so no other
    Trainer instances are affected. Errors inside the patch are silently
    suppressed to avoid interrupting training.
    """
    original_test = trainer_obj.test

    def _test_with_mcc(*args, **kwargs):
        result = original_test(*args, **kwargs)
        try:
            y_true, y_prob = collect_predictions(
                trainer_obj.model, val_gen, device, n_class,
            )
            _, val_mcc = optimize_threshold_on_validation(
                y_true, y_prob, metric="mcc",
            )
            print(f"  → Val MCC={val_mcc:.4f}")
        except Exception:
            pass
        return result

    trainer_obj.test = _test_with_mcc
