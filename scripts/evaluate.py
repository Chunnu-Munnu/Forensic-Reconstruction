"""
Phase 2, Step 3: evaluate all three trained checkpoints on the held-out
CISS test set (real-world crashes, never seen during training or model
selection) and produce the paper's comparison table + confusion matrices.

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate.py

Because the CISS test set has ZERO Head-on (class 2) examples (see
config.py / documentation.txt Part 2.4), classification_report and
f1_score are called with an explicit `labels=[0,1,2,3,4]` and
`zero_division=0` so class 2 still gets a printed row (support=0) instead
of silently vanishing from the report -- and BOTH a 5-class and a
4-class-excluding-head-on macro-F1 are reported, because the 5-class
number is quietly optimistic (sklearn effectively ignores a class with 0
support when it has 0 predictions for it too) and the 4-class number is
the one that actually reflects real-world classification quality.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, NUM_CLASSES, model_dir, results_dir, tensor_dir  # noqa: E402
from models import build_model  # noqa: E402
import train as train_module  # noqa: E402
from train import MODEL_NAMES, load_split  # noqa: E402


def evaluate_checkpoint(model_key, X_test, mask_test, device, ckpt_dir):
    class_name = MODEL_NAMES[model_key]
    ckpt_path = ckpt_dir / f"{class_name}_best.pt"
    model = build_model(model_key, input_dim=X_test.shape[-1], num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    preds, probs, attn = [], [], []
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().to(device)
        mask_t = torch.from_numpy(mask_test).bool().to(device)
        # chunk to keep memory bounded even though this dataset is small
        for i in range(0, len(X_t), 256):
            out = model(X_t[i:i + 256], pad_mask=mask_t[i:i + 256])
            if isinstance(out, tuple):
                logits, a = out
                attn.append(a.cpu().numpy())
            else:
                logits = out
            p = torch.softmax(logits, dim=1)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(p.cpu().numpy())

    return (np.concatenate(preds), np.concatenate(probs),
            np.concatenate(attn) if attn else None)


def save_confusion_matrix(y_true, y_pred, class_name, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{class_name} -- test set")
    plt.tight_layout()
    out_path = out_dir / f"{class_name}_confusion.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return cm, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-mode", choices=["source", "mixed"], default="source",
                         help="'source' (default) evaluates the strict-split checkpoints in "
                              "models/strict/. 'mixed' evaluates the mixed-split checkpoints "
                              "in models/mixed/.")
    args = parser.parse_args()

    ckpt_dir = model_dir(args.split_mode)
    out_dir = results_dir(args.split_mode)
    train_module._TENSOR_DIR = tensor_dir(args.split_mode)

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test, y_test, mask_test = load_split("test")
    test_label = "CISS real-world held-out (strict split)" if args.split_mode == "source" \
        else "CISS real-world held-out 30% (mixed split)"
    print(f"Test set ({test_label}): {len(y_test)} events")
    print("Class support in test set:",
          {CLASS_NAMES[c]: int((y_test == c).sum()) for c in range(NUM_CLASSES)})

    summary = {}
    for model_key in MODEL_NAMES:
        class_name = MODEL_NAMES[model_key]
        preds, probs, attn = evaluate_checkpoint(model_key, X_test, mask_test, device, ckpt_dir)

        print(f"\n{'=' * 70}\n{class_name}\n{'=' * 70}")
        report = classification_report(
            y_test, preds, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
            zero_division=0,
        )
        print(report)

        acc = accuracy_score(y_test, preds)
        macro_f1_5 = f1_score(y_test, preds, average="macro",
                               labels=list(range(NUM_CLASSES)), zero_division=0)
        # 4-class macro-F1 excluding Head-on (class 2), which has 0 support
        # in CISS -- see module docstring.
        present_labels = [c for c in range(NUM_CLASSES) if c != 2]
        macro_f1_4 = f1_score(y_test, preds, average="macro",
                               labels=present_labels, zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted",
                                labels=list(range(NUM_CLASSES)), zero_division=0)

        cm, cm_path = save_confusion_matrix(y_test, preds, class_name, out_dir)
        print(f"Accuracy: {acc:.4f} | Macro-F1 (5-class): {macro_f1_5:.4f} | "
              f"Macro-F1 (4-class, excl. Head-on): {macro_f1_4:.4f} | "
              f"Weighted-F1: {weighted_f1:.4f}")
        print(f"Confusion matrix saved to {cm_path}")

        summary[model_key] = {
            "class_name": class_name,
            "accuracy": acc,
            "macro_f1_5class": macro_f1_5,
            "macro_f1_4class_excl_headon": macro_f1_4,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        if attn is not None:
            np.save(out_dir / f"{class_name}_test_attention_weights.npy", attn)
            print(f"Saved per-event attention weights ({attn.shape}) for Phase 3.")

        np.save(out_dir / f"{class_name}_test_probs.npy", probs)
        np.save(out_dir / f"{class_name}_test_preds.npy", preds)

    print(f"\n{'=' * 90}\nFINAL COMPARISON TABLE ({test_label}, n={len(y_test)})\n{'=' * 90}")
    header = f"{'Model':<28}{'Accuracy':>10}{'Macro-F1 (5cls)':>18}{'Macro-F1 (4cls)':>18}{'Weighted-F1':>14}"
    print(header)
    for key, s in summary.items():
        print(f"{s['class_name']:<28}{s['accuracy']:>10.4f}{s['macro_f1_5class']:>18.4f}"
              f"{s['macro_f1_4class_excl_headon']:>18.4f}{s['weighted_f1']:>14.4f}")

    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
