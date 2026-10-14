"""Train a patient-separated multi-label respiratory sound event model.

The model detects annotated crackle and wheeze events independently; both may
be present in the same cycle. These are acoustic event labels, not diagnoses.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

from .data import read_manifest, RespiratoryDataset
from .model import DualViewRespiratoryNet

PATTERNS = ["normal", "crackle", "wheeze", "both"]
EVENTS = ["crackle", "wheeze"]


def pattern_to_events(pattern):
    """Convert a cycle annotation to independent crackle/wheeze targets."""
    return [float(pattern in ("crackle", "both")),
            float(pattern in ("wheeze", "both"))]


def events_to_pattern(crackle, wheeze):
    """Convert two independent event decisions to a display label."""
    if crackle and wheeze:
        return "both"
    if crackle:
        return "crackle"
    if wheeze:
        return "wheeze"
    return "normal"


def event_targets(patterns):
    return np.asarray([pattern_to_events(str(x)) for x in patterns], dtype=np.float32)


def tune_thresholds(y_true, probabilities):
    """Tune event thresholds on validation patients only."""
    thresholds = []
    for j in range(len(EVENTS)):
        choices = np.arange(0.20, 0.801, 0.025)
        scored = []
        for threshold in choices:
            pred = probabilities[:, j] >= threshold
            score = balanced_accuracy_score(y_true[:, j], pred)
            scored.append((float(score), -abs(float(threshold) - 0.5), float(threshold)))
        thresholds.append(max(scored)[2])
    return thresholds


def predict_event_scores(model, loader, device, labels):
    model.eval()
    true_patterns = []
    probability_batches = []
    with torch.inference_mode():
        for mel, desc, y in loader:
            logits = model(mel.to(device), desc.to(device))
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
            true_patterns.extend(labels[int(i)] for i in y.tolist())
    if not probability_batches:
        raise ValueError("The selected data split has no examples.")
    return true_patterns, np.concatenate(probability_batches, axis=0)


def classify_patterns(probabilities, thresholds):
    decisions = probabilities >= np.asarray(thresholds, dtype=np.float32).reshape(1, 2)
    return [events_to_pattern(bool(bits[0]), bool(bits[1])) for bits in decisions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/labels.csv")
    parser.add_argument("--output", default="models/best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path.cwd()
    frame = read_manifest(args.manifest)
    frame["label"] = frame.label.astype(str).str.lower().str.strip()
    unknown = sorted(set(frame.label) - set(PATTERNS))
    if unknown:
        raise SystemExit(f"Unknown event labels in manifest: {unknown}. Expected {PATTERNS}.")
    if frame.patient_id.nunique() < 5:
        raise SystemExit("Need at least 5 distinct patients for grouped train/validation/test splits.")

    # Keep patient identities completely separate across all three partitions.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    trainval_idx, test_idx = next(
        outer.split(frame, frame.label, groups=frame.patient_id)
    )
    trainval = frame.iloc[trainval_idx].reset_index(drop=True)
    test_df = frame.iloc[test_idx].reset_index(drop=True)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed + 1)
    train_idx, val_idx = next(
        inner.split(trainval, trainval.label, groups=trainval.patient_id)
    )
    train_df = trainval.iloc[train_idx].reset_index(drop=True)
    val_df = trainval.iloc[val_idx].reset_index(drop=True)

    train_patients = set(train_df.patient_id)
    val_patients = set(val_df.patient_id)
    test_patients = set(test_df.patient_id)
    if (train_patients & val_patients or train_patients & test_patients
            or val_patients & test_patients):
        raise RuntimeError("Patient leakage detected between data splits.")

    # Stable class-index order for the Dataset; event order is always EVENTS.
    source_labels = sorted(frame.label.unique().tolist())
    print(f"Cycles: total={len(frame)} train={len(train_df)} validation={len(val_df)} test={len(test_df)}")
    print(f"Patients: total={frame.patient_id.nunique()} train={len(train_patients)} validation={len(val_patients)} test={len(test_patients)}")
    print(f"Source labels: {source_labels}; neural event outputs: {EVENTS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = RespiratoryDataset(train_df, root, source_labels)
    val_dataset = RespiratoryDataset(val_df, root, source_labels)
    test_dataset = RespiratoryDataset(test_df, root, source_labels)

    # Precompute each training feature once. Fit descriptor scaling on training
    # patients only, then apply those same statistics to validation and test.
    print("Preparing training features and fitting descriptor scaler...")
    for i in range(len(train_dataset)):
        train_dataset[i]
        if (i + 1) % 500 == 0:
            print(f"Prepared {i + 1}/{len(train_dataset)} training cycles")
    descriptor_matrix = torch.stack([item[1] for item in train_dataset.cache.values()])
    descriptor_mean = descriptor_matrix.mean(dim=0)
    descriptor_std = descriptor_matrix.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_dataset.set_descriptor_scaler(descriptor_mean, descriptor_std)
    val_dataset.set_descriptor_scaler(descriptor_mean, descriptor_std)
    test_dataset.set_descriptor_scaler(descriptor_mean, descriptor_std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Each of the two outputs represents presence/absence of an annotated event.
    train_y = event_targets(train_df.label.tolist())
    positives = train_y.sum(axis=0)
    negatives = len(train_y) - positives
    pos_weight = torch.tensor(
        negatives / np.maximum(positives, 1.0), dtype=torch.float32, device=device
    )
    print(f"Training event counts: crackle={int(positives[0])}, wheeze={int(positives[1])}")
    print(f"Positive-loss weights: {pos_weight.detach().cpu().numpy().round(3).tolist()}")

    model = DualViewRespiratoryNet(n_classes=len(EVENTS)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_thresholds = [0.5, 0.5]

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for mel, desc, class_index in train_loader:
            target_rows = [pattern_to_events(source_labels[int(i)]) for i in class_index.tolist()]
            target = torch.tensor(target_rows, dtype=torch.float32, device=device)
            mel = mel.to(device)
            desc = desc.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(mel, desc), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_patterns, val_probabilities = predict_event_scores(
            model, val_loader, device, source_labels
        )
        val_truth_events = event_targets(val_patterns)
        thresholds = tune_thresholds(val_truth_events, val_probabilities)
        val_predictions = classify_patterns(val_probabilities, thresholds)
        val_score = balanced_accuracy_score(val_patterns, val_predictions)
        val_macro_f1 = f1_score(
            val_patterns, val_predictions, labels=PATTERNS,
            average="macro", zero_division=0
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f} "
            f"validation_balanced_accuracy={val_score:.4f} "
            f"validation_macro_f1={val_macro_f1:.4f} thresholds={np.round(thresholds, 3).tolist()}"
        )

        if val_score > best_score:
            best_score = float(val_score)
            best_thresholds = thresholds
            checkpoint_payload = {
                "state_dict": model.state_dict(),
                "output_type": "multi_label_events",
                "labels": PATTERNS,
                "event_labels": EVENTS,
                "thresholds": [float(x) for x in best_thresholds],
                "n_descriptors": 36,
                "descriptor_mean": descriptor_mean.cpu(),
                "descriptor_std": descriptor_std.cpu(),
                "best_validation_balanced_accuracy": best_score,
                "seed": args.seed,
            }
            # Atomic replacement prevents the live app reading a half-written checkpoint.
            temporary_output = output.with_suffix(output.suffix + ".tmp")
            torch.save(checkpoint_payload, temporary_output)
            temporary_output.replace(output)

    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_patterns, test_probabilities = predict_event_scores(
        model, test_loader, device, source_labels
    )
    test_predictions = classify_patterns(test_probabilities, checkpoint["thresholds"])

    # Event-specific results make it clear if one sound event is being missed.
    truth_events = event_targets(test_patterns)
    predicted_events = test_probabilities >= np.asarray(checkpoint["thresholds"]).reshape(1, 2)
    event_report = {}
    for j, event in enumerate(EVENTS):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth_events[:, j], predicted_events[:, j], average="binary", zero_division=0
        )
        event_report[event] = {
            "threshold": float(checkpoint["thresholds"][j]),
            "balanced_accuracy": float(balanced_accuracy_score(truth_events[:, j], predicted_events[:, j])),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "positive_support": int(truth_events[:, j].sum()),
            "negative_support": int((truth_events[:, j] == 0).sum()),
        }

    report = {
        "warning": "Research metrics only; not clinical validation. Output labels are annotated sound events, not diagnoses. The test set was evaluated after validation-based checkpoint selection.",
        "task": "two independent annotated sound-event outputs (crackle, wheeze)",
        "split": "patient-grouped train/validation/test",
        "recordings": {
            "total": int(len(frame)), "train": int(len(train_df)),
            "validation": int(len(val_df)), "test": int(len(test_df)),
        },
        "patients": {
            "total": int(frame.patient_id.nunique()), "train": len(train_patients),
            "validation": len(val_patients), "test": len(test_patients),
        },
        "source_label_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
        "labels": PATTERNS,
        "event_labels": EVENTS,
        "thresholds": {EVENTS[i]: float(checkpoint["thresholds"][i]) for i in range(len(EVENTS))},
        "best_validation_balanced_accuracy": float(checkpoint["best_validation_balanced_accuracy"]),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_patterns, test_predictions)),
        "test_macro_f1": float(f1_score(test_patterns, test_predictions, labels=PATTERNS, average="macro", zero_division=0)),
        "test_classification_report": classification_report(
            test_patterns, test_predictions, labels=PATTERNS,
            output_dict=True, zero_division=0
        ),
        "test_confusion_matrix_labels": PATTERNS,
        "test_confusion_matrix": confusion_matrix(
            test_patterns, test_predictions, labels=PATTERNS
        ).tolist(),
        "test_event_metrics": event_report,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Final held-out test evaluation saved to reports/evaluation.json")
    print(f"TEST balanced_accuracy={report['test_balanced_accuracy']:.4f} macro_f1={report['test_macro_f1']:.4f}")
    for event, metrics in event_report.items():
        print(f"TEST {event}: balanced_accuracy={metrics['balanced_accuracy']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
