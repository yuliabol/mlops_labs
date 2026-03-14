import os
import json
import pytest


def test_artifacts_existence():
    assert os.path.exists("data/models/model.pkl"), "Model file missing"
    assert os.path.exists("metrics.json"), "Metrics file missing"
    assert os.path.exists("confusion_matrix.png"), "Confusion matrix image missing"


def test_quality_gate():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    # Threshold for F1 score (adjust based on requirements)
    f1_threshold = float(os.getenv("F1_THRESHOLD", "0.2"))
    assert (
        metrics["test_f1"] >= f1_threshold
    ), f"F1 score {metrics['test_f1']} is below threshold {f1_threshold}"


def test_roc_auc_gate():
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    # Threshold for ROC-AUC
    roc_auc_threshold = float(os.getenv("ROC_AUC_THRESHOLD", "0.75"))
    assert (
        metrics["test_roc_auc"] >= roc_auc_threshold
    ), f"ROC-AUC {metrics['test_roc_auc']} is below threshold {roc_auc_threshold}"
