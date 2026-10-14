import numpy as np
import torch
from rural_stetho.model import DualViewRespiratoryNet
from rural_stetho.train import pattern_to_events, events_to_pattern, classify_patterns


def test_sound_events_are_independent_and_can_coexist():
    assert pattern_to_events("normal") == [0.0, 0.0]
    assert pattern_to_events("crackle") == [1.0, 0.0]
    assert pattern_to_events("wheeze") == [0.0, 1.0]
    assert pattern_to_events("both") == [1.0, 1.0]
    assert events_to_pattern(True, True) == "both"
    assert events_to_pattern(False, False) == "normal"


def test_classification_uses_independent_thresholds():
    probabilities = np.array(
        [[0.8, 0.1], [0.1, 0.8], [0.8, 0.8], [0.1, 0.1]],
        dtype=np.float32,
    )
    assert classify_patterns(probabilities, [0.5, 0.5]) == [
        "crackle", "wheeze", "both", "normal"
    ]


def test_neural_model_supports_two_independent_event_logits():
    model = DualViewRespiratoryNet(n_classes=2)
    logits = model(torch.randn(2, 64, 313), torch.randn(2, 36))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
