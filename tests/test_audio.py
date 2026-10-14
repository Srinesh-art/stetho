import numpy as np
from rural_stetho.audio import quality_check, features, SR

def test_empty_audio_is_rejected():
    ok, reasons, duration = quality_check(np.array([], dtype=np.float32))
    assert not ok and reasons and duration == 0

def test_silence_is_rejected():
    ok, reasons, _ = quality_check(np.zeros(SR * 3, dtype=np.float32))
    assert not ok and any('quiet' in x.lower() or 'silent' in x.lower() for x in reasons)

def test_usable_signal_features_have_expected_shapes():
    t = np.arange(SR * 3, dtype=np.float32) / SR
    y = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    ok, reasons, _ = quality_check(y)
    assert ok, reasons
    mel, desc = features(y)
    assert mel.shape[0] == 64
    assert desc.shape == (36,)
    assert np.isfinite(mel).all() and np.isfinite(desc).all()
