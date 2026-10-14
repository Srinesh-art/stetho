import numpy as np
import pandas as pd
import soundfile as sf
from rural_stetho.audio import SR
from rural_stetho.data import RespiratoryDataset, read_manifest

def test_dataset_applies_training_descriptor_scaler(tmp_path):
    audio_path = tmp_path / 'cycle.wav'
    t = np.arange(SR * 3, dtype=np.float32) / SR
    signal = (0.08 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    sf.write(audio_path, signal, SR)
    frame = pd.DataFrame([{'path': str(audio_path), 'label': 'normal', 'patient_id': '101'}])
    dataset = RespiratoryDataset(frame, tmp_path, ['normal'])
    mel, raw_desc, label = dataset[0]
    assert mel.shape[0] == 64 and label == 0
    dataset.set_descriptor_scaler(np.zeros(36, dtype=np.float32), np.full(36, 2.0, dtype=np.float32))
    _, scaled_desc, _ = dataset[0]
    assert np.allclose(scaled_desc.numpy(), raw_desc.numpy() / 2.0)

def test_manifest_requires_expected_columns(tmp_path):
    path = tmp_path / 'labels.csv'
    pd.DataFrame([{'path': 'a.wav', 'label': 'normal'}]).to_csv(path, index=False)
    try:
        read_manifest(path)
    except ValueError as error:
        assert 'path,label,patient_id' in str(error)
    else:
        raise AssertionError('Manifest missing patient_id should be rejected')
