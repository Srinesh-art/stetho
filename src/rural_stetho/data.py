from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from .audio import load_audio, features


class RespiratoryDataset(Dataset):
    """Cycle dataset with per-process feature caching for practical CPU training.

    The strict audio-quality gate is intentionally reserved for live inference;
    annotated research recordings can contain clipping and should not silently
    crash the entire training run.
    """
    def __init__(self, frame, root, labels):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.mapping = {v: i for i, v in enumerate(labels)}
        self.cache = {}
        self.descriptor_mean = None
        self.descriptor_std = None

    def set_descriptor_scaler(self, mean, std):
        self.descriptor_mean = torch.as_tensor(mean, dtype=torch.float32)
        self.descriptor_std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        path = Path(row.path)
        if not path.is_absolute():
            path = self.root / path
        key = str(path)
        if key not in self.cache:
            y = load_audio(key)
            if y.size == 0:
                raise ValueError(f"Empty audio: {path}")
            mel, desc = features(y)
            self.cache[key] = (torch.from_numpy(mel), torch.from_numpy(desc))
        mel, desc = self.cache[key]
        if self.descriptor_mean is not None and self.descriptor_std is not None:
            desc = (desc - self.descriptor_mean) / self.descriptor_std
        return mel, desc, self.mapping[str(row.label)]


def read_manifest(path):
    f = pd.read_csv(path)
    required = {'path', 'label', 'patient_id'}
    if required - set(f.columns):
        raise ValueError('CSV needs path,label,patient_id columns')
    if f.empty or f[list(required)].isna().any().any():
        raise ValueError('Manifest is empty or has blank required fields')
    return f
