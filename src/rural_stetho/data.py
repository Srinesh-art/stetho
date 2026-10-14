from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from .audio import load_audio,features,quality_check

class RespiratoryDataset(Dataset):
    def __init__(self,frame,root,labels): self.frame=frame.reset_index(drop=True); self.root=root; self.mapping={v:i for i,v in enumerate(labels)}
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]; path=Path(row.path)
        if not path.is_absolute(): path=self.root/path
        y=load_audio(str(path)); ok,reasons,_=quality_check(y)
        if not ok: raise ValueError(f'Unusable audio {path}: {reasons}')
        mel,desc=features(y)
        return torch.from_numpy(mel),torch.from_numpy(desc),self.mapping[str(row.label)]

def read_manifest(path):
    f=pd.read_csv(path); required={'path','label','patient_id'}
    if required-set(f.columns): raise ValueError('CSV needs path,label,patient_id columns')
    if f.empty or f[list(required)].isna().any().any(): raise ValueError('Manifest is empty or has blank required fields')
    return f
