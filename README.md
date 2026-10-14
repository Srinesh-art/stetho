# RuralStethoAI

AI-powered respiratory-sound research prototype for rural screening. Not a medical device; do not use to diagnose or rule out pneumonia, asthma, or TB.

## Innovation
- Audio-quality gate rejects silence, clipping, and unusable recordings.
- Dual-view neural network fuses log-Mel spectrogram CNN features with acoustic descriptors.
- Uncertainty-aware abstention returns REVIEW REQUIRED rather than forcing a prediction.
- Patient-group evaluation to reduce leakage between train and test data.
- Offline-first inference and privacy-conscious temporary audio handling.

## Data
Use the public ICBHI 2017 Respiratory Sound Database subject to its terms: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
Organize permitted files under data/audio and create data/labels.csv with columns path,label,patient_id. Example labels should describe sound patterns such as normal, crackle, wheeze, both, or other when supported by the dataset. These are not disease diagnoses.

## Setup
PowerShell:
```powershell
cd D:\Jarvis\RuralStethoAI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = 'src'
python -m rural_stetho.train --manifest data/labels.csv --output models/best.pt
streamlit run app.py
```

TB cannot be reliably ruled in or out from lung sounds alone. Clinical use requires external prospective validation, calibration, bias analysis, privacy/security review and regulatory review. Never let a model result delay urgent care. No patient recordings or trained weights should be pushed to a public repository.