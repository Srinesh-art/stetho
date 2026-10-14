# RuralStethoAI

AI-powered respiratory-sound research prototype for rural screening. This is not a medical device and must not be used to diagnose or rule out pneumonia, asthma, or TB.

## What is innovative?
- **Audio-quality gate:** rejects silence, clipped, too-short, and very quiet recordings before inference.
- **Dual-view neural model:** a convolutional neural network (CNN) learns time-frequency patterns from log-Mel spectrograms; a second branch learns acoustic descriptors (MFCC summaries, RMS, zero-crossing rate, spectral centroid, bandwidth, and rolloff). The two representations are fused for sound-pattern classification.
- **Uncertainty-aware abstention:** low-confidence/high-entropy results return `REVIEW REQUIRED` rather than forcing a label. The threshold is only a demo placeholder until calibration on independent clinical data.
- **Patient-group evaluation:** recordings from a patient are kept in one split to reduce identity leakage.
- **Rural-ready direction:** local inference after setup, lightweight upload workflow, no default audio retention, and clear escalation messaging.

## 1. Dataset
Use the ICBHI 2017 Respiratory Sound Database only after reviewing its access and use terms: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge . This project does not download, redistribute, or include the recordings. Put your authorized local dataset files (WAV plus matching annotation TXT files) somewhere on disk.

Generate cycle-level sound-pattern labels:
```powershell
python prepare_icbhi.py --dataset-dir "D:\path\to\your\ICBHI\audio_and_txt_files"
```
This creates `data/derived_cycles/` and `data/labels.csv`. The target labels (`normal`, `crackle`, `wheeze`, `both`) describe annotated sound events, not diseases. No example audio is bundled. Do not push recordings, manifests containing private metadata, or trained weights to a public repository.

## 2. Setup (Windows PowerShell)
Python 3.12 is a conservative option for ML package compatibility; use a supported PyTorch build for your installed Python version.
```powershell
cd D:\Jarvis\RuralStethoAI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train and evaluate
```powershell
python -m rural_stetho.train --manifest data/labels.csv --output models/best.pt
```
If running from a source checkout, set the import path first:
```powershell
$env:PYTHONPATH = 'src'
python -m rural_stetho.train --manifest data/labels.csv --output models/best.pt
```
The trainer writes `reports/evaluation.json`, including balanced accuracy, macro-F1, per-class metrics, and a confusion matrix. These are internal research metrics only, not clinical validation. Inspect patient split, class balance, per-class sensitivity, and uncertainty; do not report accuracy alone.

## 4. Launch the demo
```powershell
streamlit run app.py
```
Upload a WAV file. Without a trained checkpoint, the app performs only the audio-quality check and explicitly says that no prediction was generated.

## Safety, limits, and privacy
- Lung-sound patterns are not equivalent to pneumonia, asthma, or TB diagnoses.
- TB cannot be reliably ruled in or out using lung sounds alone; follow established clinical screening and confirmatory pathways.
- A negative or uncertain model output must never delay medical care.
- Obtain consent, de-identify metadata, restrict access, and delete recordings when no longer needed.
- Clinical use requires prospective external validation, calibration, bias analysis across devices/populations, privacy/security review, and regulatory review.
