# RuralStethoAI

AI-powered respiratory-sound research prototype for rural screening. This is not a medical device and must not be used to diagnose or rule out pneumonia, asthma, or TB.

## What is innovative?
- **Audio-quality gate:** rejects silence, clipped, too-short, and very quiet recordings before inference.
- **Dual-view neural model:** a CNN learns time-frequency patterns from log-Mel spectrograms; a second branch learns acoustic descriptors (MFCC summaries, RMS, zero-crossing rate, spectral centroid, bandwidth, and rolloff). The representations are fused for sound-pattern classification.
- **Uncertainty-aware abstention:** low-confidence/high-entropy results return `REVIEW REQUIRED` rather than forcing a label. The threshold is a demo placeholder until calibrated on independent clinical data.
- **Patient-group evaluation:** train, validation, and held-out test partitions are separated by patient. The test partition is not used to choose model checkpoints.
- **Rural-ready direction:** supports browser audio recording or WAV upload, local inference after setup, and temporary audio deletion after processing.

## 1. Dataset
Use the ICBHI 2017 Respiratory Sound Database only after reviewing its access and use terms: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge . This project does not download, redistribute, or include recordings. Put your authorized local dataset files (WAV plus matching annotation TXT files) somewhere on disk.

Generate cycle-level sound-pattern labels:
```powershell
python prepare_icbhi.py --dataset-dir "D:\path\to\your\ICBHI\audio_and_txt_files"
```
This creates `data/derived_cycles/` and `data/labels.csv`. Labels (`normal`, `crackle`, `wheeze`, `both`) describe annotated sound events, not diseases. No patient audio is bundled. The Git ignore rules exclude local recordings, generated cycles, manifests, trained weights, and evaluation outputs.

## 2. Setup (Windows PowerShell)
Python 3.12 is a conservative option for ML package compatibility. The project includes a virtual environment only on your computer, not in Git.
```powershell
cd .\stetho
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
If `uv` is unavailable, install Python 3.12 and use `python -m venv .venv`, activate it, then run `python -m pip install -r requirements.txt`.

## 3. Train and evaluate
```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m rural_stetho.train --manifest data/labels.csv --output models/best.pt
```
The trainer writes `reports/evaluation.json`, including validation-based checkpoint selection, held-out test balanced accuracy, macro-F1, per-class metrics, and a confusion matrix. These are research metrics, not clinical validation. Inspect per-class sensitivity, class balance, and patient split; do not report accuracy alone.

## 4. Launch the demo
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Record using a compatible connected audio input or upload a WAV file. An ordinary phone/laptop microphone is not clinically equivalent to a digital stethoscope. Without a trained checkpoint, the app performs only the audio-quality check and explicitly says that no prediction was generated.

## Safety, limits, and privacy
- Lung-sound patterns are not equivalent to pneumonia, asthma, or TB diagnoses.
- TB cannot be reliably ruled in or out using lung sounds alone; follow established clinical screening and confirmatory pathways.
- A negative or uncertain model output must never delay medical care.
- Obtain consent, de-identify metadata, restrict access, and delete recordings when no longer needed.
- Clinical use requires prospective external validation, calibration, bias analysis across devices/populations, privacy/security review, and regulatory review.
