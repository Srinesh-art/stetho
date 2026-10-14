# StethoSense

StethoSense is a local-first respiratory audio research web app. It accepts a browser recording or WAV upload and returns a cautious **normal-like** or **abnormal-like sound-pattern indication** using the final binary sound classifier.

**Not a medical device.** This tool does not diagnose or rule out asthma, pneumonia, tuberculosis, or any other condition. Do not use its output to make treatment decisions.

## Run locally

Requirements: Windows, Python 3.12, and the model file at `models/binary_sound_classifier.joblib`.

```powershell
cd D:\StethoSense
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`. The included `run_app.bat` launcher can also be used.

## Deployment preparation

The app is prepared for a Streamlit-compatible hosting platform, but **it has not been deployed**.

- Entry point: `app.py`
- Dependencies: `requirements.txt`
- Required trained model artifact: `models/binary_sound_classifier.joblib`
- Audio processing code: `src/rural_stetho/audio.py`

Keep the trained model artifact available to the hosting build. The dataset, training scripts, research reports, and raw audio are not needed for serving predictions. Do not upload private or licensed audio data to a public repository. Check dataset terms before redistributing any source data.

## Audio handling

- Accepts WAV uploads or browser audio recordings.
- Checks duration, silence, and clipping before inference.
- Analyses a fixed-length audio segment using the trained binary sound-pattern model.
- Deletes the temporary audio file after processing.
- Does not maintain an account system, audio history, or user-facing evaluation dashboard.

## Safety and limitations

The output describes similarity to annotated sound patterns, not a disease diagnosis. Normal-like output does not rule out illness. The model has not been clinically validated, and its development evaluation is exploratory. It should not be used as a screening or treatment tool. Public/clinical use requires independent validation, calibration, device and population bias analysis, privacy/security review, and appropriate regulatory review. Seek qualified medical care for symptoms; urgent or worsening symptoms should not be assessed with this app.
