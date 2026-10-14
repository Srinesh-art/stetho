from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rural_stetho.audio import load_audio, quality_check, features

MODEL_PATH = ROOT / "models" / "binary_sound_classifier.joblib"
DECISION_THRESHOLD = 0.37

st.set_page_config(page_title="StethoSense | Respiratory Sound Assistant", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#eaf2f7; --muted:#9bb0c0; --mint:#73e0c0; }
.stApp { background:radial-gradient(ellipse at 15% 0%,#153747 0%,#0a1722 42%,#08121b 100%); color:var(--ink); }
html,body,[class*="css"] { font-family:'DM Sans',sans-serif; }
.block-container { max-width:1120px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
.brandline { color:var(--mint); font-size:.78rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; }
.hero { padding:2.1rem 0 1.4rem; }
.hero h1 { font-size:clamp(2.5rem,6vw,4.6rem); line-height:1.03; margin:.55rem 0 1rem; color:#f4f8fb; }
.hero p { color:#b2c4d0; font-size:1.08rem; max-width:690px; line-height:1.8; }
.panel { background:rgba(16,35,48,.82); border:1px solid rgba(151,190,204,.16); border-radius:22px; padding:1.35rem 1.45rem; }
.panel-label { color:var(--mint); font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; margin-bottom:.65rem; }
.subtle { color:var(--muted); line-height:1.7; font-size:.93rem; }
.status-pill { display:inline-block; border:1px solid rgba(115,224,192,.38); color:#9af0d8; border-radius:99px; padding:.38rem .7rem; font-size:.78rem; font-weight:600; }
.result-card { background:linear-gradient(135deg,rgba(25,67,74,.9),rgba(16,35,48,.92)); border:1px solid rgba(115,224,192,.25); border-radius:22px; padding:1.6rem; margin:1rem 0; }
.result-title { color:#a9bfc9; font-size:.8rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; }
.result-value { color:#f3fbf8; font-family:'Manrope',sans-serif; font-size:clamp(1.8rem,4vw,2.5rem); font-weight:800; margin:.55rem 0; }
div[data-testid="stFileUploader"],div[data-testid="stAudioInput"] { background:rgba(7,20,29,.48); border:1px dashed rgba(151,190,204,.28); border-radius:16px; padding:.7rem; }
.stButton button,.stDownloadButton button { border-radius:12px; font-weight:700; }
div[data-testid="stAlert"] { border-radius:14px; }
hr { border-color:rgba(151,190,204,.15); }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brandline">STETHOSENSE / RESPIRATORY AUDIO</div>
<div class="hero">
<span class="status-pill">Research prototype · Local-ready</span>
<h1>Listen closer.<br/>Understand the sound.</h1>
<p>Explore respiratory recordings with an AI-assisted sound-pattern tool. Add a WAV recording or record directly in your browser to receive a simple, cautious sound-pattern indication.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.25, .75], gap="large")
with left:
    st.markdown('<div class="panel-label">01 / Add a recording</div>', unsafe_allow_html=True)
    st.markdown("### Start with respiratory audio")
    st.markdown('<p class="subtle">Use a compatible digital stethoscope recording where possible. A standard phone or laptop microphone may capture different sound characteristics.</p>', unsafe_allow_html=True)
    recording = st.audio_input("Record respiratory sounds")
    upload = st.file_uploader("Or choose a WAV file", type=["wav"], help="WAV audio, 2–30 seconds.")
    source = recording if recording is not None else upload

with right:
    st.markdown("""
<div class="panel">
<div class="panel-label">02 / What it does</div>
<h3>Sound-pattern assistance</h3>
<p class="subtle">The model estimates whether the recording resembles a normal-labeled sound or contains sound patterns annotated with crackles and/or wheezes.</p>
<hr/>
<div class="panel-label">Privacy by design</div>
<p class="subtle">Audio is processed for this session. The temporary working copy is deleted after analysis; this app does not include an audio history or account system.</p>
<hr/>
<div class="panel-label">Important</div>
<p class="subtle">This is a research prototype, not a medical device. It cannot diagnose, exclude, or treat a disease.</p>
</div>
""", unsafe_allow_html=True)

if source is not None:
    st.divider()
    st.markdown('<div class="panel-label">03 / Recording review</div>', unsafe_allow_html=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="stethosense_", suffix=".wav", delete=False) as handle:
            handle.write(source.getvalue())
            tmp_path = Path(handle.name)
        with st.spinner("Checking audio and reviewing the sound pattern…"):
            audio = load_audio(str(tmp_path))
            valid, reasons, duration = quality_check(audio)

        if not valid:
            st.error("This recording could not be analysed. " + " · ".join(reasons))
            st.info("Try a clear recording between 2 and 30 seconds, without long silence or distorted/clipped audio.")
        elif not MODEL_PATH.exists():
            st.error("The trained model file is missing. Restore models/binary_sound_classifier.joblib and restart the app.")
        else:
            with st.spinner("Analysing the recording…"):
                mel, descriptors = features(audio)
                mel_summary = np.concatenate([mel.mean(axis=1), mel.std(axis=1), np.quantile(mel, [0.10, 0.50, 0.90], axis=1).reshape(-1)])
                vector = np.concatenate([descriptors, mel_summary]).astype(np.float32).reshape(1, -1)
                model = joblib.load(MODEL_PATH)
                abnormal_score = float(model.predict_proba(vector)[0, 1])

            near_boundary = abs(abnormal_score - DECISION_THRESHOLD) < 0.08
            result = "Review recommended" if near_boundary else "Abnormal-like pattern" if abnormal_score >= DECISION_THRESHOLD else "Normal-like pattern"
            st.markdown(f'<div class="result-card"><div class="result-title">Sound-pattern indication</div><div class="result-value">{result}</div><p class="subtle">Recording duration: {duration:.1f} seconds · Analysis complete</p></div>', unsafe_allow_html=True)

            if near_boundary:
                st.warning("The recording is close to the model's internal decision boundary. The result is inconclusive; do not treat it as a normal finding.")
            elif abnormal_score >= DECISION_THRESHOLD:
                st.info("The model detected a pattern more similar to recordings labelled with crackles and/or wheezes. This does not identify a disease. Consider review by a qualified healthcare professional.")
            else:
                st.info("The model found a pattern more similar to recordings labelled normal in its training data. This does not rule out illness or replace medical assessment.")
    except Exception:
        st.error("The recording could not be processed. Please try another clear WAV recording.")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

st.divider()
st.markdown("### Use responsibly")
st.markdown("""
<p class="subtle">StethoSense is intended for research and educational exploration only. Model outputs can be wrong, especially for recordings made with unfamiliar devices or in noisy environments. Do not use this tool to make treatment decisions or delay medical care. Severe breathlessness, blue lips, confusion, chest pain, coughing blood, or rapidly worsening symptoms require urgent medical attention.</p>
<p class="subtle">No recording is stored by the app after analysis. Before any public or clinical use, the system needs independent validation, security and privacy review, and appropriate regulatory review.</p>
""", unsafe_allow_html=True)
st.caption("StethoSense · Respiratory audio research")
