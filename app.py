from pathlib import Path
import sys
import streamlit as st
import torch
ROOT=Path(__file__).parent; sys.path.insert(0,str(ROOT/'src'))
from rural_stetho.audio import load_audio,quality_check,features
from rural_stetho.model import DualViewRespiratoryNet
st.set_page_config(page_title='RuralStethoAI',page_icon='🫁')
st.title('🫁 RuralStethoAI')
st.caption('AI-assisted respiratory sound research prototype — not a diagnostic device')
st.warning('This tool cannot diagnose or rule out pneumonia, asthma, or TB. Severe breathlessness, blue lips, confusion, chest pain, coughing blood, or rapidly worsening symptoms require urgent medical care.')
upload=st.file_uploader('Upload a short digital-stethoscope WAV recording',type=['wav'])
checkpoint=ROOT/'models'/'best.pt'
if upload:
 tmp=ROOT/'data'/'_temporary_upload.wav'; tmp.parent.mkdir(exist_ok=True); tmp.write_bytes(upload.getvalue())
 try:
  y=load_audio(str(tmp)); ok,reasons,duration=quality_check(y); st.write({'duration_seconds':round(duration,2),'quality_gate':'PASS' if ok else 'REJECT'})
  if not ok: st.error('Recording rejected: '+'; '.join(reasons))
  elif not checkpoint.exists(): st.info('Audio quality passed, but no trained model is present. No prediction has been generated. Add appropriately licensed labelled data and train first.')
  else:
   mel,desc=features(y); ck=torch.load(checkpoint,map_location='cpu',weights_only=True); model=DualViewRespiratoryNet(len(ck['labels']),ck.get('n_descriptors',36)); model.load_state_dict(ck['state_dict']); model.eval()
   with torch.no_grad(): probs=torch.softmax(model(torch.from_numpy(mel).unsqueeze(0),torch.from_numpy(desc).unsqueeze(0)),dim=1)[0]
   conf,idx=probs.max(0); entropy=float(-(probs*torch.log(probs+1e-8)).sum()); st.bar_chart({label:float(probs[i]) for i,label in enumerate(ck['labels'])})
   if float(conf)<.60 or entropy>.90: st.warning('REVIEW REQUIRED — uncertain output; do not treat as a negative screen.')
   else: st.info(f'Research sound-pattern output: {ck["labels"][int(idx)]} (uncalibrated score {float(conf):.1%}); not a disease diagnosis.')
   st.caption('Confidence thresholds are placeholders, not clinically calibrated thresholds.')
 except Exception as e: st.error(f'Audio processing failed: {e}')
 finally: tmp.unlink(missing_ok=True)
st.subheader('Innovation')
st.markdown('- Quality gate rejects silence, clipped, too-short and very quiet recordings.\n- Dual-view neural network combines log-Mel CNN features and acoustic descriptors.\n- Uncertainty-aware abstention prompts human review.\n- Local inference after setup; temporary upload is removed after processing.')
st.caption('Clinical use requires external prospective validation, calibration, bias analysis, privacy/security and regulatory review. Never use this demo to make treatment decisions.')
