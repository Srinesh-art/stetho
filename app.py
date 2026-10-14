from pathlib import Path
import sys
import tempfile
import streamlit as st
import torch
ROOT=Path(__file__).parent; sys.path.insert(0,str(ROOT/'src'))
from rural_stetho.audio import load_audio,quality_check,features
from rural_stetho.model import DualViewRespiratoryNet
st.set_page_config(page_title='RuralStethoAI',page_icon='🫁',layout='centered')
st.title('🫁 RuralStethoAI')
st.caption('AI-assisted respiratory sound research prototype — not a diagnostic device')
st.warning('This tool cannot diagnose or rule out pneumonia, asthma, or TB. Severe breathlessness, blue lips, confusion, chest pain, coughing blood, or rapidly worsening symptoms require urgent medical care.')
st.markdown('### Capture or upload a recording')
st.caption('Use a compatible digital stethoscope or approved microphone as the selected input. Ordinary laptop/phone microphones are not clinically equivalent to a stethoscope.')
recording=st.audio_input('Record respiratory sounds')
upload=st.file_uploader('Or upload a WAV recording',type=['wav'])
source=recording if recording is not None else upload
checkpoint=ROOT/'models'/'best.pt'
if source is not None:
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(prefix='rural_stetho_',suffix='.wav',delete=False) as handle:
            handle.write(source.getvalue())
            tmp=Path(handle.name)
        y=load_audio(str(tmp)); ok,reasons,duration=quality_check(y)
        st.write({'duration_seconds':round(duration,2),'quality_gate':'PASS' if ok else 'REJECT'})
        if not ok:
            st.error('Recording rejected: '+'; '.join(reasons))
        elif not checkpoint.exists():
            st.info('Audio quality passed, but no trained model is present. No prediction has been generated. Prepare authorized labelled data and train first.')
        else:
            mel,desc=features(y); ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if 'descriptor_mean' in ck and 'descriptor_std' in ck:
                desc=(desc-ck['descriptor_mean'].cpu().numpy())/ck['descriptor_std'].cpu().numpy().clip(1e-6)
            model=DualViewRespiratoryNet(2 if ck.get('output_type')=='multi_label_events' else len(ck['labels']),ck.get('n_descriptors',36)); model.load_state_dict(ck['state_dict']); model.eval()
            with torch.inference_mode():
                logits=model(torch.from_numpy(mel).unsqueeze(0),torch.from_numpy(desc).unsqueeze(0))[0]
            st.subheader('Research sound-pattern scores')
            if ck.get('output_type')=='multi_label_events':
                event_probs=torch.sigmoid(logits)
                event_names=ck.get('event_labels',['crackle','wheeze'])
                thresholds=ck.get('thresholds',[0.5,0.5])
                st.bar_chart({label:float(event_probs[i]) for i,label in enumerate(event_names)})
                decisions=[bool(float(event_probs[i])>=float(thresholds[i])) for i in range(2)]
                pattern='both' if all(decisions) else 'crackle' if decisions[0] else 'wheeze' if decisions[1] else 'normal'
                near_threshold=any(abs(float(event_probs[i])-float(thresholds[i]))<0.08 for i in range(2))
                st.write('Independent event scores (uncalibrated):')
                st.write({event_names[i]:{'score':round(float(event_probs[i]),3),'research_threshold':float(thresholds[i])} for i in range(2)})
                if near_threshold:
                    st.warning('REVIEW REQUIRED — at least one sound-event score is near its research threshold. Do not treat this as a negative screen.')
                else:
                    st.info(f'Predicted annotated sound pattern: {pattern}. This is not a disease diagnosis or a clinically validated screening result.')
                st.caption('Thresholds were selected on validation data for this research run and are not medically calibrated.')
            else:
                probs=torch.softmax(logits,dim=0)
                conf,idx=probs.max(0); entropy=float(-(probs*torch.log(probs+1e-8)).sum())
                st.bar_chart({label:float(probs[i]) for i,label in enumerate(ck['labels'])})
                if float(conf)<.60 or entropy>.90:
                    st.warning('REVIEW REQUIRED — uncertain output; do not treat as a negative screen.')
                else:
                    st.info(f'Research sound-pattern output: {ck["labels"][int(idx)]} (uncalibrated score {float(conf):.1%}); not a disease diagnosis.')
                st.caption('Confidence thresholds are placeholders, not clinically calibrated thresholds.')
    except Exception as e:
        st.error(f'Audio processing failed: {e}')
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
report_path=ROOT/'reports'/'evaluation.json'
if report_path.exists():
    import json
    report=json.loads(report_path.read_text(encoding='utf-8'))
    st.subheader('Held-out research evaluation')
    c1,c2,c3=st.columns(3)
    c1.metric('Test balanced accuracy', f"{report['test_balanced_accuracy']:.1%}")
    c2.metric('Test macro F1', f"{report['test_macro_f1']:.1%}")
    c3.metric('Test cycles', str(report['recordings']['test']))
    st.caption('Research metrics only; not clinical validation.')
    st.json(report['test_classification_report'])
st.subheader('Innovation')
st.markdown('- Quality gate rejects silence, clipped, too-short and very quiet recordings.\n- Dual-view neural network combines log-Mel CNN features and acoustic descriptors.\n- Uncertainty-aware abstention prompts human review.\n- Local inference after setup; temporary audio is removed after processing.')
st.caption('Clinical use requires external prospective validation, calibration, bias analysis, privacy/security and regulatory review. Never use this demo to make treatment decisions.')
