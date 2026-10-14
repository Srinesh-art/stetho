import numpy as np
import librosa
SR=16000
SECONDS=5

def load_audio(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    return np.asarray(y,dtype=np.float32)

def quality_check(y):
    if y.size==0: return False,["Empty audio"],0.0
    duration=len(y)/SR
    rms=float(np.sqrt(np.mean(np.square(y,dtype=np.float64))+1e-12))
    clip=float(np.mean(np.abs(y)>=.999))
    reasons=[]
    if duration<2: reasons.append("Recording shorter than 2 seconds")
    if duration>30: reasons.append("Recording longer than 30 seconds; trim it")
    if rms<.003: reasons.append("Audio too quiet or silent")
    if clip>.01: reasons.append("Audio clipping exceeds 1%")
    return not reasons,reasons,duration

def features(y):
    target=SR*SECONDS
    if len(y)>target:
        start=(len(y)-target)//2; y=y[start:start+target]
    else: y=np.pad(y,(0,target-len(y)))
    # Fixed-scale audio; do not normalize quiet recordings upward.
    peak=float(np.max(np.abs(y))) if len(y) else 0
    if peak>1: y=y/peak
    mel=librosa.feature.melspectrogram(y=y,sr=SR,n_fft=1024,hop_length=256,n_mels=64,fmin=50,fmax=4000,power=2)
    mel=np.clip(librosa.power_to_db(mel+1e-10,ref=np.max)/80,-1,0).astype(np.float32)
    mfcc=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=13,n_fft=1024,hop_length=256)
    rms=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)
    zcr=librosa.feature.zero_crossing_rate(y,frame_length=1024,hop_length=256)
    cent=librosa.feature.spectral_centroid(y=y,sr=SR,n_fft=1024,hop_length=256)
    bw=librosa.feature.spectral_bandwidth(y=y,sr=SR,n_fft=1024,hop_length=256)
    roll=librosa.feature.spectral_rolloff(y=y,sr=SR,n_fft=1024,hop_length=256)
    desc=np.concatenate([mfcc.mean(1),mfcc.std(1),[rms.mean(),rms.std(),zcr.mean(),zcr.std(),cent.mean()/SR,cent.std()/SR,bw.mean()/SR,bw.std()/SR,roll.mean()/SR,roll.std()/SR]]).astype(np.float32)
    return mel,desc
