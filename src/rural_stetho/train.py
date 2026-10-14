import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix,balanced_accuracy_score,f1_score
from .data import read_manifest,RespiratoryDataset
from .model import DualViewRespiratoryNet

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',default='data/labels.csv'); p.add_argument('--output',default='models/best.pt'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=16); a=p.parse_args()
 random.seed(42); np.random.seed(42); torch.manual_seed(42); root=Path.cwd(); frame=read_manifest(a.manifest)
 if frame.patient_id.nunique()<3: raise SystemExit('Need at least 3 distinct patients for a minimally useful patient-group split.')
 labels=sorted(frame.label.astype(str).unique().tolist())
 if len(labels)<2: raise SystemExit('Need at least two label classes.')
 tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(frame,frame.label,groups=frame.patient_id))
 train_df,test_df=frame.iloc[tr],frame.iloc[te]
 if set(train_df.patient_id)&set(test_df.patient_id): raise RuntimeError('Patient leakage detected')
 print('Recordings:',len(frame),'Patients:',frame.patient_id.nunique(),'Labels:',labels)
 device='cuda' if torch.cuda.is_available() else 'cpu'; train=DataLoader(RespiratoryDataset(train_df,root,labels),batch_size=a.batch_size,shuffle=True); test=DataLoader(RespiratoryDataset(test_df,root,labels),batch_size=a.batch_size)
 model=DualViewRespiratoryNet(len(labels)).to(device); counts=train_df.label.value_counts(); weights=torch.tensor([len(train_df)/(len(labels)*counts.get(x,1)) for x in labels],dtype=torch.float32,device=device); loss_fn=nn.CrossEntropyLoss(weight=weights); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3); best=-1
 for epoch in range(a.epochs):
  model.train(); losses=[]
  for mel,desc,y in train:
   mel,desc,y=mel.to(device),desc.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(mel,desc),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); losses.append(loss.item())
  model.eval(); ys=[]; ps=[]
  with torch.no_grad():
   for mel,desc,y in test: ps.extend(model(mel.to(device),desc.to(device)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
  score=balanced_accuracy_score(ys,ps); print(f'Epoch {epoch+1}/{a.epochs} loss={np.mean(losses):.4f} balanced_accuracy={score:.4f}')
  if score>best:
   best=score; out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'labels':labels,'n_descriptors':36},out)
 model.load_state_dict(torch.load(a.output,map_location=device,weights_only=True)['state_dict']); model.eval(); ys=[]; ps=[]
 with torch.no_grad():
  for mel,desc,y in test: ps.extend(model(mel.to(device),desc.to(device)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
 report={'warning':'Research split only; not clinical validation.','patients':int(frame.patient_id.nunique()),'recordings':int(len(frame)),'labels':labels,'balanced_accuracy':float(balanced_accuracy_score(ys,ps)),'macro_f1':float(f1_score(ys,ps,average='macro')),'classification_report':classification_report(ys,ps,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(ys,ps).tolist()}
 Path('reports').mkdir(exist_ok=True); Path('reports/evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print('Evaluation saved to reports/evaluation.json')
if __name__=='__main__': main()
