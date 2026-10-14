import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from .data import read_manifest, RespiratoryDataset
from .model import DualViewRespiratoryNet


def predict(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for mel, desc, y in loader:
            logits=model(mel.to(device),desc.to(device))
            ps.extend(logits.argmax(1).cpu().tolist()); ys.extend(y.tolist())
    return ys,ps


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='data/labels.csv')
    parser.add_argument('--output',default='models/best.pt')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    root=Path.cwd(); frame=read_manifest(args.manifest)
    if frame.patient_id.nunique()<5:
        raise SystemExit('Need at least 5 distinct patients for train/validation/test patient-group splits.')
    labels=sorted(frame.label.astype(str).unique().tolist())
    if len(labels)<2: raise SystemExit('Need at least two label classes.')
    # 64/16/20 approx, split by patient so a patient never leaks across sets.
    outer=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    trainval_idx,test_idx=next(outer.split(frame,frame.label,groups=frame.patient_id))
    trainval=frame.iloc[trainval_idx].reset_index(drop=True); test_df=frame.iloc[test_idx].reset_index(drop=True)
    inner=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=43)
    train_idx,val_idx=next(inner.split(trainval,trainval.label,groups=trainval.patient_id))
    train_df=trainval.iloc[train_idx].reset_index(drop=True); val_df=trainval.iloc[val_idx].reset_index(drop=True)
    train_patients=set(train_df.patient_id); val_patients=set(val_df.patient_id); test_patients=set(test_df.patient_id)
    if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
        raise RuntimeError('Patient leakage detected between data splits.')
    print(f'Recordings: total={len(frame)} train={len(train_df)} validation={len(val_df)} test={len(test_df)}')
    print(f'Patients: train={len(train_patients)} validation={len(val_patients)} test={len(test_patients)}; labels={labels}')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_loader=DataLoader(RespiratoryDataset(train_df,root,labels),batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(RespiratoryDataset(val_df,root,labels),batch_size=args.batch_size,shuffle=False)
    test_loader=DataLoader(RespiratoryDataset(test_df,root,labels),batch_size=args.batch_size,shuffle=False)
    model=DualViewRespiratoryNet(len(labels)).to(device)
    counts=train_df.label.astype(str).value_counts()
    weights=torch.tensor([len(train_df)/(len(labels)*max(1,int(counts.get(label,0)))) for label in labels],dtype=torch.float32,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    best=-1.0; output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for mel,desc,y in train_loader:
            mel,desc,y=mel.to(device),desc.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(mel,desc),y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step(); losses.append(loss.item())
        val_y,val_pred=predict(model,val_loader,device)
        val_score=balanced_accuracy_score(val_y,val_pred)
        print(f'Epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.4f} validation_balanced_accuracy={val_score:.4f}')
        if val_score>best:
            best=val_score
            torch.save({'state_dict':model.state_dict(),'labels':labels,'n_descriptors':36,'best_validation_balanced_accuracy':best},output)
    checkpoint=torch.load(output,map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    # Test set is evaluated once after model selection; never used to choose epochs.
    test_y,test_pred=predict(model,test_loader,device)
    report={
      'warning':'Research metrics only; not clinical validation. Test set evaluated after validation-based checkpoint selection.',
      'split':'patient-grouped train/validation/test',
      'recordings':{'total':int(len(frame)),'train':int(len(train_df)),'validation':int(len(val_df)),'test':int(len(test_df))},
      'patients':{'total':int(frame.patient_id.nunique()),'train':len(train_patients),'validation':len(val_patients),'test':len(test_patients)},
      'labels':labels,
      'best_validation_balanced_accuracy':float(best),
      'test_balanced_accuracy':float(balanced_accuracy_score(test_y,test_pred)),
      'test_macro_f1':float(f1_score(test_y,test_pred,labels=list(range(len(labels))),average='macro',zero_division=0)),
      'test_classification_report':classification_report(test_y,test_pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),
      'test_confusion_matrix':confusion_matrix(test_y,test_pred,labels=list(range(len(labels)))).tolist()
    }
    Path('reports').mkdir(exist_ok=True)
    Path('reports/evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Final test evaluation saved to reports/evaluation.json')

if __name__=='__main__': main()
