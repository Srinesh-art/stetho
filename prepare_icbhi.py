"""Build a patient-grouped sound-event manifest from a locally obtained ICBHI dataset.

Use only files you are authorized to access. Does not download or redistribute dataset files.
"""
import argparse
from pathlib import Path
import soundfile as sf
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True, help='Directory containing ICBHI .wav and matching annotation .txt files')
    parser.add_argument('--output-dir', default='data/derived_cycles')
    parser.add_argument('--manifest', default='data/labels.csv')
    args = parser.parse_args()
    source = Path(args.dataset_dir).resolve()
    out = Path(args.output_dir).resolve()
    manifest = Path(args.manifest).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    skipped=0
    for wav in sorted(source.rglob('*.wav')):
        ann=wav.with_suffix('.txt')
        if not ann.exists():
            skipped += 1
            continue
        try:
            audio,sr=sf.read(wav, dtype='float32', always_2d=False)
            if audio.ndim > 1:
                audio=audio.mean(axis=1)
            table=pd.read_csv(ann, sep=r'\s+', header=None, names=['start','end','crackle','wheeze'])
            patient_id=wav.name.split('_')[0]
            for i,row in table.iterrows():
                start=max(0,int(float(row.start)*sr)); end=min(len(audio),int(float(row.end)*sr))
                if end <= start:
                    continue
                crackle=bool(int(row.crackle)); wheeze=bool(int(row.wheeze))
                label='both' if crackle and wheeze else 'crackle' if crackle else 'wheeze' if wheeze else 'normal'
                clip=audio[start:end]
                if len(clip)/sr < 2.0:
                    continue
                target=out/f'{wav.stem}_cycle{i:03d}.wav'
                sf.write(target,clip,sr)
                rows.append({'path':str(target.relative_to(Path.cwd())) if target.is_relative_to(Path.cwd()) else str(target), 'label':label, 'patient_id':patient_id})
        except Exception as exc:
            print(f'SKIP {wav.name}: {exc}')
    if not rows:
        raise SystemExit('No usable annotated cycles found. Check dataset directory and matching .txt annotations.')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest,index=False)
    print(f'Wrote {len(rows)} cycles to {manifest}; skipped recordings without annotations: {skipped}')
    print(pd.DataFrame(rows).label.value_counts().to_string())
    print('Important: labels represent annotated sound events, not disease diagnoses.')

if __name__=='__main__': main()
