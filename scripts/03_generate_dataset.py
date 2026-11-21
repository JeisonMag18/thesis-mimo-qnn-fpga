import argparse, json, math
from pathlib import Path
import numpy as np
from config import *
from mimo_system import simulate_block

SIZES={"train":TRAIN_SAMPLES,"val":VAL_SAMPLES,"test":TEST_SAMPLES}

def build(n,seed):
    rng=np.random.default_rng(seed)
    nb=math.ceil(n/ND_PER_BLOCK)
    snrs=np.resize(np.asarray(SNR_DB_LIST,dtype=np.float32),nb); rng.shuffle(snrs)
    X=np.empty((nb*ND_PER_BLOCK,12),np.float32); y=np.empty(nb*ND_PER_BLOCK,np.int64)
    ss=np.empty(nb*ND_PER_BLOCK,np.float32); bid=np.empty(nb*ND_PER_BLOCK,np.int32)
    ht=np.empty((nb,2,2),np.complex64); hh=np.empty_like(ht); nm=np.empty(nb)
    p=0
    for k,snr in enumerate(snrs):
        b=simulate_block(float(snr),rng,NP_DEFAULT,ND_PER_BLOCK); sl=slice(p,p+ND_PER_BLOCK)
        X[sl]=b["features"]; y[sl]=b["labels"]; ss[sl]=snr; bid[sl]=k
        ht[k]=b["h"]; hh[k]=b["h_hat"]; nm[k]=b["nmse"]; p+=ND_PER_BLOCK
    return dict(X=X[:n],y=y[:n],snr_db=ss[:n],block_id=bid[:n],h_true=ht,h_hat=hh,nmse=nm)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=Path("data"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    manifest={
      "Np":NP_DEFAULT,"Nd_per_block":ND_PER_BLOCK,"snr_db":list(SNR_DB_LIST),
      "feature_order":["Re_y1","Re_y2","Im_y1","Im_y2","Re_H11","Re_H12","Re_H21","Re_H22","Im_H11","Im_H12","Im_H21","Im_H22"],
      "class_map":{"0":[-1,-1],"1":[-1,1],"2":[1,-1],"3":[1,1]},
      "note":"Balanced shuffled block-level SNR assignment is a reconstruction choice."
    }
    for split,n in SIZES.items():
        d=build(n,DATASET_SEEDS[split]); np.savez_compressed(a.out/f"{split}.npz",**d)
        manifest[split]={"samples":n,"seed":DATASET_SEEDS[split],"channel_realizations":len(d["h_true"])}
        print(split,d["X"].shape)
    (a.out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("[PASS] dataset")
if __name__=="__main__":
    main()
