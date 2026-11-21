import argparse,csv
from pathlib import Path
import numpy as np, torch
from config import *
from models import arch_name,build_quant_mlp,count_params_and_macs
from train_utils import *
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=Path("data"))
    ap.add_argument("--out",type=Path,default=Path("results/qat")); ap.add_argument("--models",type=Path,default=Path("models"))
    ap.add_argument("--epochs",type=int,default=TRAINING["epochs"]); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); a.models.mkdir(parents=True,exist_ok=True)
    Xtr,ytr=load_split(a.data/"train.npz"); Xv,yv=load_split(a.data/"val.npz")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows=[]
    for hidden in QAT_ARCHITECTURES:
        base=arch_name(hidden); pars,macs=count_params_and_macs(hidden)
        for wb,ab in QAT_PRECISIONS:
            tag=f"{base}_w{wb}a{ab}"; bers=[]; accs=[]
            for seed in MODEL_SEEDS:
                print(f"\n=== QAT {tag} seed={seed} ==="); set_seed(seed); model=build_quant_mlp(hidden,wb,ab)
                tr=loader(Xtr,ytr,TRAINING["batch_size"],True,seed); va=loader(Xv,yv,TRAINING["batch_size"],False,seed)
                ck=a.models/f"qat_{tag}_seed{seed}_best.pt"
                hist,ber,acc=train(model,tr,va,a.epochs,TRAINING["lr"],dev,ck)
                save_history(a.out/f"history_{tag}_seed{seed}.csv",hist); bers.append(ber); accs.append(acc)
            rows.append([base,"-".join(map(str,hidden)),wb,ab,pars,macs,np.mean(bers),np.std(bers,ddof=1),np.mean(accs),np.std(accs,ddof=1)])
    with (a.out/"qat_summary.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["architecture","hidden","w_bits","a_bits","params","macs","ber_mean","ber_std","acc_mean","acc_std"]); w.writerows(rows)
if __name__=="__main__": main()
