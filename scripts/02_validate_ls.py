import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import NP_DEFAULT, PILOT_SWEEP, SNR_DB_LIST
from mimo_system import simulate_pilot_estimation, nmse_per_realization

def mean_nmse(snr, npilots, blocks, seed):
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(blocks):
        h,hh,*_=simulate_pilot_estimation(snr,rng,npilots)
        vals.append(nmse_per_realization(h,hh))
    a=np.asarray(vals)
    return float(a.mean()), float(a.std(ddof=1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--blocks",type=int,default=10000)
    ap.add_argument("--pilot-snr",type=float,default=12.0)
    ap.add_argument("--out",type=Path,default=Path("results/ls"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,snr in enumerate(SNR_DB_LIST):
        m,s=mean_nmse(snr,NP_DEFAULT,a.blocks,10000+i)
        rows.append((snr,NP_DEFAULT,m,s)); print(snr,m)
    with (a.out/"nmse_vs_snr.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["snr_db","npilots","nmse_mean","nmse_std"]); w.writerows(rows)
    plt.figure(); plt.semilogy([r[0] for r in rows],[r[2] for r in rows],marker="o")
    plt.xlabel("SNR (dB)"); plt.ylabel("NMSE"); plt.grid(True); plt.tight_layout()
    plt.savefig(a.out/"nmse_vs_snr.png",dpi=200); plt.close()
    prows=[]
    for i,npilots in enumerate(PILOT_SWEEP):
        m,s=mean_nmse(a.pilot_snr,npilots,a.blocks,20000+i)
        prows.append((a.pilot_snr,npilots,m,s)); print(npilots,m)
    with (a.out/"nmse_vs_pilots.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["snr_db","npilots","nmse_mean","nmse_std"]); w.writerows(prows)
    plt.figure(); plt.semilogy([r[1] for r in prows],[r[2] for r in prows],marker="o")
    plt.xlabel("Pilot length Np"); plt.ylabel("NMSE"); plt.grid(True); plt.tight_layout()
    plt.savefig(a.out/"nmse_vs_pilots.png",dpi=200); plt.close()
    print("[PASS]",a.out)
if __name__=="__main__":
    main()
