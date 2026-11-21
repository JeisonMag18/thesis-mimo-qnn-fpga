import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import NP_DEFAULT,ND_PER_BLOCK,SNR_DB_LIST
from mimo_system import simulate_block,detect_zf,detect_mmse,symbols_to_labels

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--blocks-per-snr",type=int,default=2000)
    ap.add_argument("--out",type=Path,default=Path("results/baselines")); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); rows=[]
    for si,snr in enumerate(SNR_DB_LIST):
        rng=np.random.default_rng(50000+si); ez=em=vz=vm=tb=tv=0
        for _ in range(a.blocks_per_snr):
            b=simulate_block(snr,rng,NP_DEFAULT,ND_PER_BLOCK)
            z=detect_zf(b["y"],b["h_hat"]); m=detect_mmse(b["y"],b["h_hat"],b["sigma2"])
            ez+=np.count_nonzero(z!=b["x"]); em+=np.count_nonzero(m!=b["x"]); tb+=b["x"].size
            vz+=np.count_nonzero(symbols_to_labels(z)==b["labels"])
            vm+=np.count_nonzero(symbols_to_labels(m)==b["labels"]); tv+=len(b["labels"])
        rows.append((snr,ez/tb,em/tb,vz/tv,vm/tv)); print(rows[-1])
    with (a.out/"zf_mmse_by_snr.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["snr_db","zf_ber","mmse_ber","zf_vector_acc","mmse_vector_acc"]); w.writerows(rows)
    plt.figure()
    plt.semilogy([r[0] for r in rows],[r[1] for r in rows],marker="o",label="ZF")
    plt.semilogy([r[0] for r in rows],[r[2] for r in rows],marker="s",label="MMSE")
    plt.xlabel("SNR (dB)"); plt.ylabel("BER"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(a.out/"ber_zf_mmse.png",dpi=200); plt.close()
    print("[PASS] baselines")
if __name__=="__main__":
    main()
