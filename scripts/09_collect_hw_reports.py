import argparse,json
from pathlib import Path
NAMES=["estimate_network_performance.json","estimate_layer_resources.json","estimate_layer_cycles.json","op_and_param_counts.json","ooc_synth_and_timing.json","rtlsim_performance.json"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--build",type=Path,required=True)
    ap.add_argument("--model",required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    out={"model":a.model,"build_dir":str(a.build)}
    for n in NAMES:
        hits=list(a.build.rglob(n))
        out[n]={"path":str(hits[0]),"data":json.loads(hits[0].read_text())} if hits else None
        print("[OK]" if hits else "[--]",n)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)); print("[PASS]",a.out)
if __name__=="__main__": main()
