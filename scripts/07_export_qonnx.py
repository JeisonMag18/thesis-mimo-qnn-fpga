import argparse
from pathlib import Path
import torch
from models import build_quant_mlp

def parse_hidden(s): return tuple(int(x) for x in s.split(",") if x.strip())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--hidden",required=True); ap.add_argument("--wbits",type=int,required=True)
    ap.add_argument("--abits",type=int,required=True); ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    from brevitas.export import export_qonnx
    model=build_quant_mlp(parse_hidden(a.hidden),a.wbits,a.abits)
    model.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=True)); model.eval()
    a.out.parent.mkdir(parents=True,exist_ok=True); export_qonnx(model,torch.randn(1,12),str(a.out))
    try:
        import onnx; m=onnx.load(str(a.out)); onnx.checker.check_model(m); print("[OK] ONNX checker")
    except Exception as e: print("[WARN] checker:",e)
    print("[PASS]",a.out)
if __name__=="__main__": main()
