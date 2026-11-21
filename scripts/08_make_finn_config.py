import argparse,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--part",default="xc7z020clg400-1"); ap.add_argument("--clock-ns",type=float,default=10.0)
    ap.add_argument("--target-fps",type=int,default=1000000); a=ap.parse_args()
    cfg={
      "output_dir":"build",
      "synth_clk_period_ns":a.clock_ns,
      "fpga_part":a.part,
      "target_fps":a.target_fps,
      "generate_outputs":["estimate_reports","ooc_synth","stitched_ip","rtlsim_performance"],
      "save_intermediate_models":True
    }
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(cfg,indent=2))
    print("[PASS]",a.out)
    print("Check enum/key spelling against the DataflowBuildConfig in your exact FINN revision before synthesis.")
if __name__=="__main__": main()
