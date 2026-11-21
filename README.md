# Thesis MIMO QNN FPGA

Reproducible scripts for a MIMO 2x2 BPSK neural equalizer with LS channel
estimation, QAT and FINN/Vivado implementation.

## Core assumptions
- MIMO 2x2, BPSK.
- h_ij ~ CN(0, 1/Nt), Nt=2.
- Block fading.
- Orthogonal pilots, Np=8.
- LS channel estimation.
- ZF/MMSE references.
- Neural input: 12 real features.
- Neural output: 4 classes.
- Train/validation/test: 200k / 40k / 40k.
- SNR: 0,4,8,12,16,20 dB.
- Model seeds: 4101,4102,4103.
- Adam, lr=1e-3, batch=1024, epochs=30.

## Run order
From the repository root:

```bash
python3 scripts/00_check_environment.py
python3 scripts/01_smoke_test_ls.py
python3 scripts/02_validate_ls.py --blocks 1000 --out results/ls
python3 scripts/03_generate_dataset.py --out data
python3 scripts/04_eval_zf_mmse.py --blocks-per-snr 100 --out results/baselines
```

Then neural-network steps (PyTorch/Brevitas environment):

```bash
python3 scripts/05_search_fp32.py --data data --out results/fp32 --models models
python3 scripts/06_search_qat.py --data data --out results/qat --models models
```

Export a candidate:

```bash
python3 scripts/07_export_qonnx.py \
  --checkpoint models/qat_h64_w6a6_seed4101_best.pt \
  --hidden 64 --wbits 6 --abits 6 \
  --out export/h64_w6a6.onnx
```

Prepare a FINN build folder:

```bash
mkdir -p hardware/finn_h64_w6a6
cp export/h64_w6a6.onnx hardware/finn_h64_w6a6/model.onnx

python3 scripts/08_make_finn_config.py \
  --out hardware/finn_h64_w6a6/dataflow_build_config.json \
  --part xc7z020clg400-1 \
  --clock-ns 10 \
  --target-fps 1000000
```

The hardware phase targets ESTIMATE_REPORTS, OOC_SYNTH, STITCHED_IP and
RTLSIM_PERFORMANCE. Inspect the FINN revision's DataflowBuildConfig schema
before launching synthesis because configuration field names may evolve.

## Git
```bash
git add .
git commit -m "Add reproducible MIMO QNN thesis pipeline"
git push
```

## Reproducibility note
The pilot sweep (2,4,8,16,32) and balanced block-level SNR assignment are
explicit reconstruction choices. They should not be presented as values that
were specified by the thesis manuscript.
