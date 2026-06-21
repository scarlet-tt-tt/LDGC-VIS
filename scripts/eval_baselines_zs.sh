#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/pan/anaconda3/envs/reloc3r/bin/python}
OUT=${OUT:-baseline_eval_results_zs}
DUST3R_CKPT=${DUST3R_CKPT:-/home/pan/下载/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth}
LOFTR_CKPT=${LOFTR_CKPT:-/home/pan/下载/indoor_ds_new.ckpt}
EFFICIENTLOFTR_REPO=${EFFICIENTLOFTR_REPO:-/tmp/EfficientLoFTR}
EFFICIENTLOFTR_CKPT=${EFFICIENTLOFTR_CKPT:-/home/pan/下载/eloftr_outdoor.ckpt}
TUM_ROOT=${TUM_ROOT:-/tmp/TUM}

# RoMa: ARKitScenes, TUM KF4, TUM KF8.
$PY eval_baselines.py --model roma --benchmark arkit --output-dir "$OUT"
$PY eval_baselines.py --model roma --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --output-dir "$OUT"
$PY eval_baselines.py --model roma --benchmark tum --keyframes 8 --dataroot "$TUM_ROOT" --output-dir "$OUT"

# Tiny RoMa: ARKitScenes, TUM KF4, TUM KF8, ScanNet1500.
$PY eval_baselines.py --model tinyroma --benchmark arkit --output-dir "$OUT"
$PY eval_baselines.py --model tinyroma --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --output-dir "$OUT"
$PY eval_baselines.py --model tinyroma --benchmark tum --keyframes 8 --dataroot "$TUM_ROOT" --output-dir "$OUT"
$PY eval_baselines.py --model tinyroma --benchmark scannet1500 --pair-mode adjacent --output-dir "$OUT"

# DUSt3R: ARKitScenes, TUM KF4, TUM KF8, ScanNet1500.
$PY eval_baselines.py --model dust3r --benchmark arkit --dust3r-ckpt "$DUST3R_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model dust3r --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --dust3r-ckpt "$DUST3R_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model dust3r --benchmark tum --keyframes 8 --dataroot "$TUM_ROOT" --dust3r-ckpt "$DUST3R_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model dust3r --benchmark scannet1500 --pair-mode adjacent --dust3r-ckpt "$DUST3R_CKPT" --output-dir "$OUT"

# Reloc3r-S: ARKitScenes, TUM KF4, TUM KF8, ScanNet1500.
$PY eval_baselines.py --model reloc3r_s --benchmark arkit --resolution 224,224 --output-dir "$OUT"
$PY eval_baselines.py --model reloc3r_s --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --output-dir "$OUT"
$PY eval_baselines.py --model reloc3r_s --benchmark tum --keyframes 8 --dataroot "$TUM_ROOT" --output-dir "$OUT"
$PY eval_baselines.py --model reloc3r_s --benchmark scannet1500 --resolution 224,224 --pair-mode adjacent --output-dir "$OUT"

# Reloc3r-L: ARKitScenes, TUM KF4, TUM KF8, ScanNet1500.
$PY eval_baselines.py --model reloc3r_l --benchmark arkit --resolution 512,384 --output-dir "$OUT"
$PY eval_baselines.py --model reloc3r_l --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --output-dir "$OUT"
$PY eval_baselines.py --model reloc3r_l --benchmark tum --keyframes 8 --dataroot "$TUM_ROOT" --output-dir "$OUT"
$PY eval_baselines.py --model reloc3r_l --benchmark scannet1500 --resolution 512,384 --pair-mode adjacent --output-dir "$OUT"

# LoFTR: ARKitScenes, TUM KF4, TUM KF8, ScanNet1500.
$PY eval_baselines.py --model loftr --benchmark arkit --loftr-ckpt "$LOFTR_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model loftr --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --loftr-ckpt "$LOFTR_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model loftr --benchmark tum --keyframes 8 --dataroot "$TUM_ROOT" --loftr-ckpt "$LOFTR_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model loftr --benchmark scannet1500 --pair-mode adjacent --loftr-ckpt "$LOFTR_CKPT" --output-dir "$OUT"

# Efficient LoFTR: ARKitScenes, TUM KF4, TUM KF8, ScanNet1500.
$PY eval_baselines.py --model efficientloftr --benchmark arkit --efficientloftr-repo "$EFFICIENTLOFTR_REPO" --efficientloftr-ckpt "$EFFICIENTLOFTR_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model efficientloftr --benchmark tum --keyframes 4 --dataroot "$TUM_ROOT" --efficientloftr-repo "$EFFICIENTLOFTR_REPO" --efficientloftr-ckpt "$EFFICIENTLOFTR_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model efficientloftr --benchmark tum --keyframes 8 --dataroot "$TUM_ROOT" --efficientloftr-repo "$EFFICIENTLOFTR_REPO" --efficientloftr-ckpt "$EFFICIENTLOFTR_CKPT" --output-dir "$OUT"
$PY eval_baselines.py --model efficientloftr --benchmark scannet1500 --pair-mode adjacent --efficientloftr-repo "$EFFICIENTLOFTR_REPO" --efficientloftr-ckpt "$EFFICIENTLOFTR_CKPT" --output-dir "$OUT"
