#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp

# ==============================
# Basic settings
# ==============================
DATASET="mnist"
N_SAMPLES=800
NOISE=0.12
EPOCHS=10
BATCH_SIZE=128
VQC_LAYERS=1

N_QUBITS=16
LR=0.001
DIGITS="0,1"
DATA_ROOT="./datasets"
IMG_SIZE=4

# encoder argument只是給資料夾命名
ENCODER_TAG="ensemble"

# ==============================
# Train Hadamard OFF
# ==============================
python train.py \
  --dataset "$DATASET" \
  --n_samples "$N_SAMPLES" \
  --noise "$NOISE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --encoder "$ENCODER_TAG" \
  --n_qubits "$N_QUBITS" \
  --vqc_layers "$VQC_LAYERS" \
  --lr "$LR" \
  --standardize \
  --digits "$DIGITS" \
  --data_root "$DATA_ROOT" \
  --img_size "$IMG_SIZE"

# ==============================
# Train Hadamard ON
# ==============================
python train.py \
  --dataset "$DATASET" \
  --n_samples "$N_SAMPLES" \
  --noise "$NOISE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --encoder "$ENCODER_TAG" \
  --n_qubits "$N_QUBITS" \
  --vqc_layers "$VQC_LAYERS" \
  --lr "$LR" \
  --standardize \
  --digits "$DIGITS" \
  --data_root "$DATA_ROOT" \
  --img_size "$IMG_SIZE" \
  --hadamard

echo "All ensemble experiments completed."