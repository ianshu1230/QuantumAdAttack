#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp

# =========================
# Base data / model setting
# =========================
DATASET="mnist"
N_SAMPLES=800
NOISE=0.12
EPOCHS=10
BATCH_SIZE=128
N_QUBITS=16
VQC_LAYERS=1
DIGITS="0,1"
DATA_ROOT="./datasets"
IMG_SIZE=4

# =========================
# Controller ADV training
# =========================
CONTROLLER_LR=0.001
VQC_LR=0.0001          # only used in full mode (end-to-end); recommend ~1/10 of CONTROLLER_LR
TRAIN_MODE="controller_only"   # "controller_only" | "full"
WEIGHT_DECAY=1e-4
GAMMA=0.95

LAMBDA_CLEAN=1.0
LAMBDA_ADV=1.0
LAMBDA_GATE=0.1

TRAIN_ATTACK="pgd"
TRAIN_EPS=0.15
TRAIN_ALPHA=0.02
TRAIN_STEPS=7

EVAL_ATTACK="pgd"
EVAL_EPS=0.15
EVAL_ALPHA=0.02
EVAL_STEPS=10

OUTDIR="./runs"

# ==========================================
# Pretrained checkpoints from train_ensemble.py
# ==========================================
BASE_PRETRAIN="${OUTDIR}/${DATASET}/ensemble/q${N_QUBITS}_L${VQC_LAYERS}/vqc/checkpoints/best_search.pth"
H_PRETRAIN="${OUTDIR}/${DATASET}/ensemble/q${N_QUBITS}_L${VQC_LAYERS}/vqc_h/checkpoints/best_search.pth"

echo "[info] Base pretrained ckpt: ${BASE_PRETRAIN}"
echo "[info] Hadamard pretrained ckpt: ${H_PRETRAIN}"

# =========================
# Controller ADV training: no hadamard
# =========================
python train_controller.py \
  --dataset "$DATASET" \
  --n_samples "$N_SAMPLES" \
  --noise "$NOISE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --encoder "ensemble" \
  --n_qubits "$N_QUBITS" \
  --vqc_layers "$VQC_LAYERS" \
  --standardize \
  --digits "$DIGITS" \
  --data_root "$DATA_ROOT" \
  --img_size "$IMG_SIZE" \
  --pretrained_path "$BASE_PRETRAIN" \
  --outdir "$OUTDIR" \
  --train_mode "$TRAIN_MODE" \
  --controller_lr "$CONTROLLER_LR" \
  --vqc_lr "$VQC_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --gamma "$GAMMA" \
  --lambda_clean "$LAMBDA_CLEAN" \
  --lambda_adv "$LAMBDA_ADV" \
  --lambda_gate "$LAMBDA_GATE" \
  --attack "$TRAIN_ATTACK" \
  --attack_eps "$TRAIN_EPS" \
  --attack_alpha "$TRAIN_ALPHA" \
  --attack_steps "$TRAIN_STEPS" \
  --eval_attack "$EVAL_ATTACK" \
  --eval_eps "$EVAL_EPS" \
  --eval_alpha "$EVAL_ALPHA" \
  --eval_steps "$EVAL_STEPS"

# =========================
# Controller ADV training: hadamard
# =========================
python train_controller.py \
  --dataset "$DATASET" \
  --n_samples "$N_SAMPLES" \
  --noise "$NOISE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --encoder "ensemble" \
  --n_qubits "$N_QUBITS" \
  --vqc_layers "$VQC_LAYERS" \
  --standardize \
  --digits "$DIGITS" \
  --data_root "$DATA_ROOT" \
  --img_size "$IMG_SIZE" \
  --pretrained_path "$H_PRETRAIN" \
  --outdir "$OUTDIR" \
  --train_mode "$TRAIN_MODE" \
  --controller_lr "$CONTROLLER_LR" \
  --vqc_lr "$VQC_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --gamma "$GAMMA" \
  --lambda_clean "$LAMBDA_CLEAN" \
  --lambda_adv "$LAMBDA_ADV" \
  --lambda_gate "$LAMBDA_GATE" \
  --attack "$TRAIN_ATTACK" \
  --attack_eps "$TRAIN_EPS" \
  --attack_alpha "$TRAIN_ALPHA" \
  --attack_steps "$TRAIN_STEPS" \
  --eval_attack "$EVAL_ATTACK" \
  --eval_eps "$EVAL_EPS" \
  --eval_alpha "$EVAL_ALPHA" \
  --eval_steps "$EVAL_STEPS" \
  --hadamard