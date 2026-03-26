#!/usr/bin/env bash
set -uo pipefail

export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp

# =========================
# Base settings
# =========================
DATASET="mnist"
N_SAMPLES=800
NOISE=0.12
EPOCHS=10
BATCH_SIZE=128
N_QUBITS=10
VQC_LAYERS=1
DIGITS="0,1"
DATA_ROOT="./datasets"
IMG_SIZE=4

# =========================
# Scaler ADV training
# =========================
SCALER_LR=0.001
VQC_LR=0.0001          # only used in full mode; recommend ~1/10 of SCALER_LR
WEIGHT_DECAY=1e-4
GAMMA=0.95

LAMBDA_CLEAN=1.0
LAMBDA_ADV=1.0

OUTDIR="./runs"
SUCCESS_RUNS=()
FAILED_RUNS=()

# Pretrained VQC checkpoints (from run.sh / train.py)
# amplitude (no hadamard)
AMP_PRETRAIN="${OUTDIR}/${DATASET}/amplitude/q${N_QUBITS}_L${VQC_LAYERS}/vqc/checkpoints/best_search.pth"
# amplitude + hadamard
H_AMP_PRETRAIN="${OUTDIR}/${DATASET}/h_amplitude/q${N_QUBITS}_L${VQC_LAYERS}/vqc_h/checkpoints/best_search.pth"

echo "[info] amplitude pretrained:   ${AMP_PRETRAIN}"
echo "[info] h_amplitude pretrained: ${H_AMP_PRETRAIN}"

# ============================================================
# Helper: run one experiment
#   $1 = train_mode  ("scaler_only" | "full")
#   $2 = encoder     ("amplitude"   | "h_amplitude")
#   $3 = train_attack ("fgsm"       | "pgd")
#   $4 = eval_attack  ("fgsm"       | "pgd")
#   $5 = pretrained_path (or "scratch" to skip)
#   $6 = hadamard flag  ("--hadamard" | "")
# ============================================================
run_exp() {
  local TRAIN_MODE="$1"
  local ENCODER="$2"
  local TRAIN_ATTACK="$3"
  local EVAL_ATTACK="$4"
  local PRETRAIN="$5"
  local HADAMARD_FLAG="$6"

  # Attack hyperparams
  local TRAIN_ALPHA
  local TRAIN_STEPS
  local EVAL_ALPHA
  local EVAL_STEPS
  local TRAIN_EPS=0.15
  local EVAL_EPS=0.15

  if [ "$TRAIN_ATTACK" = "fgsm" ]; then
    TRAIN_ALPHA=0.02
    TRAIN_STEPS=1
  else
    TRAIN_ALPHA=0.02
    TRAIN_STEPS=7
  fi

  if [ "$EVAL_ATTACK" = "fgsm" ]; then
    EVAL_ALPHA=0.02
    EVAL_STEPS=1
  else
    EVAL_ALPHA=0.02
    EVAL_STEPS=10
  fi

  local PRETRAIN_ARG=()
  if [ "$PRETRAIN" != "scratch" ]; then
    PRETRAIN_ARG=(--pretrained_path "$PRETRAIN")
  fi

  local LOGDIR="${OUTDIR}/logs"
  mkdir -p "$LOGDIR"

  local TAG="${TRAIN_MODE}_${ENCODER}_train-${TRAIN_ATTACK}_eval-${EVAL_ATTACK}"
  local LOGFILE="${LOGDIR}/${TAG}.log"

  echo ""
  echo "======================================================"
  echo " mode=${TRAIN_MODE}  encoder=${ENCODER}"
  echo " train_attack=${TRAIN_ATTACK}  eval_attack=${EVAL_ATTACK}"
  echo " pretrain=${PRETRAIN}"
  echo " log=${LOGFILE}"
  echo "======================================================"

  python3 trainScaledAmp.py \
    --dataset        "$DATASET" \
    --n_samples      "$N_SAMPLES" \
    --noise          "$NOISE" \
    --epochs         "$EPOCHS" \
    --batch_size     "$BATCH_SIZE" \
    --encoder        "$ENCODER" \
    --n_qubits       "$N_QUBITS" \
    --vqc_layers     "$VQC_LAYERS" \
    --standardize \
    --digits         "$DIGITS" \
    --data_root      "$DATA_ROOT" \
    --img_size       "$IMG_SIZE" \
    --outdir         "$OUTDIR" \
    --train_mode     "$TRAIN_MODE" \
    --scaler_lr      "$SCALER_LR" \
    --vqc_lr         "$VQC_LR" \
    --weight_decay   "$WEIGHT_DECAY" \
    --gamma          "$GAMMA" \
    --lambda_clean   "$LAMBDA_CLEAN" \
    --lambda_adv     "$LAMBDA_ADV" \
    --attack         "$TRAIN_ATTACK" \
    --attack_eps     "$TRAIN_EPS" \
    --attack_alpha   "$TRAIN_ALPHA" \
    --attack_steps   "$TRAIN_STEPS" \
    --eval_attack    "$EVAL_ATTACK" \
    --eval_eps       "$EVAL_EPS" \
    --eval_alpha     "$EVAL_ALPHA" \
    --eval_steps     "$EVAL_STEPS" \
    "${PRETRAIN_ARG[@]}" \
    ${HADAMARD_FLAG:+$HADAMARD_FLAG} \
    2>&1 | tee "$LOGFILE"

  local status=${PIPESTATUS[0]}

  if [ "$status" -ne 0 ]; then
    echo "[error] failed: ${TAG} (exit code=${status})"
    echo "[error] see log: ${LOGFILE}"
    FAILED_RUNS+=("${TAG}")
  else
    echo "[ok] success: ${TAG}"
    SUCCESS_RUNS+=("${TAG}")
  fi

  return 0
}

# ==========================================================
# Experiments
#
# Layout:
#   encoder   | train_mode    | train_atk | eval_atk
#   ----------|---------------|-----------|----------
#   amplitude | scaler_only   | fgsm      | pgd
#   amplitude | scaler_only   | pgd       | pgd
#   amplitude | full          | fgsm      | pgd
#   amplitude | full          | pgd       | pgd
#   h_amplitude | scaler_only | fgsm      | pgd
#   h_amplitude | scaler_only | pgd       | pgd
#   h_amplitude | full        | fgsm      | pgd
#   h_amplitude | full        | pgd       | pgd
# ==========================================================

# ---- amplitude, scaler_only ----
run_exp "scaler_only" "amplitude" "fgsm" "pgd" "$AMP_PRETRAIN"   ""
run_exp "scaler_only" "amplitude" "pgd"  "pgd" "$AMP_PRETRAIN"   ""

# ---- amplitude, full ----
run_exp "full"        "amplitude" "fgsm" "pgd" "$AMP_PRETRAIN"   ""
run_exp "full"        "amplitude" "pgd"  "pgd" "$AMP_PRETRAIN"   ""

# # ---- h_amplitude, scaler_only ----
# run_exp "scaler_only" "h_amplitude" "fgsm" "pgd" "$H_AMP_PRETRAIN" "--hadamard"
# run_exp "scaler_only" "h_amplitude" "pgd"  "pgd" "$H_AMP_PRETRAIN" "--hadamard"

# # ---- h_amplitude, full ----
# run_exp "full"        "h_amplitude" "fgsm" "pgd" "$H_AMP_PRETRAIN" "--hadamard"
# run_exp "full"        "h_amplitude" "pgd"  "pgd" "$H_AMP_PRETRAIN" "--hadamard"

   
echo ""
echo "================ Summary ================"
echo "Success: ${#SUCCESS_RUNS[@]}"
for x in "${SUCCESS_RUNS[@]}"; do
  echo "  [OK] $x"
done

echo ""
echo "Failed: ${#FAILED_RUNS[@]}"
for x in "${FAILED_RUNS[@]}"; do
  echo "  [FAIL] $x"
done

echo ""
echo "All scaled_amp experiments completed."