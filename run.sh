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
digits="0,1"    

# ==============================
# Encoder list
# ==============================
ENCODERS=(
    "angle_rx"
    "angle_ry"
    "angle_rz"
)

# ==============================
# Train (Hadamard ON)
# ==============================
for encoder in "${ENCODERS[@]}"; do
    python train.py \
        --dataset "$DATASET" \
        --n_samples $N_SAMPLES \
        --noise $NOISE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --encoder "$encoder" \
        --n_qubits $N_QUBITS \
        --vqc_layers $VQC_LAYERS \
        --lr $LR \
        --standardize \
        --digits "$digits" \
done

# ==============================
# Train (Hadamard OFF)
# ==============================
for encoder in "${ENCODERS[@]}"; do
    python train.py \
        --dataset "$DATASET" \
        --n_samples $N_SAMPLES \
        --noise $NOISE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --encoder "$encoder" \
        --n_qubits $N_QUBITS \
        --vqc_layers $VQC_LAYERS \
        --lr $LR \
        --standardize \
        --digits "$digits" \
        --hadamard False
done

echo "All synthetic experiments completed."