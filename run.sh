export TMPDIR=/tmp
export TEMP=/tmp
export TMP=/tmp

# ==============================
# Basic settings
# ==============================
DATASET="two_moons"
N_SAMPLES=800
NOISE=0.12
EPOCHS=60
BATCH_SIZE=128
VQC_LAYERS=2
N_QUBITS=2
LR=0.001

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
        --standardize
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
        --hadamard False
done

echo "All synthetic experiments completed."