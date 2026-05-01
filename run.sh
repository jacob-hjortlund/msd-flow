#!/bin/bash
#SBATCH -A m1727
#SBATCH -J msd-flow-train
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --signal=TERM@600
#SBATCH -o /pscratch/sd/h/%u/msd-flow/slurm/%x-%j.out
#SBATCH -e /pscratch/sd/h/%u/msd-flow/slurm/%x-%j.out

set -euo pipefail

# ----------------------------
# User paths
# ----------------------------
export REPO_DIR="$HOME/msd-flow"
export ENV_DIR="$ENV_ROOT/msd-flow"

export PROJECT_ROOT="$PSCRATCH/msd-flow"
export DATA_ROOT="$PROJECT_ROOT/data"
export CACHE_ROOT="$PROJECT_ROOT/caches"
export CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"


mkdir -p "$PROJECT_ROOT/slurm" \
         "$DATA_ROOT" \
         "$CACHE_ROOT" \
         "$CHECKPOINT_ROOT"

# ----------------------------
# Modules / Python environment
# ----------------------------
module load conda
conda activate "$ENV_DIR"

# Avoid importing stale ~/.local packages
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Avoid mixing NERSC CUDA modules with JAX pip CUDA wheels
module unload gpu craype-accel-nvidia80 cudatoolkit nccl cudnn 2>/dev/null || true

export HYDRA_FULL_ERROR=1

# ----------------------------
#  Threading / CPU behavior
# ----------------------------
export SLURM_CPU_BIND=cores

# Important for PyTorch DataLoader + NumPy/SciPy transforms.
# Prevent every worker from spawning many BLAS/OpenMP threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ----------------------------
# JAX behavior
# ----------------------------
# Make JAX fail clearly if CUDA is not available rather than silently using CPU.
export JAX_PLATFORMS=cuda

export XLA_PYTHON_CLIENT_MEM_FRACTION=.85
export JAX_TRACEBACK_FILTERING=off
export NVIDIA_TF32_OVERRIDE=1

# ----------------------------
# Sanity checks
# ----------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Repo: $REPO_DIR"
echo "Env: $ENV_DIR"
echo "Data root: $DATA_ROOT"
echo "Python: $(which python)"

cd "$REPO_DIR"

# ---------------------------------------------------------------------
# Start SOCKS tunnel to Sunrise for ClearML access
# ---------------------------------------------------------------------

SOCKS_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)

echo "Starting SOCKS proxy on 127.0.0.1:${SOCKS_PORT}"

ssh -N \
  -D 127.0.0.1:${SOCKS_PORT} \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  sunrise &

SSH_TUNNEL_PID=$!

cleanup() {
    echo "Stopping SOCKS proxy"
    kill "${SSH_TUNNEL_PID}" 2>/dev/null || true
}

trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

export HTTPS_PROXY="socks5h://127.0.0.1:${SOCKS_PORT}"
export HTTP_PROXY="socks5h://127.0.0.1:${SOCKS_PORT}"
export ALL_PROXY="socks5h://127.0.0.1:${SOCKS_PORT}"

export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"
export all_proxy="$ALL_PROXY"

export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

echo "Checking ClearML API through SOCKS proxy..."

for i in {1..30}; do
    http_code=$(
        curl -sS -I \
          --proxy "$HTTPS_PROXY" \
          --connect-timeout 5 \
          -o /dev/null \
          -w "%{http_code}" \
          https://api.cml.fysik.su.se \
          || true
    )

    if [[ "$http_code" =~ ^[2345][0-9][0-9]$ ]]; then
        echo "ClearML API reachable through SOCKS proxy; HTTP status: ${http_code}"
        break
    fi

    if ! kill -0 "$SSH_TUNNEL_PID" 2>/dev/null; then
        echo "ERROR: SSH SOCKS tunnel died"
        exit 1
    fi

    if [ "$i" -eq 30 ]; then
        echo "ERROR: ClearML API not reachable through SOCKS proxy"
        exit 1
    fi

    sleep 1
done

python - <<'PY'
from clearml.backend_api.session.client import APIClient
APIClient().projects.get_all()
print("ClearML SDK reachable through SOCKS proxy")
PY

# ----------------------------
# Training
# ----------------------------

clearml_overrides=(
  "clearml.enabled=true"
  "clearml.task_name=${SLURM_JOB_NAME}"
  "clearml.use_dataset=false"
)

arcsinh_overrides=(
  "data/transforms@data.dataloader.transforms=arcsinh"
  "data.dataloader.transforms.n_workers=32"
  "data.dataloader.transforms.sample_fraction=0.25"
  "data.dataloader.transforms.percentile=75"
)

data_overrides=(
  "train.data_parallel.enabled=true"
  "train.data_parallel.min_devices=2"

  "data.dataset.data_dir=${DATA_ROOT}"

  "image_size=256"
  "clip_pad_size=512"

  "train.buffer_size=8"
  "train.grad_accum_steps=4"
  "data.dataloader.batch_size=128"
  "train.num_steps_per_epoch=128"
  "data.dataloader.num_workers=8"
  "data.dataloader.prefetch_factor=4"
  
  "${arcsinh_overrides[@]}"
)

#  "train.num_steps_per_epoch=128"

train_overrides=(
  "train.num_epochs=1500"
  "train.ema_decay=0.995"
  "train.optimizer.learning_rate=5e-5"
  "train.checkpoint_dir=${CHECKPOINT_ROOT}"
  "train.resume.restart=false"
  "train.resume.save_on_sigterm=true"
  'train.samples_plot_method="uint8"'
  "train._epoch_metrics_dict.fid_metric.n_real=0"
  "train._epoch_metrics_dict.fid_metric.n_samples=2048"
  "train._epoch_metrics_dict.fid_metric.gen_batch_size=512"  
)

model_overrides=(
  "model.base_channels=64"
  "model.channel_multipliers=[1,2,4,8,16]"
  "model.num_res_blocks=2"
  "model.attn_resolutions=[16,32,64,128]"
  "model.num_heads=8"
  "model.num_groups=32"
  'model.attention_type="dot_product"'
  'model.attention_implementation="cudnn"'
  'model.attention_dtype="${jnp_dtype: bfloat16}"'	
)

overrides=(
  "${clearml_overrides[@]}"
  "${data_overrides[@]}"
  "${train_overrides[@]}"
  "${model_overrides[@]}"  
)

srun --nodes=1 --ntasks=1 --cpu-bind=cores --gpu-bind=none \
  python -u train_model.py "${overrides[@]}"
