CHECKPOINT_DIR="${1:-YOUR_CHECKPOINT_DIR}"
SAMPLER="${2:-meanflow}"
NUM_STEPS="${3:-10}"
SEED="${4}"

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ] || [ -z "$SAMPLER" ] || [ -z "$NUM_STEPS" ] || [ -z "$SEED" ]; then
    echo "Usage: $0 <checkpoint_dir> <sampler> <num_steps> <seed>"
    exit 1
fi

python -u -m main \
  mode=sample_eval \
  seed=$SEED \
  model=small \
  model.length=128 \
  algo=duo_finetune \
  algo.use_curriculum=True \
  eval.checkpoint_path=$CHECKPOINT_DIR \
  loader.batch_size=2 \
  loader.eval_batch_size=64 \
  data=lm1b-wrap \
  sampling.num_sample_batches=16 \
  sampling.noise_removal=$SAMPLER \
  training.pred_type=x0 \
  sampling.steps=$NUM_STEPS \
  training.loss_type=$SAMPLER \
  +wandb.offline=true
