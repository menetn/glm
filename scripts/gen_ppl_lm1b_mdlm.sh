CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR must be set"
    exit 1
fi

python -u -m main \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=64 \
  data=lm1b-wrap \
  model=small \
  model.length=128 \
  algo=mdlm \
  eval.checkpoint_path=$CHECKPOINT_DIR \
  sampling.num_sample_batches=16 \
  +wandb.offline=true
