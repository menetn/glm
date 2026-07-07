CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR must be set"
    exit 1
fi

python -u -m main \
  mode=ppl_eval \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=lm1b-wrap \
  algo=ar \
  model.length=128 \
  eval.checkpoint_path=$CHECKPOINT_DIR \
  +wandb.offline=true
