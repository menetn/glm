CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR must be set"
    exit 1
fi

python -u -m main \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  algo=ar \
  model.length=1024 \
  eval.checkpoint_path=$CHECKPOINT_DIR \
  +wandb.offline=true
