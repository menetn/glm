CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"
SEED=1

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR must be set"
    exit 1
fi

python -u -m main \
  mode=sample_eval \
  seed=$SEED \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=ar \
  model=small \
  eval.checkpoint_path=$CHECKPOINT_DIR/last.ckpt \
  sampling.num_sample_batches=100 \
  +wandb.offline=true \
  eval.generated_samples_path=$CHECKPOINT_DIR/$SEED-ckpt-last.json
