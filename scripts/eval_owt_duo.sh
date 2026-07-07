CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR must be set"
    exit 1
fi

python -u -m main \
  mode=ppl_eval \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  model=small \
  algo=duo_base \
  eval.checkpoint_path=$CHECKPOINT_DIR \
  sampling.num_sample_batches=0 \
  +wandb.offline=true
