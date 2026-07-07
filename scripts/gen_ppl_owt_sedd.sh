CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"
CKPT="last"
STEPS=32
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
  algo=sedd \
  model=small \
  eval.checkpoint_path=$CHECKPOINT_DIR/$CKPT.ckpt \
  sampling.num_sample_batches=100 \
  sampling.steps=$STEPS \
  sampling.predictor=analytic \
  eval.generated_samples_path=$CHECKPOINT_DIR/$SEED-$STEPS-ckpt-$CKPT.json \
  +wandb.offline=true
