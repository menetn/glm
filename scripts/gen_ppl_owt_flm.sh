CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"
STEPS=1024

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR must be set"
    exit 1
fi

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=1024 \
      data=openwebtext-split \
      algo=flm \
      eval.checkpoint_path=$CHECKPOINT_DIR \
      loader.batch_size=2 \
      loader.eval_batch_size=16 \
      sampling.num_sample_batches=1 \
      sampling.steps=$STEPS \
      algo.double_temb=False \
      eval.disable_ema=False \
      +wandb.offline=true \
