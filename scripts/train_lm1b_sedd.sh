#!/bin/bash

DATA_DIR="YOUR_DATA_DIR"

CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"

if [ "$DATA_DIR" = "YOUR_DATA_DIR" ] || [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: DATA_DIR and CHECKPOINT_DIR must be set"
    exit 1
fi

python -u -m main \
  checkpointing.save_dir=$CHECKPOINT_DIR \
  checkpointing.resume_from_ckpt=True \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  wandb.project=lm1b_full \
  wandb.name=lm1b_full_sedd \
  model=small \
  algo=sedd \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=[128] \
  sampling.predictor=analytic \
  trainer.max_steps=1000000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000
