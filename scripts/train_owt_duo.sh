#!/bin/bash

DATA_DIR="YOUR_DATA_DIR"

CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"

python -u -m main \
  checkpointing.save_dir=$CHECKPOINT_DIR \
  checkpointing.resume_from_ckpt=True \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  data.cache_dir=$DATA_DIR \
  wandb.project=owt_full \
  wandb.name=owt_full_duo \
  model=small \
  algo=duo \
  model.length=1024 \
  sampling.num_sample_batches=1 \
  sampling.steps=[1024] \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.55 \
  algo.gamma_max=-1.85 \
  algo.curriculum_start=0 \
  algo.curriculum_end=500000
