CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"
CKPT="duo_distilled"
STEPS=2
SEED=42
TEMPERATURE=1.0
DISABLE_EMA=False

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
    algo=duo_base \
    model=small \
    eval.checkpoint_path=$CHECKPOINT_DIR/$CKPT.ckpt \
    sampling.num_sample_batches=2 \
    sampling.steps=$STEPS \
    sampling.predictor=ancestral \
    +wandb.offline=true \
    eval.generated_samples_path=$CHECKPOINT_DIR/samples_ancestral_greedy/$SEED-$STEPS-$CKPT-$TEMPERATURE-disable-ema-$DISABLE_EMA-llama3_1.json \
    sampling.noise_removal=ancestral \
    eval.disable_ema=$DISABLE_EMA \
    sampling.temperature=$TEMPERATURE \
    +algo.use_curriculum=True
