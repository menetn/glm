#!/bin/bash

# Check arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <algo> <ckpt_path> <steps> <data_dir> [out_dir] [bsub_template]"
    exit 1
fi

ALGO=$1
CKPT_PATH=$2
STEPS=$3
DATA_DIR=$4
OUT_DIR=${5:-"outputs/sweeps"}
BSUB_TEMPLATE=$6

mkdir -p "$OUT_DIR"

# Decide sweep parameter depending on the model type
# For pure continuous flow models (flm), we sweep temperature.
# For discrete/hybrid models (mdlm, sedd, duo, smflm), we sweep p_nucleus.
if [ "$ALGO" = "flm" ]; then
    PARAM_NAME="temperature"
    VALUES=(0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
else
    PARAM_NAME="p_nucleus"
    VALUES=(0.8 0.825 0.85 0.875 0.9 0.925 0.95 0.975 1.0)
fi

echo "Starting evaluation sweep for $ALGO (T = $STEPS steps) using checkpoint $CKPT_PATH"
echo "Sweeping $PARAM_NAME over: ${VALUES[*]}"

for val in "${VALUES[@]}"; do
    echo "Processing: $PARAM_NAME = $val"
    
    # Define JSON output path
    json_path="${OUT_DIR}/${ALGO}_T-${STEPS}_${PARAM_NAME}-${val}.json"
    
    # Configure predictor based on algorithm
    if [ "$ALGO" = "mdlm" ]; then
        PREDICTOR="sampling.predictor=ancestral_cache"
    elif [ "$ALGO" = "sedd" ]; then
        PREDICTOR="sampling.predictor=analytic"
    else
        PREDICTOR=""
    fi

    # Configure command depending on sweep parameter
    if [ "$PARAM_NAME" = "temperature" ]; then
        cmd="python -u -m main \
            mode=sample_eval \
            data=openwebtext-split \
            data.cache_dir=\"$DATA_DIR\" \
            model=small \
            model.length=1024 \
            algo=\"$ALGO\" \
            eval.checkpoint_path=\"$CKPT_PATH\" \
            eval.disable_ema=False \
            eval.compute_generative_perplexity=True \
            eval.perplexity_batch_size=16 \
            loader.batch_size=16 \
            loader.eval_batch_size=16 \
            sampling.num_sample_batches=4 \
            sampling.steps=\"$STEPS\" \
            sampling.temperature=\"$val\" \
            sampling.p_nucleus=1.0 \
            $PREDICTOR \
            eval.generated_samples_path=\"$json_path\" \
            +wandb.offline=true"
    else
        cmd="python -u -m main \
            mode=sample_eval \
            data=openwebtext-split \
            data.cache_dir=\"$DATA_DIR\" \
            model=small \
            model.length=1024 \
            algo=\"$ALGO\" \
            eval.checkpoint_path=\"$CKPT_PATH\" \
            eval.disable_ema=False \
            eval.compute_generative_perplexity=True \
            eval.perplexity_batch_size=16 \
            loader.batch_size=16 \
            loader.eval_batch_size=16 \
            sampling.num_sample_batches=4 \
            sampling.steps=\"$STEPS\" \
            sampling.temperature=1.0 \
            sampling.p_nucleus=\"$val\" \
            $PREDICTOR \
            eval.generated_samples_path=\"$json_path\" \
            +wandb.offline=true"
    fi

    if [ -n "$BSUB_TEMPLATE" ]; then
        # When submitting via bsub, explicitly run through micromamba to activate the environment
        FULL_CMD="micromamba run -n glm $cmd"
        
        # Define a unique job name for this sweep point
        JOB_NAME="${ALGO}_T-${STEPS}_${PARAM_NAME}-${val}"
        
        # Replace occurrences of <job_name> with our unique job name
        JOB_SUBMIT_CMD="${BSUB_TEMPLATE//<job_name>/$JOB_NAME}"
        
        # If "SHELL COMMANDS" placeholder is in the template, replace it; otherwise append the command at the end
        if [[ "$JOB_SUBMIT_CMD" == *"SHELL COMMANDS"* ]]; then
            JOB_SUBMIT_CMD="${JOB_SUBMIT_CMD//SHELL COMMANDS/$FULL_CMD}"
        else
            JOB_SUBMIT_CMD="$JOB_SUBMIT_CMD \"$FULL_CMD\""
        fi
        
        echo "Submitting: $JOB_SUBMIT_CMD"
        eval "$JOB_SUBMIT_CMD"
    else
        echo "Running locally/sequentially: $cmd"
        eval "$cmd"
    fi
done

echo "Submission/Sweep script loop complete!"
