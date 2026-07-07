CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"

if [ "$CHECKPOINT_DIR" = "YOUR_CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR must be set"
    exit 1
fi

datasets=("ag_news"
          "scientific_papers_pubmed"
          "scientific_papers_arxiv"
          "lambada"
          "wikitext2"
          "wikitext103"
          "ptb"
          "lm1b-gpt2")
for data in "${datasets[@]}"; do
  echo "$data"
  python -u -m main \
    mode=ppl_eval \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.eval_global_batch_size=128 \
    data="$data" \
    data.insert_valid_eos=False \
    model=small \
    algo=duo_base \
    model.length=1024 \
    eval.checkpoint_path=$CHECKPOINT_DIR \
    sampling.num_sample_batches=0 \
    +wandb.offline=true
done
