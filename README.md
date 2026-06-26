<h1 align="center">Fast and Slow Thinking with Generator Language Models</h1>

<div align="center">
  
**[Author 1]()**, &nbsp; **[Author 2]()**, &nbsp; **[Author 3]()**

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-grey?style=flat&logo=arxiv)]()
[![Project Page](https://img.shields.io/badge/Project_Page-grey?style=flat&logo=github)]()
[![Blog](https://img.shields.io/badge/Blog-grey?style=flat&logo=rss)]()
[![Google Drive](https://img.shields.io/badge/Google_Drive-grey?style=flat&logo=googledrive&logoColor=white)]()
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-grey?style=flat&logoColor=white)]()

</div>

## News

## TL;DR

<p align="center">
  <img src="figures/overview.gif" width="100%">
</p>

<p align="center">
  <img src="figures/overview.png" width="100%">
</p>

We introduce the **Generator Language Model (GLM)** framework, unifying flow matching and discrete jumps into a single simplex-aware generative process, and its distilled flow-map variant enabling **one-step parallel text generation** ("fast thinking") alongside multi-step sampling ("slow thinking").

## Overview

**Generator Language Models (GLMs)** unify continuous flows and discrete jumps over sequence manifolds into a single generative framework. By encoding text as one-hot vectors and using Generator Matching, GLMs can interpolate between deterministic continuous ODE flows ("slow thinking" with many steps) and stochastic discrete updates. Unlike standard discrete diffusion models that operate purely on discrete state transitions, or continuous flow models that discard masking entirely, GLMs bridge these paradigms (e.g., via Soft-Masked Flow Language Models) to perform hybrid continuous-discrete denoising. Furthermore, we support flow-map distillation to learn the direct solution operator of the generative path, enabling **one-step parallel language generation** ("fast thinking").

## How to Run

### Install Dependencies

```bash
pip install torch>=2.3.0
pip install -r requirements.txt
# Install flash-attn separately matching your python / torch version (see https://github.com/Dao-AILab/flash-attention/releases)
pip install flash-attn==2.8.3 --no-build-isolation
```

Our DiT backbone supports `torch.compile` with `max-autotune` for faster training. Enable it by setting the environment variable before running any script:

```bash
export DIT_USE_COMPILE=TRUE
```

With the option, we are able to train OpenWebText experiments with 512 batch size on 8 H100 (80GB VRAM), with local batch size of 32.

### Training

Before running, update the script's configuration placeholders:
* `DATA_DIR="YOUR_DATA_DIR"`: directory where OpenWebText datasets will be cached.
* `CHECKPOINT_DIR="YOUR_CHECKPOINT_DIR"`: directory where checkpoints will be saved.

All scripts enable `checkpointing.resume_from_ckpt=True` by default, allowing training to automatically resume from the latest saved checkpoint on cluster preemption.

Set `algo.teacher_path` to your pre-trained FLM checkpoint before running FMLM distillation.


| Model | Dataset     | Script                                                                     |
| ----- | ----------- | -------------------------------------------------------------------------- |
| FLM   | LM1B        | [scripts/train_lm1b_flm.sh](scripts/train_lm1b_flm.sh)                     |
| FMLM  | LM1B        | [scripts/train_lm1b_fmlm_denoiser.sh](scripts/train_lm1b_fmlm_denoiser.sh) |
| FLM   | OpenWebText | [scripts/train_owt_flm.sh](scripts/train_owt_flm.sh)                       |
| SM-FLM| OpenWebText | [scripts/train_owt_smflm.sh](scripts/train_owt_smflm.sh)                   |
| FMLM  | OpenWebText | [scripts/train_owt_fmlm_denoiser.sh](scripts/train_owt_fmlm_denoiser.sh)   |
| MDLM (Baseline) | OpenWebText | [scripts/train_owt_mdlm.sh](scripts/train_owt_mdlm.sh)                   |
| SEDD (Baseline) | OpenWebText | [scripts/train_owt_sedd.sh](scripts/train_owt_sedd.sh)                   |
| DUO (Baseline)  | OpenWebText | [scripts/train_owt_duo.sh](scripts/train_owt_duo.sh)                     |

### Evaluation

Set `CKPT_PATH` in the script to your trained checkpoint before running.


| Model | Dataset     | Script                                                       |
| ----- | ----------- | ------------------------------------------------------------ |
| FLM   | LM1B        | [scripts/gen_ppl_lm1b_flm.sh](scripts/gen_ppl_lm1b_flm.sh)   |
| FMLM  | LM1B        | [scripts/gen_ppl_lm1b_fmlm.sh](scripts/gen_ppl_lm1b_fmlm.sh) |
| FLM   | OpenWebText | [scripts/gen_ppl_owt_flm.sh](scripts/gen_ppl_owt_flm.sh)     |
| FMLM  | OpenWebText | [scripts/gen_ppl_owt_fmlm.sh](scripts/gen_ppl_owt_fmlm.sh)   |
| MDLM (Baseline) | OpenWebText | [scripts/gen_ppl_owt_mdlm.sh](scripts/gen_ppl_owt_mdlm.sh) |
| SEDD (Baseline) | OpenWebText | [scripts/gen_ppl_owt_sedd.sh](scripts/gen_ppl_owt_sedd.sh) |
| DUO (Baseline)  | OpenWebText | [scripts/gen_ppl_owt_duo.sh](scripts/gen_ppl_owt_duo.sh)   |

## Soft-Masked Flow Language Model (SM-FLM)

We support the **Soft-Masked Flow Language Model (SM-FLM)** algorithm, a hybrid continuous-discrete generative process that blends continuous Euclidean flow matching and discrete state transitions. 

At any timestep $t$:
- Tokens are stochastically "committed" (permanently predicted as discrete states) based on a time-dependent Bernoulli schedule $\gamma_t$ (linear or cosine schedule).
- The remaining uncommitted tokens are updated via continuous flow matching.
- Setting `gamma_scale=0` recovers standard pure FLM, while setting `beta_t=0` recovers a pure masked diffusion model.

Hyperparameters (such as `gamma_min`, `gamma_max`, `curriculum_start`, and `curriculum_end`) can be configured in [configs/algo/smflm.yaml](configs/algo/smflm.yaml).

## Checkpoints
### Pretrained Checkpoints

Pretrained FLM and FMLM checkpoints are available at [Google Drive]() or [Huggingface]().


| Model | Dataset     | Checkpoint       |
| ----- | ----------- | ---------------- |
| FLM   | LM1B        | `lm1b_flm.ckpt`  |
| FMLM  | LM1B        | `lm1b_fmlm.ckpt` |
| FLM   | OpenWebText | `owt_flm.ckpt`   |
| FMLM  | OpenWebText | `owt_fmlm.ckpt`  |


Set `eval.checkpoint_path` (or `algo.teacher_path` for distillation) to the downloaded checkpoint path when running evaluation or distillation scripts.

### Baseline Checkpoints

Reproduced baseline checkpoints for LM1B are available at [here]().

For other checkpoints, mostly for OpenWebText, refer to [Duo](https://github.com/s-sahoo/duo), [SDTT](https://github.com/jdeschena/sdtt), [RDLM](https://github.com/harryjo97/RDLM), [di4c](https://github.com/sony/di4c) repositories.


### Full results 

#### FLM (Undistilled)
<p align="center">
  <img src="figures/flm_figure.png" width="100%">
</p>

<table border="0" style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="width: 50%; vertical-align: top; border: none; padding-right: 10px;">
      <h4 align="center">LM1B</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">243.36</td><td style="text-align: center;">2.41</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">198.53</td><td style="text-align: center;">4.22</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">152.01</td><td style="text-align: center;">4.40</td></tr>
          <tr><td style="text-align: center;"><b>64</b></td><td style="text-align: center;">126.51</td><td style="text-align: center;">4.36</td></tr>
          <tr><td style="text-align: center;"><b>128</b></td><td style="text-align: center;">112.54</td><td style="text-align: center;">4.34</td></tr>
          <tr><td style="text-align: center;"><b>256</b></td><td style="text-align: center;">104.59</td><td style="text-align: center;">4.32</td></tr>
          <tr><td style="text-align: center;"><b>512</b></td><td style="text-align: center;">99.75</td><td style="text-align: center;">4.30</td></tr>
          <tr><td style="text-align: center;"><b>1024</b></td><td style="text-align: center;">96.91</td><td style="text-align: center;">4.29</td></tr>
        </tbody>
      </table>
    </td>
    <td style="width: 50%; vertical-align: top; border: none; padding-left: 10px;">
      <h4 align="center">OpenWebText</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">449.15</td><td style="text-align: center;">5.21</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">380.99</td><td style="text-align: center;">5.66</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">240.11</td><td style="text-align: center;">5.72</td></tr>
          <tr><td style="text-align: center;"><b>64</b></td><td style="text-align: center;">147.28</td><td style="text-align: center;">5.68</td></tr>
          <tr><td style="text-align: center;"><b>128</b></td><td style="text-align: center;">103.30</td><td style="text-align: center;">5.58</td></tr>
          <tr><td style="text-align: center;"><b>256</b></td><td style="text-align: center;">82.05</td><td style="text-align: center;">5.48</td></tr>
          <tr><td style="text-align: center;"><b>512</b></td><td style="text-align: center;">70.22</td><td style="text-align: center;">5.40</td></tr>
          <tr><td style="text-align: center;"><b>1024</b></td><td style="text-align: center;">62.23</td><td style="text-align: center;">5.33</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

#### FMLM (Distilled)

<p align="center">
  <img src="figures/fmlm_figure.png" width="100%">
</p>

<table border="0" style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="width: 50%; vertical-align: top; border: none; padding-right: 10px;">
      <h4 align="center">LM1B</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>1</b></td><td style="text-align: center;">119.34</td><td style="text-align: center;">4.16</td></tr>
          <tr><td style="text-align: center;"><b>2</b></td><td style="text-align: center;">110.19</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>4</b></td><td style="text-align: center;">98.76</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">86.32</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">78.35</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">69.21</td><td style="text-align: center;">4.21</td></tr>
        </tbody>
      </table>
    </td>
    <td style="width: 50%; vertical-align: top; border: none; padding-left: 10px;">
      <h4 align="center">OpenWebText</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>1</b></td><td style="text-align: center;">168.30</td><td style="text-align: center;">5.17</td></tr>
          <tr><td style="text-align: center;"><b>2</b></td><td style="text-align: center;">133.29</td><td style="text-align: center;">5.25</td></tr>
          <tr><td style="text-align: center;"><b>4</b></td><td style="text-align: center;">111.31</td><td style="text-align: center;">5.26</td></tr>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">86.50</td><td style="text-align: center;">5.36</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">63.63</td><td style="text-align: center;">5.29</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">45.09</td><td style="text-align: center;">5.25</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

## BibTeX

```bibtex

```

---

## Acknowledgements

This repository is built upon the codebases of **[FLM](https://github.com/david3684/flm)**, **[Duo](https://github.com/s-sahoo/duo)**, and **[ReDi](https://github.com/Ugness/ReDi)**.