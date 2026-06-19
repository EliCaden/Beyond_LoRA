# Beyond LoRA: Is Sparsity-Induced Adaptation Better?

<p align="center">
  <a href="https://elicaden.github.io/Beyond_LoRA/">
    <img src="https://img.shields.io/badge/Project-Website-224b8d?style=for-the-badge" alt="Project Website">
  </a>
  <a href="https://arxiv.org/pdf/2606.13767">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge" alt="Paper PDF">
  </a>
  <a href="https://arxiv.org/abs/2606.13767">
    <img src="https://img.shields.io/badge/arXiv-2606.13767-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
</p>

<p align="center">
  <a href="https://github.com/EliCaden">Elijah Cadenhead</a><sup>1</sup> ·
  <a href="https://github.com/nizswan">Cristian McGee</a><sup>1</sup> ·
  <a href="https://scholar.google.com/citations?user=6j5YOXMAAAAJ&hl=en&oi=ao">Xin Li</a><sup>1</sup> ·
  <a href="https://scholar.google.com/citations?user=qzxprWoAAAAJ&hl=en&oi=ao">El Houcine Bergou</a><sup>2</sup> ·
  <a href="https://scholar.google.com/citations?user=vquoiHsAAAAJ&hl=en&oi=ao">Aritra Dutta</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup><a href="https://www.ucf.edu/">University of Central Florida</a> &nbsp;&nbsp;
  <sup>2</sup><a href="https://um6p.ma/en">Mohammed VI Polytechnic University</a>
</p>

<p align="center">
  <strong>Sparse, structured LoRA variants for cheaper and competitive parameter-efficient fine-tuning</strong>
</p>

<p align="center">
  <a href="https://github.com/EliCaden/Beyond_LoRA">
    <img src="https://img.shields.io/badge/If%20you%20like%20our%20project%2C%20please%20give%20us%20a%20star%20%E2%AD%90%20on%20GitHub%20for%20the%20latest%20update-red?style=for-the-badge" alt="If you like our project, please give us a star ⭐ on GitHub for the latest update">
  </a>
</p>

---

## Latest Updates

* **Jun 2026** — Paper, project page, and code released: [Project Page](https://elicaden.github.io/Beyond_LoRA/) · [Paper](https://arxiv.org/pdf/2606.13767) · [arXiv](https://arxiv.org/abs/2606.13767)

---

## Highlights

### Abstract

Low-rank adaptation (LoRA) and its variants provide a memory- and compute-efficient alternative to full fine-tuning of pre-trained models. However, questions remain about the comparative generalizability of these approaches and how the structural restrictions on low-rank updates preserve effective adaptation performance. We present a historical framing, covering the past (full fine-tuning and original LoRA), the present (different variants of LoRA), and propose simpler, cheaper, parameter-efficient extensions by inducing sparsity within existing LoRA variants: Cheap LoRA (cLA), training a single low-rank factor with the other fixed (deterministically or, in its randomized variant, stochastically), and the chained circulant variant, c<sup>3</sup>LA.

We frame cLA as a structured instance of asymmetric LoRA, serving as a controlled column-subspace restriction of full fine-tuning. We derive information-theoretic generalization error bounds for these variants, marking one of the first endeavors in this area. Empirically, we evaluate **11 fine-tuning methods** across **10 pre-trained models and 14 datasets**, analyzing the fine-tuned models' performance and generalization using tools such as loss landscapes and spectral analysis. Despite the sensitivity of fine-tuned models to the pre-trained model, datasets, and other factors, our study suggests that restricting LoRA-based PEFT methods' adaptation to a sparse, structured column space remains competitive across tasks with their parameter-matched baselines while reducing **up to 10% training time** and **peak GPU memory up to 15%**, even with a naïve, non-optimized, sparse implementation. Our theoretical and empirical generalization measures provide a more consistent and principled approach to their cost-effective adaptation than commonly used analytical tools.

### Key Contributions

| Contribution                    | Summary                                                                                                                                                                                                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sparse LoRA Variants**        | We introduce cLA, random-cLA, c<sup>3</sup>LA, and random-c<sup>3</sup>LA as simple, sparse extensions of state-of-the-art LoRA variants. These methods train restricted column-subspace updates by fixing part of the low-rank structure, thereby separating trainable parameter count from update geometry. |
| **Generalization Bounds**       | We derive information-theoretic generalization bounds for LoRA-family updates. The resulting framework connects rank, chain length, layer dimensions, bitwidth, dataset size, and update support to the generalization behavior of fine-tuned models.                                                         |
| **Benchmarking and Evaluation** | We benchmark 11 fine-tuning methods across 10 pretrained models and 14 datasets spanning NLP, vision, code generation, and logical reasoning, while measuring accuracy, empirical generalization, loss landscapes, spectral behavior, runtime, throughput, and memory.                                        |

---

## Overview of Proposed Sparsity-Induced LoRA Variants

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>High-level idea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>cLA</code></td>
      <td>Fix <code>A = [I_r | 0]</code> and train only <code>B</code>, restricting adaptation to a deterministic <code>r</code>-column subspace.</td>
    </tr>
    <tr>
      <td><code>random-cLA</code></td>
      <td>Randomize the fixed column selector while still training only <code>B</code>, spreading the sparse update over a randomized column restriction.</td>
    </tr>
    <tr>
      <td><code>c<sup>3</sup>LA</code></td>
      <td>Chain cLA modules and shift the identity block by <code>r</code> columns across chains, expanding the covered columns of the pretrained layer.</td>
    </tr>
    <tr>
      <td><code>random-c<sup>3</sup>LA</code></td>
      <td>Combine randomized selectors with the chained cLA construction, yielding a randomized sparse chained update.</td>
    </tr>
  </tbody>
</table>

### Algorithms

The pseudocode below sketches the sparse adaptation mechanisms used by the proposed variants.

<details>
<summary>cLA</summary>

```text
Given pretrained layer W0 ∈ R^{n×m}
Choose rank r and scale α

Set A = [ I_r | 0_{r×(m-r)} ]
Initialize B = 0_{n×r}

For each training step:
  Compute loss L using W = W0 + (α/r) · B · A
  Update B ← B − η · ∇_B L

Return adapted layer W
```

</details>

<details>
<summary>c³LA</summary>

```text
Given pretrained layer W0 ∈ R^{n×m}
Choose rank r, chain length k, scale α

For j = 1..k:
  Define A_j as a shifted selector
    A_1 = [ I_r | 0 | 0 | ... ]
    A_2 = [ 0 | I_r | 0 | ... ]
    ...
  Initialize B_j = 0_{n×r}

For j = 1..k:
  For each training step:
    Compute loss L using
      W = W0 + Σ_{t=1..j} (α/r) · B_t · A_t
    Update B_j ← B_j − η · ∇_{B_j} L

Return adapted layer W
```

</details>

<details>
<summary>random-cLA</summary>

```text
Given pretrained layer W0 ∈ R^{n×m}
Choose rank r and scale α

Sample a fixed randomized selector A
Initialize B = 0_{n×r}

For each training step:
  Compute loss L using W = W0 + (α/r) · B · A
  Update B ← B − η · ∇_B L

Return adapted layer W
```

</details>

<details>
<summary>random-c³LA</summary>

```text
Given pretrained layer W0 ∈ R^{n×m}
Choose rank r, chain length k, scale α

For j = 1..k:
  Sample a fixed randomized selector without replacement A_j from the shifted selectors of c^{3}LA
  Initialize B_j = 0_{n×r}

For j = 1..k:
  For each training step:
    Compute loss L using
      W = W0 + Σ_{t=1..j} (α/r) · B_t · A_t
    Update B_j ← B_j − η · ∇_{B_j} L

Return adapted layer W
```

</details>

### Naïve Sparse Implementation

The sparse construction can be implemented without multiplying by the full fixed selector. For cLA and random-cLA, the selector `A` simply chooses `r` coordinates of the input. Instead of computing `A(x)` as a dense matrix multiplication, the implementation stores the selected column indices and directly gathers `[x_c1, …, x_cr]`. This avoids unnecessary selector FLOPs and helps explain the observed runtime and memory reductions.

<p align="center">
  <img src="docs/Sparse_Figure_Final.png" width="850" alt="Naive sparse implementation diagram">
</p>

### Bridge to PaCA

Partial Connection Adaptation (PaCA) was motivated from a systems perspective: it reduces activation memory by only training a subset of the columns of each original layer's weights. Our sparse LoRA variants provide a theoretical bridge between PaCA and LoRA; when PaCA fine-tunes the first `r` columns of the pretrained layer, it updates the same parameters as cLA. Thus PaCA can be reframed theoretically as a LoRA-style update using cLA with a corresponding selector matrix `A`, allowing us to apply theoretical results from LoRA to PaCA.

<p align="center">
  <img src="docs/lora_paca_connect.png" width="850" alt="Connection between LoRA, PaCA, and sparsity-induced LoRA variants">
</p>

---

## Theoretical Contributions

Theorem 1 is a general bound for an arbitrary fully connected `L`-layer neural network. It upper bounds the generalization error of the fine-tuned model `W₀ + ΔW` using the generalization behavior of either the pretrained backbone or the update. This makes it a reusable template: once a PEFT method specifies the structure of `ΔW`, the theorem can comment on its generalizability.

The standalone correction terms `Φ_ΔW` and `Φ_W0` collect the Lipschitz constants of the loss and activations, layerwise spectral norms of the base and update weights, and zero-activation offset terms from recursively collapsing the difference between the fine-tuned and pretrained networks.

<p align="center">
  <img src="docs/Theorem_1_itself.png" width="850" alt="Generalization error upper bound theorem">
</p>

**Intuition.** The theorem converts the problem of comparing PEFT updates into a spectral and information-theoretic control problem. Each layer contributes either base-model spectral magnitude or update spectral magnitude, and the LoRA-family table is obtained by plugging in the number and structure of trainable update parameters.

**Extension to transformer architectures.** Theorem 1 applies to any architecture that can be written as a composition of linear maps and Lipschitz maps, under bounded input. We therefore view transformer blocks as fitting the theorem. For the specifics on adapting Theorem 1 to the attention mechanism, see Appendix D.1.5 of the paper.

With the additional assumption that the loss function `ℓ(·)` is `σ`-sub-Gaussian, we obtain upper bounds for the LoRA variants studied in this paper and for PaCA. The table below summarizes these bounds. For the derivation of each variant-specific bound, see Appendix D.1.6 of the paper.

<p align="center">
  <img src="docs/Gen_Bound_Theory_Table.png" width="850" alt="Generalization upper bounds for different PEFT methods">
</p>

Notation:

<table>
  <thead>
    <tr>
      <th>Symbol</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>m_i</code></td>
      <td>input dimension of layer <code>i</code></td>
    </tr>
    <tr>
      <td><code>n_i</code></td>
      <td>output dimension of layer <code>i</code></td>
    </tr>
    <tr>
      <td><code>r</code></td>
      <td>adapter rank</td>
    </tr>
    <tr>
      <td><code>k</code></td>
      <td>chain length</td>
    </tr>
    <tr>
      <td><code>q</code></td>
      <td>bitwidth of the stored weights</td>
    </tr>
    <tr>
      <td><code>σ</code></td>
      <td>sub-Gaussian parameter of the loss in the mutual-information bound</td>
    </tr>
    <tr>
      <td><code>|N|</code></td>
      <td>fine-tuning dataset size</td>
    </tr>
  </tbody>
</table>

---

## Empirical Benchmarks

The full empirical comparison reports performance and generalization over 11 fine-tuning methods. For CoLA we report the Matthews correlation coefficient (higher is better); for GPT2-small, perplexity (lower is better); and for the remaining datasets, accuracy. We use **green**, **red**, and **blue** to indicate the best, second best, and third best result. For the sparse variants, ↓ indicates the accuracy drop percentage compared to the best.

Here, FFT denotes full fine-tuning, LoRA is the standard low-rank adapter baseline, and CoLA is its chained counterpart. Asymmetric LoRA trains only one low-rank factor (`B`) and freezes `A`. RAC is its chained counterpart. cLA/c<sup>3</sup>LA and their random variants are ours.

<p align="center">
  <img src="docs/Table2_Performance.png" width="1000" alt="Table 2: performance of fine-tuned models">
</p>

**Key takeaways.** No single method substantially outperforms the others for adapting the model to their downstream tasks, including FFT. The sparsity-induced SOTA LoRA variants outperform FFT and LoRA in some tasks by a large margin and in many cases their performance drop is modest. This suggests that when fine-tuning a model for a downstream task, it may be optimal to select a fine-tuning method based on its other characteristics and user-specific needs, rather than just the generated accuracy. Although the sparse variants do not reduce the number of trainable parameters compared to their non-sparse LoRA counterparts, they reduce training time by 5–10% and peak GPU memory by 5–15%, with a naïve, non-optimized, sparse implementation.

<p align="center">
  <img src="docs/Table3_Generalization.png" width="1000" alt="Table 3: empirical generalization error">
</p>

Empirical generalization error, `𝒢(W)`, of the fine-tuning methods over various models and datasets. Lower values are better. These values are approximations for how far off the loss of the model obtained on the training set will match the loss of the model on its entire input space.

**Key takeaways.** Drawing a connection from our theoretical upper bounds in our LoRA table in the theoretical section above, we find PEFT methods with the same upper bounds perform similarly in practice. More precisely, cLA has a smaller upper bound on `𝒢(W)` than r-c<sup>3</sup>LA in practice matching the theory. This observation also holds for cLA and RAC, and c<sup>3</sup>LA and Asymmetric LoRA pairs. On the other hand, cLA and r-cLA have the same upper bound on `𝒢(W)`, and they also perform almost similarly in practice. Nevertheless, there are some discrepancies, and we attribute them to the fact that Table 1 gives us an upper bound on `𝒢(W)`.

---

## Loss Landscapes

To expand on why the theoretical bounds are valuable, we introduce two alternative methods of analyzing the generalizability of the fine-tuned models: loss landscapes and intruder dimensions. We show that, while there are valuable insights and consistencies among them, their ability to predict which of the fine-tuned methods generalize the best is less consistent than using our theoretical bounds.

3D-loss landscapes visualize how a model’s empirical loss differs under small parameter perturbations. A sharper loss landscape indicates worse generalization, smoother landscapes indicate the PEFT method is more robust to initialization. For more details, refer to Appendix E.4.1 of the paper.

<p align="center">
  <img src="docs/Updated_Teaser_Good.png" width="850" alt="Loss landscape comparison across fine-tuning methods">
</p>

Loss landscapes of ViT-Base fine-tuned on OfficeHome (top row) with PCA directions, and RoBERTa-Base fine-tuned on CoLA (bottom row) with random directions. For a comparison of the difference between the two methods, see Appendix E.4.1.

**Key takeaways.** The loss-landscape heuristic does not consistently align with empirical generalization in our experiments. Chain methods such as RAC-LoRA, CoLA, and c<sup>3</sup>LA often produce sharper landscapes than their non-chain counterparts, which would normally suggest worse generalization. However, this is not always what we observe empirically. This discrepancy between practice and theory is consistent across vision and text model modalities.

---

## Intruder Dimensions

Intruder dimensions compare the performance between the fine-tuned models of LoRA and FFT. Given the pretrained and fine-tuned models, `W₀` and `W₀ + ΔW`, the number of intruder dimensions correlates with their performance on the pretraining task, with more intruders indicating worse performance. For further details, see Appendix E.4.2 of the paper. We ask: will forgetting less of the more diverse dataset indicate better generalizability?

<p align="center">
  <img src="docs/Intruder_Dimensions_horizontal.png" width="1000" alt="Intruder dimension counts across fine-tuned models">
</p>

Average number of intruder dimensions present in different fine-tuned models at the end of training. The panels compare RoBERTa-Base fine-tuned on CoLA, ViT-Base fine-tuned on OfficeHome, and ViT-Base fine-tuned on CIFAR-10 over varying cosine similarity thresholds `ε ∈ (0, 1]`.

**Key takeaways.** The chain variant of any LoRA PEFT method produces more intruders than its non-chain counterpart; see LoRA compared to CoLA, Asymmetric LoRA to RAC, and cLA to c<sup>3</sup>LA in the above figure. This correlates with our loss landscapes, where chain variants produce sharper landscapes. However, the expected worse generalizability of these chain methods is not observed empirically as consistently as our theoretical bounds.

---

## Closing Takeaways

* PEFT performance is task-dependent: no single fine-tuning method dominates across all models and datasets.
* Our proposed sparse extensions of SOTA LoRA variants perform well across multiple modalities and models while substantially reducing training time and memory requirements.
* From a theoretical perspective, our sparsity-induced variants serve as a bridge between LoRA and PaCA, two different families of PEFT methods. While these sparse variants may require larger budgets to maintain robustness in certain settings, they remain overall effective, highlighting the importance of selecting fine-tuning methods based on task characteristics and user constraints.
* We show that, in theory, the sparse methods have the same generalization error upper bounds as their non-sparse counterparts, and closely track the empirical generalization trend across most models and modalities. This insight provides a more consistent and guided pathway for selecting PEFT methods, complementing existing diagnostic tools such as loss-landscape and intruder-dimension analyses.

---

# Repository and Implementation Guide

This repository contains the code accompanying the paper **"Beyond LoRA: Is Sparsity-Induced Adaptation Better?"** The paper studies full fine-tuning, LoRA, asymmetric LoRA, chain-style LoRA variants, cheaper/sparser LoRA variants, and related PEFT methods across language, vision, code, and reasoning tasks.

The repository is organized as two related codebases:

```text
Beyond_LoRA/
├── ProjectModelsOne/      # RoBERTa, GPT-2, ViT, LoRA/PaCA/sparse variants, and landscape tools
├── ProjectModelsTwo/      # Unified PEFT trainer suite for DeBERTa, TinyLlama, Llama-3, and DeepSeek-Coder
├── setup/                 # Environment setup helper
└── environment.template.yml
```

---

## Repository Contents

### `ProjectModelsOne/`

`ProjectModelsOne` contains the main training and analysis suite for RoBERTa, GPT-2, and ViT experiments. It includes standard full fine-tuning, LoRA, asymmetric LoRA, chain-style variants, sparse variants, PaCA variants, and loss-landscape / PCA analysis tooling.

```text
ProjectModelsOne/
├── analysis_tools/      # PCA directions, 2D/3D loss landscapes, plotting helpers
├── data/                # GLUE, E2E, CIFAR, OfficeHome, VTAB-style dataset loaders
├── methods/             # Full FT, LoRA q/v, head-only, LoRA-only, PaCA q/v methods
├── models/              # RoBERTa, GPT-2, ViT, LoRA layers/recipes, PaCA layers/recipes
├── scripts/             # Sweep entrypoints for text, vision, and E2E experiments
├── tasks/               # Task wrappers for causal LM, GLUE-style text, and image classification
├── trainers/            # Generic trainer and callbacks
├── utils/               # Adapter injection and target-module helpers
└── requirements.txt
```

Important entrypoints:

| Script                                | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `scripts/sweep_glue.py`               | RoBERTa-Base/Large experiments on GLUE-style tasks.    |
| `scripts/sweep_vision.py`             | ViT-Tiny/Base/Large/XXL experiments on image datasets. |
| `scripts/sweep_e2e.py`                | GPT-2 experiments on E2E natural language generation.  |
| `analysis_tools/weights_pca.py`       | PCA directions from checkpoint trajectories.           |
| `analysis_tools/plot_2d_with_path.py` | 2D loss landscapes with optimization paths.            |
| `analysis_tools/plot_3d_landscape.py` | 3D loss landscape visualizations.                      |

### `ProjectModelsTwo/`

`ProjectModelsTwo` contains a unified trainer suite for additional PEFT experiments on larger language/code/reasoning models.

```text
ProjectModelsTwo/
├── run.py                    # CLI dispatcher
├── deberta_trainer.py        # DeBERTa-v3-base and DeBERTa-v2-xxlarge experiments
├── deepseekcoder_trainer.py  # DeepSeek-Coder-1.3B experiments
├── tinyllama_trainer.py      # TinyLlama-1.1B experiments
├── llama3_trainer.py         # Llama-3-8B experiments
└── requirements.txt
```

Supported model families and datasets include:

| Model family   | Datasets/tasks                            |
| -------------- | ----------------------------------------- |
| DeBERTa        | `mrpc`, `rte`, `sts-b`, `trec50`, `paws`  |
| TinyLlama      | `openbookqa`, `folio`, `logiqa`, `clutrr` |
| Llama-3        | `openbookqa`, `clutrr`                    |
| DeepSeek-Coder | `django`                                  |

---

## Methods

The code covers the PEFT method families studied in the paper. Naming differs slightly between the two project folders, but the major method families are:

| Family             | Common CLI names                                          | Description                                                                       |
| ------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Full fine-tuning   | `full`, `fft`, `ft`                                       | Train all model parameters.                                                       |
| Head-only          | `head`, `heads`, `head_only`                              | Train only the task-specific head.                                                |
| LoRA               | `lora`, `base`, `vanilla`                                 | Standard low-rank adaptation.                                                     |
| Asymmetric LoRA    | `asym_a`, `asym_b`, `fixa`                                | Freeze one low-rank factor and train the other.                                   |
| Cheap LoRA / cLA   | `cheap`, `cla`                                            | Fixed structured low-rank factor; train the other factor.                         |
| Random cLA         | `random_cheap`, `rcla`, `random`                          | Randomized fixed-factor variant.                                                  |
| Chain LoRA / CoLA  | `cola`, `chain`                                           | Merge the current low-rank update into the base model and reinitialize adapters.  |
| RAC                | `rac`, `rac_a`, `rac_b`                                   | Chain-style merge/reinitialize variant with randomized resets.                    |
| c3LA               | `c3la`, `modest`                                          | Chained/sliding structured factor variant.                                        |
| Shuffled c3LA      | `shuffle`, `rc3la`                                        | Sliding-window chain variant with random shuffling.                               |
| LoRA+              | `lora_plus`, `lora+`, `plus`                              | LoRA with different learning rates for the two factors.                           |
| Sparse variants    | `sparse_cheap`, `sparse_shuffle`, `sparse_c3la`           | Sparse versions of structured LoRA variants.                                      |
| BA-sparse variants | `ba_sparse_lora`, `ba_sparse_final`, `ba_sparse_fix_mask` | Mask the effective `BA` update during training, at the end, or with a fixed mask. |
| PaCA variants      | `paca`, `dpaca`, `cpaca`, `dcpaca`                        | Partial connection adaptation variants.                                           |

---

## Environment Setup

A lightweight conda environment template is provided at the repository root.

```bash
conda env create -f environment.template.yml
conda activate beyond_lora
```

Then install the project-specific requirements for the codebase you plan to run:

```bash
pip install -r ProjectModelsOne/requirements.txt
pip install -r ProjectModelsTwo/requirements.txt
```

Alternatively, use the helper script:

```bash
bash setup/create_beyond_lora_env.sh environment.template.yml beyond_lora
conda activate beyond_lora
```

For GPU experiments, make sure the installed PyTorch/CUDA versions match the target machine or cluster. The template is intended as a reproducible starting point rather than a machine-specific lockfile.

---

## Reproducing Representative Experiments from `ProjectModelsOne`

Run these commands from inside `ProjectModelsOne`:

```bash
cd ProjectModelsOne
```

The paper uses rank `r=16` and scaling factor `alpha=32` for LoRA-style PEFT methods in the representative settings. Full fine-tuning uses a separate learning rate and weight decay.

### RoBERTa-Base / RoBERTa-Large on MRPC and CoLA

```bash
# RoBERTa-Base full fine-tuning
python -m scripts.sweep_glue \
  --tasks mrpc cola \
  --variant base \
  --methods full \
  --seeds 12 22 32 \
  --lrs 1e-5 \
  --batch-size 32 \
  --epochs 20 \
  --max-length 128 \
  --scheduler linear \
  --min-lr 1e-6 \
  --warmup-ratio 0.1 \
  --weight-decay-full 0.01 \
  --out-dir runs/roberta_base_table7_full

# RoBERTa-Base PEFT methods
python -m scripts.sweep_glue \
  --tasks mrpc cola \
  --variant base \
  --methods lora asym_a asym_b cheap random_cheap cola rac_a rac_b c3la shuffle lora_plus sparse_cheap sparse_shuffle sparse_c3la ba_sparse_lora ba_sparse_final ba_sparse_fix_mask paca dpaca cpaca dcpaca \
  --seeds 12 22 32 \
  --lrs 3e-4 \
  --ranks 16 \
  --alphas 32 \
  --chain-every-epochs 3 \
  --batch-size 32 \
  --epochs 20 \
  --max-length 128 \
  --scheduler linear \
  --min-lr 1e-6 \
  --warmup-ratio 0.1 \
  --weight-decay-lora 0 \
  --out-dir runs/roberta_base_table7_peft
```

For RoBERTa-Large, change `--variant base` to `--variant large` and update the output directory.

### GPT-2 on E2E

```bash
# GPT-2 full fine-tuning
python -m scripts.sweep_e2e \
  --dataset e2e \
  --backbone gpt2 \
  --methods full \
  --seeds 12 22 32 \
  --lr-full 5e-5 \
  --epochs 30 \
  --batch-size 16 \
  --max-length 64 \
  --chain-every-epochs 1 \
  --out-dir runs/gpt2_e2e_table7_full

# GPT-2 PEFT methods
python -m scripts.sweep_e2e \
  --dataset e2e \
  --backbone gpt2 \
  --methods lora asym_a asym_b cheap random_cheap cola rac_a rac_b c3la shuffle lora_plus sparse_cheap sparse_shuffle sparse_c3la ba_sparse_lora ba_sparse_final \
  --seeds 12 22 32 \
  --lr-others 3e-4 \
  --ranks 16 \
  --alphas 32 \
  --epochs 30 \
  --batch-size 16 \
  --max-length 64 \
  --chain-every-epochs 1 \
  --out-dir runs/gpt2_e2e_table7_peft
```

### ViT-Tiny / ViT-Base on OfficeHome and CIFAR-10

```bash
# ViT-Tiny full fine-tuning
python -m scripts.sweep_vision \
  --datasets officehome cifar10 \
  --vit-variant tiny \
  --methods full \
  --seeds 12 22 32 \
  --lr-full 3e-4 \
  --batch-size 64 \
  --epochs 30 \
  --img-size 224 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --warmup-ratio 0.05 \
  --weight-decay-full 0.05 \
  --out-dir runs/vit_tiny_table7_full

# ViT-Tiny PEFT methods
python -m scripts.sweep_vision \
  --datasets officehome cifar10 \
  --vit-variant tiny \
  --methods lora asym_a asym_b cheap random_cheap cola rac_a rac_b c3la shuffle lora_plus sparse_cheap sparse_shuffle sparse_c3la ba_sparse_lora ba_sparse_final ba_sparse_fix_mask paca dpaca cpaca dcpaca \
  --seeds 12 22 32 \
  --lr-others 1e-3 \
  --ranks 16 \
  --alphas 32 \
  --chain-every-epochs 5 \
  --batch-size 64 \
  --epochs 30 \
  --img-size 224 \
  --scheduler cosine \
  --min-lr 1e-6 \
  --warmup-ratio 0.05 \
  --weight-decay-lora 0 \
  --out-dir runs/vit_tiny_table7_peft
```

For ViT-Base, change `--vit-variant tiny` to `--vit-variant base` and update the output directory.

---

## Running `ProjectModelsTwo`

Run these commands from inside `ProjectModelsTwo`:

```bash
cd ProjectModelsTwo
```

Basic CLI examples:

```bash
python run.py
python run.py deberta lora mrpc
python run.py tinyllama chain clutrr epochs=10 chainReset=2 rank=16
python run.py llama3 plus openbookqa lr_ratio=32 learningRate=2e-5
python run.py deepseek fft django maxLength=2048 batchSize=2
```

Programmatic sweep example:

```python
#!/usr/bin/env python3
import tinyllama_trainer as TinyLlama

for method in ("rac", "chain", "c3la", "rc3la"):
    for seed in (100, 101, 102):
        TinyLlama.Run(
            model="tinyllama",
            dataset="logiqa",
            method=method,
            maxLength=512,
            batchSize=8,
            learningRate=1e-3,
            epochs=10,
            chainReset=2,
            rank=16,
            alpha=32,
            seed=seed,
        )
```

The `Run(...)` API accepts:

```python
Run(
    model, dataset, method,
    maxLength, batchSize, learningRate, epochs,
    chainReset,   # chain, rac, c3la, rc3la
    rank, alpha,  # alpha=0 sets alpha to 2 * rank
    lr_ratio,     # LoRA+ only
    seed,
)
```

Notes:

* `chainReset` is count-based: `chainReset=2` means two resets total, spread across training.
* `alpha=0` automatically sets `alpha = 2 * rank`.
* Unused hyperparameters for a method are accepted for sweep parity and ignored.
* `rte`, `sst2`, and `stsb` print train/validation metrics only when public test labels are unavailable. STS-B reports Pearson correlation in the accuracy slot.

---

## Weights & Biases Logging

For scripts that support WandB logging, add:

```bash
--wandb --wandb-project beyond-lora --wandb-group table7 --wandb-mode online
```

For offline or restricted cluster runs, use:

```bash
--wandb --wandb-project beyond-lora --wandb-group table7 --wandb-mode offline
```

---

## Outputs

`ProjectModelsOne` sweep scripts write JSONL summaries under the selected `--out-dir`, together with checkpoints unless checkpoint-saving options are disabled. Useful flags for large sweeps include:

```bash
--dry-run
--resume
--overwrite
--save-best-only
--save-json-only
```

`ProjectModelsTwo` trainers print epoch-level train/validation/test metrics in the terminal.

---

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the material for any purpose, provided that appropriate credit is given. Please cite the paper when using this repository, figures, or results in academic work.

---

## Citation

If this repository is useful for your work, please cite the accompanying paper:

```bibtex
@article{beyondlora,
  title={Beyond LoRA: Is Sparsity-Induced Adaptation Better?},
  author={Cadenhead, Elijah and McGee, Cristian and Li, Xin and Bergou, El Houcine and Dutta, Aritra},
  journal={arXiv preprint arXiv:2606.13767},
  year={2026}
}
```
