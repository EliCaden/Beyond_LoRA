# Beyond LoRA: Is Sparsity-Induced Adaptation Better?

This repository contains the code accompanying the paper **"Beyond LoRA: Is Sparsity-Induced Adaptation Better?"** The paper studies full fine-tuning, LoRA, asymmetric LoRA, chain-style LoRA variants, cheaper/sparser LoRA variants, and related PEFT methods across language, vision, code, and reasoning tasks.

The repository is organized as two related codebases:

```text
Beyond_LoRA/
├── ProjectModelsOne/      # RoBERTa, GPT-2, ViT, LoRA/PaCA/sparse variants, and landscape tools
├── ProjectModelsTwo/      # Unified PEFT trainer suite for DeBERTa, TinyLlama, Llama-3, and DeepSeek-Coder
├── setup/                 # Environment setup helper
└── environment.template.yml
```

## Repository contents

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

| Script | Purpose |
|---|---|
| `scripts/sweep_glue.py` | RoBERTa-Base/Large experiments on GLUE-style tasks. |
| `scripts/sweep_vision.py` | ViT-Tiny/Base/Large/XXL experiments on image datasets. |
| `scripts/sweep_e2e.py` | GPT-2 experiments on E2E natural language generation. |
| `analysis_tools/weights_pca.py` | PCA directions from checkpoint trajectories. |
| `analysis_tools/plot_2d_with_path.py` | 2D loss landscapes with optimization paths. |
| `analysis_tools/plot_3d_landscape.py` | 3D loss landscape visualizations. |

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

| Model family | Datasets/tasks |
|---|---|
| DeBERTa | `mrpc`, `rte`, `sts-b`, `trec50`, `paws` |
| TinyLlama | `openbookqa`, `folio`, `logiqa`, `clutrr` |
| Llama-3 | `openbookqa`, `clutrr` |
| DeepSeek-Coder | `django` |

## Methods

The code covers the PEFT method families studied in the paper. Naming differs slightly between the two project folders, but the major method families are:

| Family | Common CLI names | Description |
|---|---|---|
| Full fine-tuning | `full`, `fft`, `ft` | Train all model parameters. |
| Head-only | `head`, `heads`, `head_only` | Train only the task-specific head. |
| LoRA | `lora`, `base`, `vanilla` | Standard low-rank adaptation. |
| Asymmetric LoRA | `asym_a`, `asym_b`, `fixa` | Freeze one low-rank factor and train the other. |
| Cheap LoRA / cLA | `cheap`, `cla` | Fixed structured low-rank factor; train the other factor. |
| Random cLA | `random_cheap`, `rcla`, `random` | Randomized fixed-factor variant. |
| Chain LoRA / CoLA | `cola`, `chain` | Merge the current low-rank update into the base model and reinitialize adapters. |
| RAC | `rac`, `rac_a`, `rac_b` | Chain-style merge/reinitialize variant with randomized resets. |
| c3LA | `c3la`, `modest` | Chained/sliding structured factor variant. |
| Shuffled c3LA | `shuffle`, `rc3la` | Sliding-window chain variant with random shuffling. |
| LoRA+ | `lora_plus`, `lora+`, `plus` | LoRA with different learning rates for the two factors. |
| Sparse variants | `sparse_cheap`, `sparse_shuffle`, `sparse_c3la` | Sparse versions of structured LoRA variants. |
| BA-sparse variants | `ba_sparse_lora`, `ba_sparse_final`, `ba_sparse_fix_mask` | Mask the effective `BA` update during training, at the end, or with a fixed mask. |
| PaCA variants | `paca`, `dpaca`, `cpaca`, `dcpaca` | Partial connection adaptation variants. |

## Environment setup

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

## Reproducing representative experiments from `ProjectModelsOne`

Run these commands from inside `ProjectModelsOne`:

```bash
cd ProjectModelsOne
```

The paper uses rank `r=16` and scaling factor `alpha=32` for LoRA-style PEFT methods in the representative Table 7 settings. Full fine-tuning uses a separate learning rate and weight decay.

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

- `chainReset` is count-based: `chainReset=2` means two resets total, spread across training.
- `alpha=0` automatically sets `alpha = 2 * rank`.
- Unused hyperparameters for a method are accepted for sweep parity and ignored.
- `rte`, `sst2`, and `stsb` print train/validation metrics only when public test labels are unavailable. STS-B reports Pearson correlation in the accuracy slot.

## Weights & Biases logging

For scripts that support WandB logging, add:

```bash
--wandb --wandb-project beyond-lora --wandb-group table7 --wandb-mode online
```

For offline or restricted cluster runs, use:

```bash
--wandb --wandb-project beyond-lora --wandb-group table7 --wandb-mode offline
```

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

## Citation

If this repository is useful for your work, please cite the accompanying paper:

```bibtex
@misc{cadenhead2026beyondlora,
  title  = {Beyond LoRA: Is Sparsity-Induced Adaptation Better?},
  author = {Cadenhead, Elijah and McGee, Cristian and Li, Xin and Bergou, El Houcine and Dutta, Aritra},
  year   = {2026},
  note   = {Preprint}
}
```

## License

No license file is currently included in this repository. Please contact the authors before redistributing or reusing the code outside the scope allowed by the repository host and accompanying paper.
