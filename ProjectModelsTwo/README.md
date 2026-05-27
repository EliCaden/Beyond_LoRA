# Unified PEFT Trainer Suite

## Layout

```
run.py                      # CLI dispatcher
deberta_trainer.py          # DeBERTa-v3-base, DeBERTa-v2-xxlarge
deepseekcoder_trainer.py    # DeepSeek-Coder-1.3B
tinyllama_trainer.py        # TinyLlama-1.1B
llama3_trainer.py           # Llama-3-8B
requirements.txt            # Requirements
```

## Methods

| Canonical | Aliases | Description |
|-----------|---------|-------------|
| `fft` | `ft` | Full fine-tuning |
| `lora` | `vanilla` | Standard LoRA |
| `cla` | `cheap` | `A = [I_r \| 0]` frozen, only `B` trains |
| `fixa` | — | `A` frozen at Kaiming init, only `B` trains |
| `rcla` | `random` | Random frozen `A`, only `B` trains |
| `chain` | `cola` | Merge `BA` into base + reinject fresh `A,B` |
| `rac` | — | Merge+reinject with re-seeded `A` per reset |
| `c3la` | `modest` | Sliding identity-window `A`, advances on reset |
| `rc3la` | `shuffle` | Sliding window, fuses delta + random shuffle on reset |
| `plus` | — | LoRA+: `A` at `lr/lr_ratio`, `B` at `lr` |

## Models and datasets

| Model | Datasets |
|-------|----------|
| `deberta_v3_base` | `mrpc`, `rte`, `sts-b`, `trec50`, `paws` |
| `deberta_v2_xxl` | `mrpcs`, `paws`, `trec50` |
| `tinyllama` | `openbookqa`, `folio`, `logiqa`, `clutrr` |
| `llama3` | `openbookqa`, `clutrr` |
| `deepseekcoder` | `django` |

## `Run(...)` API

```python
Run(
    model, dataset, method,
    maxLength, batchSize, learningRate, epochs,
    chainReset,   # chain, rac, c3la, rc3la
    rank, alpha,  # alpha=0 -> 2*rank
    lr_ratio,     # plus only
    seed,
)
```

- **`chainReset` is count-based**: `chainReset=2` means *two resets total*, spread across `[2, epochs]` (linspace-style).
- **`alpha=0`** auto-sets to `2 * rank`.
- Unused hyperparameters for a method are accepted but ignored (parity).

## CLI

```bash
python run.py                                          # defaults
python run.py deberta lora mrpc
python run.py tinyllama chain clutrr epochs=10 chainReset=2 rank=16
python run.py llama3 plus openbookqa lr_ratio=32 learningRate=2e-5
python run.py deepseek fft django maxLength=2048 batchSize=2
```

## Programmatic usage (SLURM sweeps)

```python
#!/usr/bin/env python3
import tinyllama_trainer as TinyLlama

# LR sweep × 3 seeds
for lr_exp in (-3, -3.5):
    for seed in (100, 101, 102):
        TinyLlama.Run(
            model="tinyllama", dataset="logiqa", method="cla",
            maxLength=512, batchSize=8, learningRate=10**lr_exp, epochs=10,
            chainReset=2, rank=16, alpha=32, seed=seed,
        )

# Reset-based methods × 3 seeds
for method in ("rac", "chain", "c3la", "rc3la"):
    for seed in (100, 101, 102):
        TinyLlama.Run(
            model="tinyllama", dataset="logiqa", method=method,
            maxLength=512, batchSize=8, learningRate=10**-3, epochs=10,
            chainReset=2, rank=16, alpha=32, seed=seed,
        )

# LoRA+ uses lr_ratio instead of chainReset
for seed in (100, 101, 102):
    TinyLlama.Run(
        model="tinyllama", dataset="logiqa", method="plus",
        maxLength=512, batchSize=8, learningRate=10**-3.25, epochs=10,
        rank=16, alpha=32, lr_ratio=16, seed=seed,
    )
```

Other trainers work identically — `import llama3_trainer as Llama3`, `import deepseekcoder_trainer as DeepSeek`, `import deberta_trainer as DeBERTa`.

## Output

```
epoch {ep}/{epochs}- train_loss:... train_acc:... val_loss:... val_acc:... test_loss:... test_acc:...
```

`rte`/`sst2`/`stsb` print train/val only (no public test labels). STS-B reports Pearson correlation in the accuracy slot.