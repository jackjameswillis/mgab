# AGENTS.md - Guidelines for Agentic Coding

## Environment

- **Python 3.10** (`.python-version`), use the included `venv/` — do not install into system Python.
- **GPU optional.** All scripts auto-detect CUDA; fall back to CPU. Use `.to(device)` on every tensor.
- **No formal test framework.** This is a research codebase with script-based execution and wandb runs.

## Key Scripts

| File | Command | Purpose |
|------|---------|---------|
| `para.py` | `python para.py [--shapes "784,64,10"] [--population_size 32] ...` | Train microbial GA population (all key params are CLI args now) |
| `sweep.py` | `python sweep.py` | Sweep pop sizes 100–1000; saves local JSON + per-pop `.npy` checkpoints |
| `analysis.py` | `python analysis.py` | Load a saved checkpoint (`longpop.npy`) and plot accuracy histogram |
| `sgdtest.py` | `python sgdtest.py` | SGD baseline on MNIST (uses `torchvision`, not sklearn/OMNIBL) |

## Module Boundaries

- **`PopMLP.py`** — Core: stacked-tensor MLP for population of individuals. Handles forward pass, tournament selection, crossover (`uni`/`asexual`), mutation. Creates `.to(device)` internally from `torch.cuda.is_available()`.
- **`precisions.py`** — Quantization classes `Q(bits)` and `f32()`. `w_bits=32` in constructor means no quantization (float rand init); any other value uses int8 with that many bits. Bias is always f32.
- **`geography.py`** — Tournament neighbor selection topologies: `Ring` and `SmallWorld`. `PopMLP.tournaments()` defaults to `G.Ring`. `geo.py` must define a class with `.tournament(deme_size, ...)` signature for custom topologies.
- **`wandb/`** — Logged output directory (gitignored). Set `WANDB_API_KEY` to use wandb tracking; `sweep.py` stores metrics locally instead.

## Data & Conventions

1. Load MNIST via `sklearn.datasets.fetch_openml('mnist_784', version=1, as_frame=False)`.
2. One-hot encode labels (`np.zeros((n, 10))`).
3. Split: `train_test_split(test_size=1/7, random_state=42)` → 6 train : 1 test.
4. Normalize to `(x - mean) / std` using **training set** statistics (applied to both train and test).

## Checkpoints & State

- Saved via `pop_mlp.state_dict()` → torch file (`.npy`, `.pth`). Contains only `weights.*` and `biases.*` tensors.
- Reconstruct from scratch with: create a new PopMLP, then `load_state_dict(torch.load(path))`. Do not pass `map_location=device` to `torch.load` — the existing loaders call `.to(device)` after loading.
- Default training checkpoint filename in para.py: `'longpop.npy'`. sweep.py uses `'pop_size_{N}.npy'`.

## para.py CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--shapes` | `784,64,10` | Layer sizes as comma-separated string |
| `--act` | `relu` | Activation: relu, tanh, sigmoid, elu, silu, linear |
| `--population_size` | 32 | Number of individuals in the population |
| `--num_generations` | 1000 | Training epochs (generations) |
| `--BATCH_SIZE` | 64 | Mini-batch size for tournament steps |
| `--pop_batch` | population_size | Batch size for test/training evaluation ops |
| `--demesize` | 2 | Tournament deme size (neighborhood radius) |
| `--mutation_rate`, `-mr` | 0.001 | Probability of weight mutation per individual |
| `--bias_std` | 0.01 | Standard deviation for bias initialization |
| `--w_bits` | 4 | Weight quantization bits (32 = full float) |
| `--local-data` | `true` | Store metrics locally; pass `--local-data false --wandb_project <name>` to use wandb instead |
| `--output`, `-o` | `longpop.npy` | Checkpoint filename saved at the end |

Example: `python para.py --shapes "784,32,10" --population_size 64 -o my_pop.npy`

## Common Gotchas

- **All weight tensors are `requires_grad=False`** — they are fixed genomes; GA handles evolution, no autograd on weights or biases.
- If wandb is not installed, pass `--wandb_project disabled`. `sweep.py` does not use wandb.
- **Loss/metric functions are closures passed as arguments** to every script — they operate on `(pred, target)` tensors with shapes aligned to the population batch dimension: first dim = individual index, second = mini-batch index.
- `sgdtest.py` diverges from the other scripts: uses torchvision DataLoaders, cross-entropy loss directly (not custom celoss), different normalization (`transforms.Normalize((0.1307,), (0.3081,))`).

## Quick Reference — Common Operations

```python
# Evaluate fitness for individuals [start..end] on data subset
fitness = pop_mlp.evaluate(x, y, loss_fn, torch.arange(i, end), batch_idxs)

# Tournament evolution step
pop_mlp.tournaments(x_train, y_train, celoss, bidxs, deme_size, pop_batch,
                    crosstype='uni', mutation_rate=mr, version='local-uniform')

# Test metrics for all individuals
acc, loss = pop_mlp.test(data, labels, torch.arange(pop_size), [accuracy, celoss])

# Save / reload a population
torch.save(pop.state_dict(), 'checkpoint.pt')
pop = PopMLP(sz, shapes, act, oact, wbits, 'linear')  # recreate with same shapes/bits
pop.load_state_dict(torch.load('checkpoint.pt'))
```
