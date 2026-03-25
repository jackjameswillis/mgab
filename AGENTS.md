# AGENTS.md - Guidelines for Agentic Coding

## Build & Test Commands

**No formal test framework** - this is a research codebase with script-based execution.

| Action | Command |
|--------|---------|
| Run main training | `python para.py` |
| Analyze saved population | `python analysis.py` |
| Run SGD baseline | `python sgdtest.py` |
| Check Python version | `python --version` |

**Dependencies:** Install via `pip install -r requirements.txt` (torch, scikit-learn, matplotlib, numpy). Optional: `pip install wandb` for experiment logging.

## Code Style Guidelines

### Imports
- Use **absolute imports**: `from PopMLP import PopMLP`, not relative imports
- Standard library first, then third-party, then local modules
- Group imports logically (standard lib → third party → local)

### Formatting
- **4-space indentation**
- **88-character line limit** (follows black formatter conventions)
- Blank lines: one between functions, two between classes

### Naming Conventions
- **snake_case** for functions, variables, modules: `population_size`, `batch_indices`
- **PascalCase** for classes: `PopMLP`, `Precisions`
- **UPPER_CASE** for constants: `BATCH_SIZE`, `NUM_GENERATIONS`
- Private methods/attributes: single underscore prefix: `_internal_state`

### Type Hints
- Use Python's `typing` module for complex types
- Prefer explicit types over `Any` where possible
- Annotate function parameters and return values

### Error Handling
- **Assertion errors** for internal consistency checks: `assert bits <= 8`
- **ValueError** for invalid user input with descriptive messages
- **try/except** blocks for external I/O operations
- **No silent failures** - raise exceptions with clear messages

### Documentation
- **Triple quotes** for module docstrings (first line of file)
- **Docstrings** for all public functions/classes explaining purpose, parameters, returns
- Inline comments for **non-obvious logic** only
- Keep docstrings concise (1-3 sentence summary)

### PyTorch Conventions
- Move tensors to device: `.to(device)`
- Use `torch.no_grad()` for inference/evaluation
- Use `requires_grad=False` for fixed parameters
- Clone tensors when you need to preserve originals: `.clone()`

### Style Notes
- Prefer **list comprehensions** over `map/filter`
- Use **f-strings** for string formatting
- Use `torch.cat()` for tensor concatenation, not `torch.stack()` unless needed
- Avoid magic numbers; extract to named constants

### Existing Patterns to Follow
- Population-based parallel computation (stack tensors, process all at once)
- Tournament selection with deme-based local competition
- State dict pattern for model serialization (`state_dict()`/`load_state_dict()`)
- Separate precision classes (`f32`, `Q`) for quantization operations

## Project Structure

| File | Description |
|------|-------------|
| `para.py` | Main training script - microbial GA with PopMLP on MNIST |
| `analysis.py` | Analyze saved population - compute accuracy distribution |
| `sgdtest.py` | SGD baseline for comparison - standard PyTorch training |
| `PopMLP.py` | Population MLP module - parallel forward pass for all individuals |
| `precisions.py` | Precision management - f32 and Q (quantized) classes |

## Key Constants & Parameters

- `population_size`: Number of individuals (e.g., 100 or 1000)
- `BATCH_SIZE`: Training batch size for tournament selection (e.g., 64)
- `deme_size`: Local competition neighborhood size
- `pop_batch`: Batch size for evaluating population (e.g., population_size)
- `num_generations`: Evolution iterations (e.g., 5000)
- `mr` (mutation_rate): Per-element mutation rate (e.g., 0.001)
- `bias_std`: Gaussian mutation std for biases (e.g., 0.01)

## Data Pipeline

1. Load MNIST via `sklearn.datasets.fetch_openml`
2. Split 6:1 train/test with `train_test_split(test_size=1/7, random_state=42)`
3. Normalize: `(x - mean) / std` using training set statistics
4. Convert to `torch.FloatTensor` and move to GPU if available

## Common Operations

### Evaluate Fitness
```python
fitness = pop_mlp.evaluate(x, y, loss_fn, batch_indices, batch_idxs)
```

### Run Tournament Selection
```python
pop_mlp.tournaments(x, y, loss_fn, BATCH_SIZE, deme_size, pop_batch,
                    crosstype='uni', mutation_rate=mr, version='local-uniform')
```

### Test Metrics
```python
acc, loss = pop_mlp.test(x, y, batch_indices, [accuracy, celoss])
```

### Save/Load Population
```python
torch.save(pop_mlp.state_dict(), 'checkpoint.npy')
pop_mlp.load_state_dict(torch.load('checkpoint.npy'))
```

## Logging & Experiment Tracking

- Uses `wandb` (Weights & Biases) for experiment tracking
- Logs: train/test loss_mean, loss_max, accuracy_mean, accuracy_max
- Initialize with: `wandb.init(project="your-project-name")`

## PyTorch Best Practices in This Codebase

### Device Management
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
```

### Gradient Control
```python
with torch.no_grad():
    # inference or parameter updates without tracking gradients
    output = model(input)
```

### Parameter Updates
```python
# Use requires_grad=False for fixed parameters (like GA genomes)
weight = nn.Parameter(tensor, requires_grad=False)
```

### Tensor Operations
```python
# Clone to preserve original
original = tensor.clone()

# Concatenate along batch dimension
result = torch.cat([t1, t2], dim=0)

# Reshape for batch processing
batched = x.unsqueeze(0).expand(pop_size, -1, -1)
```

## GPU Usage Notes

- All tensors should be explicitly moved to device: `.to(device)`
- Check GPU availability at startup: `torch.cuda.is_available()`
- Use `map_location` when loading checkpoints on CPU: `torch.load(path, map_location=device)`
- Keep data and models on same device to avoid runtime errors
