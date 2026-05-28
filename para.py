'''
An implementation of the microbial genetic algorithm using PopMLP for mixed-precision genomes.
This file concentrates on the MNIST dataset for image classification using parallel population approach.
'''

import json
import torch
from PopMLP import PopMLP
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import argparse


def parse_list(s):
    """Parse a string like '784,64,10' into a list of ints."""
    return [int(x.strip()) for x in s.split(',')]


def get_activation(name):
    activations = {
        'relu': torch.relu,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'elu': F.elu,
        'silu': F.silu,
        'linear': lambda x: x,
    }
    fn = activations.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown activation: {name} (choose from {list(activations.keys())})")
    return fn


parser = argparse.ArgumentParser(description='Train microbial GA population')

# Network architecture
parser.add_argument('--shapes', type=parse_list, default=[784, 64, 10],
                    help='Layer sizes as comma-separated list (e.g. "784,64,10")')
parser.add_argument('--act', type=str, default='relu',
                    help='Activation function: relu, tanh, sigmoid, elu, silu, linear')

# Population & evolution
parser.add_argument('--population_size', type=int, default=32)
parser.add_argument('--num_generations', type=int, default=1000)
parser.add_argument('--BATCH_SIZE', type=int, default=64)
parser.add_argument('--pop_batch', type=int, default=None,
                    help='Batch size for PopMLP operations (default: same as population_size)')
parser.add_argument('--demesize', type=int, default=2)
parser.add_argument('--mutation_rate', '-mr', type=float, default=0.001)
parser.add_argument('--bias_std', type=float, default=0.01)
parser.add_argument('--w_bits', type=int, default=4,
                    help='Weight quantization bits (32 = no quantization/float init)')

# Logging & output
parser.add_argument('--wandb_project', type=str, default='mga-proper-migration',
                    help='Weights & Biases project name (only used when --local-data false)')
parser.add_argument('--local-data', action='store_true', default=True,
                    help='Store metrics locally instead of sending to wandb (default: true). Pass --wandb to use wandb.')
parser.add_argument('--wandb', dest='local_data', action='store_false',
                    help='Use wandb for logging instead of local storage')
parser.add_argument('--output', '-o', type=str, default='longpop.npy',
                    help='Output checkpoint filename (default: longpop.npy)')

args = parser.parse_args()

activation_fn = get_activation(args.act)
pop_batch = args.pop_batch or args.population_size

# Setup logging: local JSON (default) or wandb when --local-data false
logger_name = args.output.rstrip('.npy').rstrip('.pth') if args.output else 'para'
metrics_file = f'{logger_name}_metrics.json'

if not args.local_data:
    if args.wandb_project != 'disabled':
        try:
            import wandb
            wandb.init(project=args.wandb_project,
                       settings=wandb.Settings(code_dir="."))
        except ImportError:
            raise ImportError("wandb package not installed. Use --local-data or install wandb.")
    logger = {"type": "wandb"}
else:
    # Initialize local metrics file as empty list
    with open(metrics_file, 'w') as f:
        json.dump([], f)
    logger = {"type": "file", "path": metrics_file}

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Convert labels to one-hot encoding
y = y.astype(int)
y_onehot = np.zeros((y.shape[0], 10))
y_onehot[np.arange(y.shape[0]), y] = 1

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=1/7, random_state=42)

# Convert to torch tensors
x_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

xm = x_train.mean()
xstd = x_train.std()

# Normalize the input data
x_train = (x_train - xm) / xstd
x_test = (x_test - xm) / xstd

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Check for GPU availability and move tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

shapes = args.shapes
output_activation = lambda x: x  # sigmoid already gives class probs; keep linear for CEL

# Create PopMLP instance for the population
pop_mlp = PopMLP(args.population_size,
                 shapes,
                 activation_fn,
                 output_activation,
                 args.w_bits,
                 'scale',
                 b1=0)


def celoss(logits, targets):
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1, targets.size(-1))

    loss_per_sample = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    loss_per_sample = loss_per_sample.reshape(logits.size(0), logits.size(1))

    return -loss_per_sample.mean(dim=1)

def accuracy(logits, targets):
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1, targets.size(-1))

    acc_per_sample = (logits_flat.argmax(dim=1) == targets_flat.argmax(dim=1)).float()

    acc_per_sample = acc_per_sample.reshape(logits.size(0), logits.size(1))

    return acc_per_sample.mean(dim=1)

# Evolution loop
for generation in range(args.num_generations):
    batch_indices = torch.randperm(len(x_train))[:args.BATCH_SIZE]
    for b in range(0, len(x_train), args.BATCH_SIZE):
        idxs = torch.arange(len(x_train))
        bidxs = idxs[b:b+args.BATCH_SIZE]
        pop_mlp.tournaments(x_train,
                            y_train,
                            celoss,
                            bidxs,
                            args.demesize,
                            pop_batch,
                            'uni',
                            mutation_rate=args.mutation_rate,
                            bias_std=args.bias_std,
                            version='local-uniform',
                            rewire=0.1)

    if True:
        train_accs = torch.zeros(args.population_size, device=device)
        train_loss = torch.zeros(args.population_size, device=device)
        for i in range(0, args.population_size, pop_batch):
            end = min(i + pop_batch, args.population_size)
            a, l = pop_mlp.test(x_train[batch_indices],
                                y_train[batch_indices],
                                torch.arange(i, end, device=device),
                                [accuracy, celoss])
            train_accs[i:end] = a
            train_loss[i:end] = l

        test_accs = torch.zeros(args.population_size, device=device)
        test_loss = torch.zeros(args.population_size, device=device)
        for i in range(0, args.population_size, pop_batch):
            end = min(i + pop_batch, args.population_size)
            a, l = pop_mlp.test(x_test[:1000],
                                y_test[:1000],
                                torch.arange(i, end, device=device),
                                [accuracy, celoss])
            test_accs[i:end] = a
            test_loss[i:end] = l

        train_loss_mean = torch.mean(train_loss).item()
        train_loss_max = torch.max(train_loss).item()
        test_loss_mean = torch.mean(test_loss).item()
        test_loss_max = torch.max(test_loss).item()
        train_acc_mean = torch.mean(train_accs).item()
        train_acc_max = torch.max(train_accs).item()
        test_acc_mean = torch.mean(test_accs).item()
        test_acc_max = torch.max(test_accs).item()

        metrics = {
            "epoch": generation,
            "train_loss_mean": train_loss_mean,
            "train_loss_max": train_loss_max,
            "test_loss_mean": test_loss_mean,
            "test_loss_max": test_loss_max,
            "train_acc_mean": train_acc_mean,
            "train_acc_max": train_acc_max,
            "test_acc_mean": test_acc_mean,
            "test_acc_max": test_acc_max
        }

        if logger["type"] == "wandb":
            import wandb as wb
            wb.log({
                "train_loss_mean": train_loss_mean,
                "train_loss_max": train_loss_max,
                "test_loss_mean": test_loss_mean,
                "test_loss_max": test_loss_max,
                "train_acc_mean": train_acc_mean,
                "train_acc_max": train_acc_max,
                "test_acc_mean": test_acc_mean,
                "test_acc_max": test_acc_max
            })
        else:
            with open(logger["path"], 'r') as f:
                saved = json.load(f)
            saved.append(metrics)
            with open(logger["path"], 'w') as f:
                json.dump(saved, f, indent=2)

        print(f'Epoch: {generation}')

torch.save(pop_mlp.state_dict(), args.output)
