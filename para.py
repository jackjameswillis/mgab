'''
An implementation of the microbial genetic algorithm using PopMLP for mixed-precision genomes.
This file concentrates on the MNIST dataset for image classification using parallel population approach.
'''

import torch
from PopMLP import PopMLP
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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

# Define network parameters for MNIST
shapes = [784, 64, 10]
activation = torch.relu
output_activation = lambda x: x
w_bits = 4
mr = 0.001
bias_std = 0.01
adap = 0.0

# Initialize MGA with PopMLP
population_size = 32
pop_batch = population_size
num_generations = 1000
BATCH_SIZE = 64
hill_iters = 0

# Create PopMLP instance for the population
pop_mlp = PopMLP(population_size, 
                 shapes, 
                 activation, 
                 output_activation, 
                 w_bits, 
                 'scale', 
                 smoothBeta=0)

def celoss(logits, targets):

    logits_flat = logits.reshape(-1, logits.size(-1))  # (networks*batch_size, classes)
    targets_flat = targets.reshape(-1, targets.size(-1))  # (networks*batch_size, classes)
    
    # Compute loss for all networks at once
    loss_per_sample = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # Reshape back to (networks, batch_size) and compute mean per network
    loss_per_sample = loss_per_sample.reshape(logits.size(0), logits.size(1))  # (networks, batch_size)
    
    # Return mean loss for each network
    return -loss_per_sample.mean(dim=1)

def accuracy(logits, targets):

    logits_flat = logits.reshape(-1, logits.size(-1))  # (networks*batch_size, classes)
    targets_flat = targets.reshape(-1, targets.size(-1))  # (networks*batch_size, classes)

    acc_per_sample = (logits_flat.argmax(dim=1) == targets_flat.argmax(dim=1)).float()

    acc_per_sample = acc_per_sample.reshape(logits.size(0), logits.size(1))

    return acc_per_sample.mean(dim=1)

metrics = {
    "train":{
        "loss":[],
        "accuracy":[]
    },
    "test":{
        "loss":[],
        "accuracy":[]
    }
}

import wandb
wandb.init(project="mga-epochs")
demesize = population_size
# Evolution loop
for generation in range(num_generations):
    batch_indices = torch.randperm(len(x_train))[:BATCH_SIZE]
    for b in range(0, len(x_train), BATCH_SIZE):
        idxs = torch.arange(len(x_train))
        bidxs = idxs[b:b+BATCH_SIZE]
        pop_mlp.tournaments(x_train, 
                        y_train, 
                        celoss, 
                        bidxs,
                        'Ring',
                        demesize, 
                        pop_batch,
                        'uni',
                        mutation_rate=mr,
                        bias_std=bias_std,
                        version='local-uniform')
    '''
    if generation == 5000 and generation != 0: 
        mutation_std = mutation_std/10
        pop_mlp.set_precision_m(mutation_std)
    '''
    #if generation % 1000 == 0 and generation != 0: mr *= 0.9
    #if generation % 2500 == 0 and generation != 0: mutation_std = mutation_std/10
    
    #if generation % 10 == 0:
    if True:
        train_accs = torch.zeros(population_size, device=device)
        train_loss = torch.zeros(population_size, device=device)
        for i in range(0, population_size, pop_batch):
            end = min(i + pop_batch, population_size)
            a, l = pop_mlp.test(x_train[batch_indices], 
                                y_train[batch_indices], 
                                torch.arange(i, end, device=device), 
                                [accuracy, celoss])
            train_accs[i:end] = a
            train_loss[i:end] = l

        test_accs = torch.zeros(population_size, device=device)
        test_loss = torch.zeros(population_size, device=device)
        for i in range(0, population_size, pop_batch):
            end = min(i + pop_batch, population_size)
            a, l = pop_mlp.test(x_test[:1000], 
                                y_test[:1000], 
                                torch.arange(i, end, device=device), 
                                [accuracy, celoss])
            test_accs[i:end] = a
            test_loss[i:end] = l

        # Track mean and max of the metrics
        train_loss_mean = torch.mean(train_loss).item()
        train_loss_max = torch.max(train_loss).item()
        test_loss_mean = torch.mean(test_loss).item()
        test_loss_max = torch.max(test_loss).item()
        train_acc_mean = torch.mean(train_accs).item()
        train_acc_max = torch.max(train_accs).item()
        test_acc_mean = torch.mean(test_accs).item()
        test_acc_max = torch.max(test_accs).item()

        # Log to Weights & Biases

        wandb.log({
            "train_loss_mean": train_loss_mean,
            "train_loss_max": train_loss_max,
            "test_loss_mean": test_loss_mean,
            "test_loss_max": test_loss_max,
            "train_acc_mean": train_acc_mean,
            "train_acc_max": train_acc_max,
            "test_acc_mean": test_acc_mean,
            "test_acc_max": test_acc_max
        })
        print(f'Epoch: {generation}')

torch.save(pop_mlp.state_dict(), 'longpop.npy')
