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
shapes = [784, 100, 10]
activation = torch.tanh
output_activation = lambda x: x
precision = 'i4'
bias_std = 0.5
mutation_std = 1
scale_std = 0.

# Initialize MGA with PopMLP
population_size = 100
num_generations = 1000
BATCH_SIZE = 1000

# Create PopMLP instance for the population
pop_mlp = PopMLP(population_size, shapes, activation, output_activation, precision, bias_std, mutation_std, scale_std)

# Initialize best fitness history
best_fitness_history = []
best_tests = []

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

# Evolution loop
for generation in range(num_generations):
    batch_indices = torch.randperm(len(x_train))[:BATCH_SIZE]
    pop_mlp.tournaments(x_train[batch_indices], y_train[batch_indices], celoss, population_size)
    accs = pop_mlp.evaluate(x_test[:1000], y_test[:1000], accuracy)
    print(f'Generation: {generation} | Mean: {pop_mlp.fitnesses.mean()} | Max: {pop_mlp.fitnesses.max()} | Test Accuracy Mean: {accs.mean()} | Test Accuracy Max: {accs.max()}')
    