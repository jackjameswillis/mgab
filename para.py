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
bias_std = 0.
mutation_std = torch.pi/20
scale_std = 0.

# Initialize MGA with PopMLP
population_size = 100
num_generations = 1000
BATCH_SIZE = 1000

# Create PopMLP instance for the population
pop_mlp = PopMLP(population_size, shapes, activation, output_activation, precision, bias_std, mutation_std, scale_std)

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

best_fitness_history = []
mean_fitness_history = []

best_tests = []
mean_test_history = []

# Evolution loop
for generation in range(num_generations):
    batch_indices = torch.randperm(len(x_train))[:BATCH_SIZE]
    pop_mlp.tournaments(x_train[batch_indices], y_train[batch_indices], celoss, population_size)
    accs = pop_mlp.evaluate(x_test[:1000], y_test[:1000], accuracy)
    if generation % 10 == 0: print(f'Generation: {generation} | Mean: {pop_mlp.fitnesses.mean()} | Max: {pop_mlp.fitnesses.max()} | Test Accuracy Mean: {accs.mean()} | Test Accuracy Max: {accs.max()}')
    
    # Track metrics
    best_fitness_history.append(pop_mlp.fitnesses.max().item())
    mean_fitness_history.append(pop_mlp.fitnesses.mean().item())
    best_tests.append(accs.max().item())
    mean_test_history.append(accs.mean().item())

# Plot and save metrics
plt.figure(figsize=(12, 5))

# Plot fitness history
plt.subplot(1, 2, 1)
plt.plot(best_fitness_history)
plt.plot(mean_fitness_history)
plt.plot()
plt.title('Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')

# Plot test accuracy history
plt.subplot(1, 2, 2)
plt.plot(best_tests)
plt.plot(mean_test_history)
plt.title('Best Test Accuracy Over Generations')
plt.xlabel('Generation')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('mga_metrics.png')
plt.close()

print("Metrics plots saved as 'mga_metrics.png'")

