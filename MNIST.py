'''
An implementation of the microbial genetic algorithm for mixed-precision genomes.
This file concentrates on the MNIST dataset for image classification.
'''

import torch
from MLP import MLP
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mga import MGA
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Convert labels to one-hot encoding
y = y.astype(int)
y_onehot = np.zeros((y.shape[0], 10))
y_onehot[np.arange(y.shape[0]), y] = 1

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

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
shapes = [784, 128, 10]
activation = torch.relu
output_activation = lambda x: x
precision = 'i4'
bias_std = 1
mutation_std = 1

# Initialize MGA
population_size = 100
num_generations = 500
mutation_rate = 0 # Defaults to 1/num_parameters
BATCH_SIZE = 1000
mga = MGA(population_size, num_generations, mutation_rate)

# Initialize population
mga.initialize_population(shapes, activation, output_activation, precision, bias_std, mutation_std, x_train, y_train, torch.nn.CrossEntropyLoss())
best_tests = []
# Evolution loop
for generation in range(num_generations):
    print(f"Generation {generation}")
    best_fitness = float('-inf')
    best_test = float('-inf')
    for i in range(population_size):
        # Tournament selection and evolution with random batch
        batch_indices = torch.randperm(len(x_train))[:BATCH_SIZE]
        fitness, t = mga.tournament(x_train[batch_indices], y_train[batch_indices], torch.nn.CrossEntropyLoss(), mutation_rate, test=False)
        if fitness > best_fitness:
            best_fitness = fitness
            best_test = t
    best_tests += [best_test]

    mga.best_fitness_history.append(best_fitness)
    print(f"Best fitness: {best_fitness}")
    print(f"Best test: {best_test}")

# Evaluate on test set
best_individual = max(mga.population, key=lambda x: x.fitness)
#test_loss = torch.nn.CrossEntropyLoss()(best_individual(x_test), y_test).item()
#test_accuracy = (best_individual(x_test).argmax(dim=1) == y_test.argmax(dim=1)).float().mean().item()

train_loss = torch.nn.CrossEntropyLoss()(best_individual(x_train), y_train).item()
train_accuracy = (best_individual(x_train).argmax(dim=1) == y_train.argmax(dim=1)).float().mean().item()


print(f"Test loss of best individual: {test_loss}")
print(f"Test accuracy of best individual: {test_accuracy * 100:.2f}%")
print(f"Train loss of best individual: {train_loss}")
print(f"Train accuracy of best individual: {train_accuracy * 100:.2f}%")

# Plot fitness history

plt.plot(torch.Tensor(mga.best_fitness_history).detach().cpu().numpy())
plt.plot(torch.Tensor(best_tests).detach().cpu().numpy())
plt.legend(["Best Fitness", "Best Test"])
plt.grid(True)
plt.title("Best Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig(f'fitness_history_mnist_{precision}.png')
plt.show()
