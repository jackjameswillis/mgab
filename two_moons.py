'''
An implementation of the microbial genetic algorithm for mixed-precision genomes.
This file mainly concentrates on the scale of the population, as the precisions
file manages the precision specification including mutation operators.
'''

import torch
from MLP import MLP
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from mga import MGA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Generate two moons dataset
X, y = make_moons(n_samples=250, noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
x_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
x_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Define network parameters
shapes = [2, 64, 64, 1]
activation = torch.tanh
output_activation = lambda x: x
precision = 'i2'
bias_std = 1
mutation_std = 0.01
scale_std = 0.

# Initialize MGA
population_size = 10
num_generations = 1000
mutation_rate = 0 # Defaults to 1/num_parameters
mga = MGA(population_size, num_generations, mutation_rate, population_size)

# Initialize population
mga.initialize_population(shapes, activation, output_activation, precision, bias_std, mutation_std, scale_std, x_train, y_train, torch.nn.BCEWithLogitsLoss())

# Evolution loop
best_fitness_history = []
best_individuals = []

for generation in range(num_generations):
    print(f"Generation {generation}")
    best_fitness = float('-inf')
    for i in range(population_size):
        # Tournament selection and evolution
        fitness, t = mga.tournament(x_train, y_train, torch.nn.BCEWithLogitsLoss(), mutation_rate)
        if fitness > best_fitness:
            best_fitness = fitness

    best_fitness_history.append(best_fitness)

    # Store the best individual of this generation
    best_individual = max(mga.population, key=lambda x: x.fitness)
    best_individuals.append(best_individual)

    print(f"Best fitness: {best_fitness}")

# Evaluate on test set
final_best_individual = max(mga.population, key=lambda x: x.fitness)
test_loss = torch.nn.BCEWithLogitsLoss()(final_best_individual(x_test), y_test).item()
print(f"Test loss of best individual: {test_loss}")

# Plot decision boundary of the final best individual
fig, ax = plt.subplots(figsize=(8, 6))

def plot_decision_boundary(model, X, y, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    with torch.no_grad():
        Z = torch.sigmoid(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])))
        Z = Z.reshape(xx.shape)

    # Use a colormap similar to the reference code
    cmap_bg = plt.cm.RdBu_r  # Similar to palettable.colorbrewer.diverging.RdBu_5_r
    cmap_fg = plt.cm.coolwarm  # Similar to the reference's cmap_fg

    # Plot filled contours with transparency
    ax.contourf(xx, yy, Z.numpy(), levels=50, alpha=0.7, cmap=cmap_bg, antialiased=True)

    # Plot data points
    yf = y.flatten()
    ax.scatter(X[yf == 0, 0], X[yf == 0, 1], color=cmap_fg(0), s=100, lw=1.5, edgecolors='black', marker='s')
    ax.scatter(X[yf == 1, 0], X[yf == 1, 1], color=cmap_fg(0.999), s=100, lw=1.5, edgecolors='black')
    ax.set_title("Decision Boundary of Best Individual", fontsize=24)
    ax.set_xlabel("Feature 1", fontsize=18)
    ax.set_ylabel("Feature 2", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

plot_decision_boundary(final_best_individual, X_train, y_train, ax)
plt.savefig(f'decision_boundary_final_{precision}.png')
plt.close()




# Plot fitness history
plt.plot(torch.Tensor(best_fitness_history).detach().numpy())
plt.title("Best Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig(f'fitness_history_{precision}.png')
plt.show()
