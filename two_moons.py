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
precision = 'f32'
bias_std = 1
mutation_std = 1

# Initialize MGA
population_size = 5
num_generations = 1000
mutation_rate = 0 # Defaults to 1/num_parameters
mga = MGA(population_size, num_generations, mutation_rate)

# Initialize population
mga.initialize_population(shapes, activation, output_activation, precision, bias_std, mutation_std, x_train, y_train, torch.nn.BCEWithLogitsLoss())

# Evolution loop
for generation in range(num_generations):
    print(f"Generation {generation}")
    best_fitness = float('-inf')
    for i in range(population_size):
        # Tournament selection and evolution
        fitness = mga.tournament(x_train, y_train, torch.nn.BCEWithLogitsLoss(), mutation_rate)
        if fitness > best_fitness:
            best_fitness = fitness

    mga.best_fitness_history.append(best_fitness)
    print(f"Best fitness: {best_fitness}")

# Evaluate on test set
best_individual = max(mga.population, key=lambda x: x.fitness)
test_loss = torch.nn.BCEWithLogitsLoss()(best_individual(x_test), y_test).item()
print(f"Test loss of best individual: {test_loss}")

# Visualize the decision boundary of the best individual
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z.numpy(), levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z.numpy(), levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze().numpy(), cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title("Decision Boundary of Best Individual")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f'decision_boundary_{precision}.png')  # Save the plot locally
    plt.show()

#plot_decision_boundary(best_individual, X_train, y_train)

# Plot fitness history
plt.plot(torch.Tensor(mga.best_fitness_history).detach().numpy())
plt.title("Best Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.savefig(f'fitness_history_{precision}.png')
plt.show()
