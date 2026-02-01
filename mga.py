'''
An implementation of the microbial genetic algorithm for mixed-precision genomes. 
This file mainly concentrates on the scale of the population, as the precisions
file manages the precision specification including mutation operators.
'''

import torch
from MLP import MLP

class MGA:
    def __init__(self, population_size, num_generations, mutation_rate):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness_history = []
    
    def initialize_population(self, shapes, activation, output_activation, precision, bias_std, mutation_std, x, y, f):
        # Move data to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, y = x.to(device), y.to(device)

        for _ in range(self.population_size):
            model = MLP(shapes, activation, output_activation, precision, bias_std, mutation_std)
            model = model.to(device)  # Move model to GPU
            self.population.append(model)
            self.population[-1].evaluate(x, y, f)

    def tournament(self, x, y, f, mutation_rate):

        A, B = torch.arange(self.population_size)[torch.randperm(self.population_size)[:2]]
        if self.population[A].fitness >= self.population[B].fitness:
            W = A
            L = B
        else:
            W = B
            L = A

        self.population[L].load_state_dict(self.population[L].crossover(self.population[W].state_dict()))
        self.population[L].load_state_dict(self.population[L].mutate(mutation_rate))
        self.population[L].evaluate(x, y, f)
        return self.population[W].fitness