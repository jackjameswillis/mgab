'''
An implementation of the microbial genetic algorithm for mixed-precision genomes. 
This file mainly concentrates on the scale of the population, as the precisions
file manages the precision specification including mutation operators.
'''

import torch
from MLP import MLP

class MGA:
    def __init__(self, population_size, num_generations, deme_size=None):
        self.population_size = population_size
        self.num_generations = num_generations
        self.deme_size = deme_size if deme_size is not None else self.population_size
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness_history = []
    
    def initialize_population(self, shapes, activation, output_activation, precision, bias_std, mutation_std, scale_std, x, y, f):
        # Move data to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x, y = x.to(device), y.to(device)

        for _ in range(self.population_size):
            model = MLP(shapes, activation, output_activation, precision, bias_std, mutation_std, scale_std)
            model = model.to(device)  # Move model to GPU
            self.population.append(model)
            self.population[-1].evaluate(x, y, f)

    def tournament(self, x, y, f, test=False):

        A = torch.randint(0, self.population_size, (1,))[0].item()
        B = (A + 1 + torch.randint(0, self.deme_size-1, (1,))[0].item()) % self.population_size
        if self.population[A].fitness >= self.population[B].fitness:
            W = A
            L = B
        else:
            W = B
            L = A

        self.population[L].load_state_dict(self.population[L].crossover(self.population[W].state_dict()))
        self.population[L].load_state_dict(self.population[L].mutate())
        self.population[L].evaluate(x, y, f)
        t = None
        if test:
            t = -f(self.population[W].forward(test[0]), test[1]).sum()
        return self.population[W].fitness, t