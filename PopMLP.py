'''
Here we are implementing a parallel-population microbial genetic algorithm. The idea is that we
can more efficiently compute the fitness of the population by computing the fitness of
each individual in parallel. This is achieved by stacking the population's weights into one tensor
and then doing the forward pass on all individuals at the same time.

First, we implement a torch module that performs the same function as MLP.py, but with the population's
weights stacked into one tensor. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import precisions as P

class PopMLP(nn.Module):

    def __init__(self, population_size, shapes, activation=F.relu, output_activation=None, precision='f32',
                 bias_std=1, mutation_std=1, scale_std=1):
        
        super(PopMLP, self).__init__()
        
        self.population_size = population_size
        self.shapes = shapes
        self.activation = activation
        self.output_activation = output_activation
        self.precision = P.precisions[precision](mutation_std)
        self.bias_precision = P.precisions['f32'](bias_std)
        self.scale_precision = P.precisions['f32'](scale_std)
        self.fitnesses = torch.zeros((self.population_size, 1))
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.scales = nn.ParameterList()
        # Ensure all tensors are created on the correct device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(len(shapes)-1):
            # Create weight matrix with specified precision
            weight_tensor = self.precision.initializer((self.population_size, shapes[i + 1], shapes[i])).to(device=self.device)
            self.weights.append(nn.Parameter(weight_tensor, requires_grad=False))
            # Create bias vector with float32 precision
            bias_tensor = torch.zeros((self.population_size, 1, shapes[i + 1]), device=self.device)
            #bias_tensor = self.bias_precision.mutate(torch.zeros(shapes[i + 1], device=device))
            self.biases.append(nn.Parameter(bias_tensor, requires_grad=False))
            scale_tensor = torch.ones((self.population_size, 1, shapes[i + 1]), device=self.device)
            self.scales.append(nn.Parameter(scale_tensor, requires_grad=False))
        
        self.to(device=self.device)

    def forward(self, x):

        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        for i in range(len(self.weights)):
            x = x @ self.precision.cast_from(self.weights[i]).transpose(-1, -2) + self.biases[i]
            if i < len(self.weights) - 1:
                x = self.activation(x)
        return x

    '''
    pop_data: takes input and output data in shapes (batch_size, input_size), and (batch_size, output_size)
    and returns input data of shape (individuals, batch_size, input_size), and output data of shape (individuals, batch_size, output_size)
    '''
    def pop_data(self, x, y):

        x = x.unsqueeze(0).expand(self.population_size, -1, -1)

        y = y.unsqueeze(0).expand(self.population_size, -1, -1)

        return x, y

    def evaluate(self, x, y, f):
        x_, y_ = self.pop_data(x, y)
        return f(self.forward(x_), y_)
    
    def state_dict(self):
        # Return weights and biases with appropriate casting
        state_dict = {}
        for i in range(len(self.weights)):
            state_dict[f'weights.{i}'] = self.weights[i].clone()
        
        for i in range(len(self.biases)):
            state_dict[f'biases.{i}'] = self.biases[i].clone()

        for i in range(len(self.scales)):
            state_dict[f'scales.{i}'] = self.scales[i].clone()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        # Decide device BEFORE wiping parameters
        device = self.weights[0].device if len(self.weights) > 0 else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Clear and rebuild parameters
        self.weights = nn.ParameterList()
        self.biases  = nn.ParameterList()
        self.scales = nn.ParameterList()

        for key, value in state_dict.items():
            if key.startswith("weights"):
                self.weights.append(nn.Parameter(value.to(device), requires_grad=False))
            elif key.startswith("biases"):
                self.biases.append(nn.Parameter(value.to(device), requires_grad=False))
            elif key.startswith("scales"):
                self.scales.append(nn.Parameter(value.to(device), requires_grad=False))
    
    '''
    Carries out tournaments between individuals so that every individual participates in a tournament.
    First fitness is evaluated in parallel, before tournament pairs are selected.
    This is done by picking a random starting point, then iterating through the population picking
    the tournament partner for the current individual. In picking a partner only an individual within
    the current deme can be selected, and only if they are not already part of a tournament.

    Then crossover and mutation is performed in parallel
    '''
    def tournaments(self, x, y, f, deme_size):

        deme_size -= 1

        self.fitnesses = self.evaluate(x, y, f)

        start = torch.randint(0, self.population_size, (1,), device=self.device).item()

        selected = -torch.ones(self.population_size, device=self.device).to(torch.int)

        won = torch.ones(self.population_size, device=self.device).to(torch.bool)

        for i in range(self.population_size):

            shifted_indecies = torch.arange(start + i, start + i + self.population_size, device=self.device) % self.population_size
                
            if selected[shifted_indecies[0]] == -1:

                deme = shifted_indecies[1:deme_size]

                deme = deme[selected[deme] == -1]

                select = deme[torch.randint(0, len(deme), (1,)).item()]

                selected[shifted_indecies[0]] = select

                selected[select] = shifted_indecies[0]

                if self.fitnesses[shifted_indecies[0]] >= self.fitnesses[select]:

                    won[select] = False
                
                else:
                    
                    won[shifted_indecies[0]] = False
    
        state = self.state_dict()

        for k in state.keys():

            mask = torch.rand(state[k].size(0)//2, state[k].size(1), state[k].size(2), device=self.device) > 0.5

            for i, loser in enumerate(selected[won]):

                state[k][loser] = torch.where(mask[i], state[k][selected[loser]], state[k][loser])

            #state[k][won.logical_not()] = torch.where(mask, state[k][won], state[k][won.logical_not()])

            if 'weight' in k:

                state[k][won.logical_not()] = self.precision.mutate(state[k][won.logical_not()])
            
            if 'bias' in k:

                state[k][won.logical_not()] = self.bias_precision.mutate(state[k][won.logical_not()])
            
            if 'scale' in k:

                state[k][won.logical_not()] = self.scale_precision.mutate(state[k][won.logical_not()])
        
        self.load_state_dict(state)