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
import math
import geography as G

class PopMLP(nn.Module):

    def __init__(self, population_size, shapes, activation=F.relu, output_activation=None, w_bits=32, r='scale', smoothBeta=0):
        
        super(PopMLP, self).__init__()
        
        self.population_size = population_size
        self.shapes = shapes
        self.activation = activation
        self.output_activation = output_activation
        self.Q = P.Q(w_bits) if w_bits != 32 else P.f32()
        self.fitnesses = torch.zeros((self.population_size, 1))
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.ranges = []
        self.r = r
        self.smoothBeta = smoothBeta
        # Ensure all tensors are created on the correct device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(len(shapes)-1):
            s = math.sqrt(6/(shapes[i]))
            if self.Q.bits != 32:
                weight_tensor = torch.randint(-(2**self.Q.bits)//2, (2**self.Q.bits)//2, (self.population_size, shapes[i + 1], shapes[i]), dtype=torch.int8, device=self.device)
            else:
                weight_tensor = torch.rand((self.population_size, shapes[i + 1], shapes[i]), device=self.device)
                mini, maxi = 0, 1
                weight_tensor = -s + weight_tensor * 2 * s
            # Create weight matrix with specified precision
            #weight_tensor = self.precision.initializer((self.population_size, shapes[i + 1], shapes[i])).to(device=self.device)
            self.weights.append(nn.Parameter(weight_tensor, requires_grad=False))
            if r=='scale': 
                self.ranges.append(s)
            elif r=='linear':
                self.ranges.append(0.0)
            # Create bias vector with float32 precision
            bias_tensor = torch.zeros((self.population_size, 1, shapes[i + 1]), device=self.device)
            #bias_tensor = self.bias_precision.mutate(torch.zeros(shapes[i + 1], device=device))
            self.biases.append(nn.Parameter(bias_tensor, requires_grad=False))
        
        self.to(device=self.device)

    def forward(self, x, batch, verbose=False):

        device = next(self.parameters()).device
        x_ = x[batch]
        acts = []
        if x_.device != device:
            x_ = x_.to(device)
        with torch.no_grad():
            for i in range(len(self.weights)):
                # Proper matrix multiplication with correct tensor shapes
                x_ = torch.matmul(x_, self.Q.cast_from(self.weights[i][batch], self.ranges[i]).transpose(-1, -2))
                '''
                if i < len(self.weights) - 1: 
                    #x_ = F.batch_norm(x_, running_mean=None, running_var=None, weight=None, bias=None, training=True, momentum=0.1, eps=1e-05)
                    mean = x_.mean(dim=-1, keepdim=True)
                    var = x_.var(dim=-1, keepdim=True)
                    eps =  1e-5
                    x_ = (x_ - mean) / (var + eps).sqrt()
                '''
                x_ =  x_+ self.biases[i][batch]
                if i < len(self.weights) - 1:
                    x_ = self.activation(x_)
                if verbose:
                    acts += [x_.clone()]
                # Add batch normalization with no learnable parameters
                
            if not verbose: return x_

            return acts

    '''
    pop_data: takes input and output data in shapes (batch_size, input_size), and (batch_size, output_size)
    and returns input data of shape (individuals, batch_size, input_size), and output data of shape (individuals, batch_size, output_size)
    '''
    def pop_data(self, x, y):

        x = x.unsqueeze(0).expand(self.population_size, -1, -1)

        y = y.unsqueeze(0).expand(self.population_size, -1, -1)

        return x, y

    def evaluate(self, x, y, f, batch, batch_idxs=None):
        #if batch_idxs:
        x_, y_ = x[batch_idxs], y[batch_idxs]
        #else:
            #x_, y_ = self.pop_data(x, y)
            # Return fitness values for the specified batch
        return f(self.forward(x_, batch), y_[batch])
    
    def test(self, x, y, batch, metrics):
        x_, y_ = self.pop_data(x, y)
        r = []
        for m in metrics:
            r.append(m(self.forward(x_, batch), y_[batch]).flatten())
        return r
    
    def state_dict(self):
        # Return weights and biases with appropriate casting
        state_dict = {}
        for i in range(len(self.weights)):
            state_dict[f'weights.{i}'] = self.weights[i].clone()
        
        for i in range(len(self.biases)):
            state_dict[f'biases.{i}'] = self.biases[i].clone()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        # Decide device BEFORE wiping parameters
        device = self.weights[0].device if len(self.weights) > 0 else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Clear and rebuild parameters
        self.weights = nn.ParameterList()
        self.biases  = nn.ParameterList()
        self.ranges = []

        for key, value in state_dict.items():
            if key.startswith("weights"):
                self.weights.append(nn.Parameter(value.to(device), requires_grad=False))
                if self.r=='scale':
                    s = math.sqrt(6/(self.weights[-1].shape[-1]))
                elif self.r == 'linear':
                    s = 0.0
                self.ranges.append(s)
            elif key.startswith("biases"):
                self.biases.append(nn.Parameter(value.to(device), requires_grad=False))
    
    '''
    Carries out tournaments between individuals so that every individual participates in a tournament.
    First fitness is evaluated in parallel, before tournament pairs are selected.
    This is done by picking a random starting point, then iterating through the population picking
    the tournament partner for the current individual. In picking a partner only an individual within
    the current deme can be selected, and only if they are not already part of a tournament.

    Then crossover and mutation is performed in parallel
    '''
    def tournaments(self, x, y, f, bidxs, deme_type, deme_size, pop_batch_size, 
                    crosstype='uni', bias_std=0.01, mutation_rate=0.001, 
                    version='local-uniform'):

        if deme_type == 'Ring':

            D = G.Ring(self.population_size, x.device)
            
            selected = D.tournament(deme_size)
        
        elif deme_type == 'SmallWorld':

            D = G.SmallWorld(self.population_size, x.device, k=4, p=0.1)

        won = torch.ones(self.population_size, device=self.device).to(torch.bool)

        batch_idxs = torch.stack([bidxs]*self.population_size, dim=0)

        self.fitnesses = torch.zeros(self.population_size, device=self.device)

        for i in range(0, self.population_size, pop_batch_size):

            end = min(i + pop_batch_size, self.population_size)
            fitness_batch = self.evaluate(x, y, f, torch.arange(i, end, device=self.device), batch_idxs)
            self.fitnesses[i:end] = self.fitnesses[i:end] * self.smoothBeta + fitness_batch.flatten() * (1 - self.smoothBeta)

        for i in range(self.population_size):

            if self.fitnesses[i] >= self.fitnesses[selected[i]]:

                won[selected[i]] = False

                won[i] = True
                    
            else:
                        
                won[i] = False

                won[selected[i]] = True
    
        state = self.state_dict()

        for k in state.keys():

            losers = selected[won]
            winners = selected[won.logical_not()]

            if crosstype == 'uni':

                mask = torch.rand(state[k].size(0)//2, state[k].size(1), state[k].size(2), device=self.device) > 0.5
                
                # For each winner-loser pair, perform crossover
                state[k][losers] = torch.where(mask, state[k][winners], state[k][losers])

            elif crosstype == 'asexual':
                
                state[k][losers] = state[k][winners].clone()
        
        
        for k in state.keys():
            # Apply mutation to losers
                
            if 'weight' in k:
                state[k][losers] = self.Q.mutate(state[k][losers], mutation_rate, version)
            elif 'bias' in k:
                state[k][losers] = P.f32().mutate(state[k][losers], bias_std)
        
        self.load_state_dict(state)