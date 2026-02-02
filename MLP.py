'''
A python script implementing a mixed-precision MLP that can be optimized
via a genetic algorithm
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import precisions as P
import math


'''
An MLP implementation that allows for the precision of the weights to be specified
and for float32 biases to be used. This class doesn't track gradients at any points
as it is used for genetic algorithm optimization.
'''
class MLP(nn.Module):
    def __init__(self, shapes, activation=F.relu, output_activation=None, precision='f32',
                    bias_std=1, mutation_std=1, scale=True):
        
        super(MLP, self).__init__()
        self.shapes = shapes
        self.activation = activation
        self.output_activation = output_activation
        assert precision in P.precisions.keys()
        self.precision = P.precisions[precision](mutation_std)
        self.bias_precision = P.f32(bias_std)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.fitness = float('-inf')
        self.scale = True

        # Ensure all tensors are created on the correct device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(len(shapes)-1):
            # Create weight matrix with specified precision
            weight_tensor = self.precision.mutate(torch.zeros(shapes[i + 1], shapes[i], device=device))
            self.weights.append(nn.Parameter(weight_tensor, requires_grad=False))
            # Create bias vector with float32 precision
            bias_tensor = self.bias_precision.mutate(torch.zeros(shapes[i + 1], device=device))
            self.biases.append(nn.Parameter(bias_tensor, requires_grad=False))
        
        self.param_count = sum(w.numel() + b.numel() for w, b in zip(self.weights, self.biases))
        # Move the module to the appropriate device
        self.to(device)
    
    def forward(self, x):
        # Ensure input is on the correct device
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)

        for i in range(len(self.weights)):
            x = torch.matmul(x, self.precision.cast_from(self.weights[i]).T)
            if (self.scale):
                x = x/math.sqrt(2*x.shape[1])
            
            x = x + self.bias_precision.cast_from(self.biases[i])
            
            if (i < len(self.weights) - 1):
                x = self.activation(x)
        return x
    '''
    def load_state_dict(self, state_dict):
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        
        # Load weights and biases from state dict with appropriate casting
        device = next(self.parameters()).device
        for i, (key, value) in enumerate(state_dict.items()):
            if 'weights' in key:
                weight_tensor = value.to(device)
                self.weights.append(nn.Parameter(weight_tensor))
                self.weights[-1].requires_grad_(False)
            
            if 'biases' in key:
                bias_tensor = value.to(device)
                self.biases.append(nn.Parameter(bias_tensor))
                self.biases[-1].requires_grad_(False)
    '''
    def load_state_dict(self, state_dict):
        # Decide device BEFORE wiping parameters
        device = self.weights[0].device if len(self.weights) > 0 else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Clear and rebuild parameters
        self.weights = nn.ParameterList()
        self.biases  = nn.ParameterList()

        for key, value in state_dict.items():
            if key.startswith("weights"):
                self.weights.append(nn.Parameter(value.to(device), requires_grad=False))
            elif key.startswith("biases"):
                self.biases.append(nn.Parameter(value.to(device), requires_grad=False))


    def state_dict(self):
        # Return weights and biases with appropriate casting
        state_dict = {}
        for i in range(len(self.weights)):
            state_dict[f'weights.{i}'] = self.weights[i]
        
        for i in range(len(self.biases)):
            state_dict[f'biases.{i}'] = self.biases[i]
        
        return state_dict
    
    def crossover(self, state_dict1):
        state_dict2 = self.state_dict()
        # Perform crossover between two state dicts
        crossover_state_dict = {}
        device = next(self.parameters()).device
        for k in state_dict1.keys():
            mask = torch.rand_like(state_dict1[k].to(torch.float32)) > 0.5
            crossover_state_dict[k] = torch.where(mask, state_dict1[k], state_dict2[k])
        
        return crossover_state_dict
    
    def mutate(self, mutation_rate=0):
        if mutation_rate == 0: mutation_rate = 1/self.param_count
        state_dict = self.state_dict()
        # Perform mutation on a state dict
        mutated_state_dict = {}
        device = next(self.parameters()).device
        for k in state_dict.keys():
            op = self.precision.mutate if 'weight' in k else self.bias_precision.mutate
            mask = torch.rand_like(state_dict[k].to(torch.float32)) < mutation_rate
            mutated_state_dict[k] = state_dict[k]
            mutated_state_dict[k][mask] = op(state_dict[k][mask])
        
        return mutated_state_dict

    def evaluate(self, x, y, f):
        # Ensure input tensors are on the correct device
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        if y.device != device:
            y = y.to(device)

        y_ = self.forward(x)
        self.fitness = -f(y_, y).sum()
