'''
Precision management functions

We define a number of different precision classes which are used to 
manage the precision of the parameters in the network and the genomes of the
genetic algorithm.

For network parameters we care about casting to and from different precision,
for the genetic algorithm we care about mutation operators.
'''

import torch

class f32:
    def __init__(self, std=1):
        self.precision = 'f32'
        self.dtype = torch.float32
        self.std = std
    
    def cast_to(self, x):
        return x.to(torch.float32)

    def cast_from(self, x):
        return x.to(torch.float32)
    
    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1
    def mutate(self, x):
        return x + torch.randn_like(x, dtype=torch.float32, device=x.device) * self.std
    
class i8:
    def __init__(self, std=1):
        self.precision = 'i8'
        self.dtype = torch.float32
        self.std = std

    def cast_to(self, x):
        return x.to(torch.float32)
    
    def cast_from(self, x):
        return x.to(torch.float32)
    
    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1 which is
    # then rounded to nearest integer before casting to int8
    def mutate(self, x):
        mutated = x + torch.randn_like(x, dtype=torch.float32, device=x.device) * self.std
        # Round to nearest integer first
        mutated_rounded = torch.round(mutated)
        # Clamp to int8 range [-128, 127] before casting to int8
        mutated_clamped = torch.clamp(mutated_rounded, -128, 127)
        return mutated_clamped.to(torch.float32)

class i4:
    def __init__(self, std=1):
        self.precision = 'i4'
        self.dtype = torch.float32
        self.std = std
    
    def cast_to(self, x):
        # Clamp values to the range of int4 [-8, 7] and cast to int8
        x_clamped = torch.clamp(x, -8, 7)
        return x_clamped.to(torch.float32)
    
    def cast_from(self, x):
        return x.to(torch.float32)
    
    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1 which is
    # then rounded to nearest integer before clamping and casting to int8
    def mutate(self, x):
        mutated = x + torch.randn_like(x, dtype=torch.float32, device=x.device) * self.std
        # Round to nearest integer first
        mutated_rounded = torch.round(mutated)
        # Clamp to int4 range [-8, 7] before casting to int8
        mutated_clamped = torch.clamp(mutated_rounded, -8, 7)
        return mutated_clamped.to(torch.float32)

class i2:
    def __init__(self, std=1):
        self.precision = 'i2'
        self.dtype = torch.float32
        self.std = std

    def cast_to(self, x):
        # Clamp values to the range of int2 [-2, 1] and cast to int8
        x_clamped = torch.clamp(x, -2, 1)
        return x_clamped.to(torch.float32)

    def cast_from(self, x):
        return x.to(torch.float32)

    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1 which is
    # then rounded to nearest integer before clamping and casting to int8
    def mutate(self, x):
        mutated = x + torch.randn_like(x, dtype=torch.float32, device=x.device) * self.std
        # Round to nearest integer first
        mutated_rounded = torch.round(mutated)
        # Clamp to int2 range [-2, 1] before casting to int8
        mutated_clamped = torch.clamp(mutated_rounded, -2, 1)
        return mutated_clamped.to(torch.float32)

class binary:
    def __init__(self, std=0):
        self.precision = 'binary'
        self.dtype = torch.bool
    
    def cast_to(self, x):
        return ((x + 1)/2).to(torch.float32)
    
    def cast_from(self, x):
        return (x.to(torch.float32) * 2 - 1)
    
    def mutate(self, x):
        return x.logical_not().to(torch.float32)

precisions = dict(zip(['f32', 'i8', 'i4', 'i2', 'binary'], [f32, i8, i4, i2, binary]))