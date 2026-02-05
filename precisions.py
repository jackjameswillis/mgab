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
    
    def initializer(self, shape):
        return self.mutate(torch.empty(shape, dtype=torch.float32))

class i8:
    def __init__(self, std=1):
        self.precision = 'i8'
        self.dtype = torch.float32
        self.std = std

    def cast_to(self, x):
        return x.to(torch.int8)
    
    def cast_from(self, x):
        return x.to(torch.float32)
    
    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1 which is
    # then rounded to nearest integer before casting to int8
    def mutate(self, x):
        mutations = torch.randn_like(x, dtype=torch.float32, device=x.device) * self.std
        mutated = torch.round(x + mutations)
        # Round to nearest integer first
        # Clamp to int8 range [-128, 127] before casting to int8
        mutated_clamped = torch.clamp(mutated, -128, 127)
        return mutated_clamped.to(torch.int8)

    def initializer(self, shape):
        # Sample from a Gaussian with std that covers the full range of int8
        # int8 range is [-128, 127], so we want to sample across this range
        # A std of ~40 should cover most of the range (68% within 1 std, 99.7% within 3 std)
        # Using std=60 to ensure good coverage
        return torch.round(torch.randn(shape, dtype=torch.float32) * 60).clamp(-128, 127).to(torch.int8)

class i4:    

    def __init__(self, std=1):
        self.precision = 'i4'
        self.dtype = torch.float32
        self.std = std
    
    def cast_to(self, x):
        # Clamp values to the range of int4 [-8, 7] and cast to int8
        x_clamped = torch.clamp(x, -8, 7)
        return x_clamped.to(torch.int8)
    
    def cast_from(self, x):
        return x.to(torch.float32)
    
    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1 which is
    # then rounded to nearest integer before clamping and casting to int8
    def mutate(self, x):
        mutations = torch.randn_like(x, dtype=torch.float32, device=x.device) * self.std
        mutated = torch.round(x + mutations)
        mutated_clamped = torch.clamp(mutated, -8, 7)
        return mutated_clamped.to(torch.int8)

    def initializer(self, shape):
        # int4 range is [-8, 7], so we want to sample across this range
        # Using std=3 to cover the range (std of ~3 covers ~99.7% of values within [-9, 9])
        return torch.round(torch.randn(shape, dtype=torch.float32) * 3).clamp(-8, 7).to(torch.int8)

class i2:
    def __init__(self, std=1):
        self.precision = 'i2'
        self.dtype = torch.float32
        self.std = std

    def cast_to(self, x):
        # Clamp values to the range of int2 [-2, 1] and cast to int8
        x_clamped = torch.clamp(x, -2, 1)
        return x_clamped.to(torch.int8)

    def cast_from(self, x):
        return x.to(torch.float32)


    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1 which is
    # then rounded to nearest integer before clamping and casting to int8
    def mutate(self, x):
        mutations = torch.randn_like(x, dtype=torch.float32, device=x.device) * self.std
        mutated = torch.round(x + mutations)
        mutated_clamped = torch.clamp(mutated, -2, 1)
        return mutated_clamped.to(torch.int8)

    def initializer(self, shape):
        # int2 range is [-2, 1], so we want to sample across this range
        # Using std=1.5 to cover the range (std of ~1.5 covers ~99.7% within [-4.5, 4.5])
        return torch.round(torch.randn(shape, dtype=torch.float32) * 1.5).clamp(-2, 1).to(torch.int8)

class binary:
    def __init__(self, std=0):
        self.precision = 'binary'
        self.dtype = torch.bool
    
    def cast_to(self, x):
        return ((x + 1)/2).to(torch.int8)
    
    def cast_from(self, x):
        return (x.to(torch.float32) * 2 - 1)
    
    def mutate(self, x):
        return x.logical_not().to(torch.int8)

precisions = dict(zip(['f32', 'i8', 'i4', 'i2', 'binary'], [f32, i8, i4, i2, binary]))
