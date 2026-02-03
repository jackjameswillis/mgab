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
        mutations[mutations > 0] += 1
        mutations[mutations < 0] -= 1
        mutated = torch.round(x + mutations)
        # Round to nearest integer first
        # Clamp to int8 range [-128, 127] before casting to int8
        mutated_clamped = torch.clamp(mutated, -128, 127)
        return mutated_clamped.to(torch.int8)

    def initializer(self, shape):
        # Initialize with Xavier uniform and quantize to int8
        weight = torch.empty(shape, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(weight)
        # Quantize to signed 8-bit integers
        max_val = weight.abs().max()
        scale = max_val / (2**(8-1) - 1)  # Scale for int8 [-128, 127]
        weight_q = torch.round(weight / scale)
        weight_q = torch.clamp(weight_q, -128, 127)
        return weight_q.to(torch.int8)

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
        mutations[mutations > 0] += 1
        mutations[mutations < 0] -= 1
        mutated = torch.round(x + mutations)
        mutated_clamped = torch.clamp(mutated, -8, 7)
        return mutated_clamped.to(torch.int8)

    def initializer(self, shape):
        # Initialize with Xavier uniform and quantize to int4
        weight = torch.empty(shape, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(weight)
        # Quantize to signed 4-bit integers [-8, 7]
        max_val = weight.abs().max()
        scale = max_val / (2**(4-1) - 1)  # Scale for int4 [-8, 7]
        weight_q = torch.round(weight / scale)
        weight_q = torch.clamp(weight_q, -8, 7)
        return weight_q.to(torch.int8)

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
        mutations[mutations > 0] += 1
        mutations[mutations < 0] -= 1
        mutated = torch.round(x + mutations)
        mutated_clamped = torch.clamp(mutated, -2, 1)
        return mutated_clamped.to(torch.int8)

    def initializer(self, shape):
        # Initialize with Xavier uniform and quantize to int2
        weight = torch.empty(shape, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(weight)
        # Quantize to signed 2-bit integers [-2, 1]
        max_val = weight.abs().max()
        scale = max_val / (2**(2-1) - 1)  # Scale for int2 [-2, 1]
        weight_q = torch.round(weight / scale)
        weight_q = torch.clamp(weight_q, -2, 1)
        return weight_q.to(torch.int8)

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
