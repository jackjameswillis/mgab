'''
Precision management functions

We define a number of different precision classes which are used to 
manage the precision of the parameters in the network and the genomes of the
genetic algorithm.

For network parameters we care about casting to and from different precision,
for the genetic algorithm we care about mutation operators.
'''

import torch
import math

'''
Precision management functions

We define a number of different precision classes which are used to 
manage the precision of the parameters in the network and the genomes of the
genetic algorithm.

For network parameters we care about casting to and from different precision,
for the genetic algorithm we care about mutation operators.
'''

import torch
import math

class f32:
    def __init__(self):

        self.bits = 32
    
    # Assumes x is only parameters that you want mutated
    # Performs a gaussian mutation with default std 1
    def mutate(self, x, std, version=None):
        return x + torch.randn_like(x, dtype=torch.float32, device=x.device) * std

    def cast_from(self, x, r):
        return x

'''
Class: Q
Implements a general wrapper for quantization operations.

bits -> number of unique values that can be taken

func cast_to(self, x) -> maps int8 onto space returning space.dtype

'''
class Q:

    def __init__(self, bits):

        try:

            assert bits <= 8
        
        except:

            raise ValueError("bits must be <= 8")
        
        self.bits = bits

    def cast_from(self, x, r=0.0):

        try:

            assert x.dtype == torch.int8
        
        except:

            raise ValueError("x must be int8")
        
        try:

            assert type(r) == float
        
        except:

            raise ValueError("range must be float")
        
        if r:

            mini, maxi = -(2**self.bits)//2, (2**self.bits)//2 - 1

            return -r + (x.to(torch.float32) - mini) * 2 * r/(maxi - mini)

        else:

            return x.to(torch.float32)
    
    '''
    Selects a number of elements to mutate according to mutation rate m, and returns x but mutated at selected indexes
    '''
    def mutate(self, x, m, version='local-uniform'):

        try:

            assert x.dtype == torch.int8
        
        except:

            raise ValueError("x must be int8")
        
        if version == 'uniform':
        
            mn = int(x.numel() * m)

            mut_index = torch.randint(0, x.numel(), (mn,), device=x.device)

            x.view(-1)[mut_index] = torch.randint(-(2**self.bits)//2, (2**self.bits)//2, size=(mn,), dtype=torch.int8, device=x.device)

            return x
        
        if version == 'local-uniform':

            mn = int(x.numel() * m)

            mut_index = torch.randint(0, x.numel(), (mn,), device=x.device)

            mags = torch.randint(1, self.bits, size=(mn,), device=x.device, dtype=torch.int8)

            signs = torch.randint(0, 2, size=(mn,), device=x.device) * 2 - 1

            xm = x.view(-1)
            
            xm[mut_index] += mags*signs

            # overflow corrections
            '''
            c = xm[mut_index][(signs + 1)//2] < x[mut_index][(signs + 1)//2]
            
            xm[mut_index][(signs + 1)//2][c] -= 2*mags[(signs + 1)//2][c]

            c = xm[mut_index][1 - (signs + 1)//2] > x[mut_index][1 - (signs + 1)//2]
            
            xm[mut_index][1 - (signs + 1)//2][c] += 2*mags[1 - (signs + 1)//2][c]
            '''

            return x.clamp(-(2**self.bits)//2, (2**self.bits)//2 - 1)