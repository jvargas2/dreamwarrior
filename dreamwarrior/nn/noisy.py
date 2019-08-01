import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Noisy(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(Noisy, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        """NoisyNets replace a standard linear layer. Instead of:

        y = wx + b

        they do:

        y = (μ+σ⊙ε)x + μ+σ⊙ε

        where μ+σ⊙ε replaces w and μ+σ⊙ε replaces b
        """
        if self.training:
            # μ + σ ⊙ ε
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        """Reset parameters according to NoisyNets paper.

        ```For factorised noisy networks, each element μi,j was initialised by a sample from an
        independent uniform distributions U [− √p , + √p ] and each element σi,j was
        initialised to a constant (σ0)/√p . The hyperparameter σ0 is set to 0.5.```
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        # f(x) = sgn(x)√|x|
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
