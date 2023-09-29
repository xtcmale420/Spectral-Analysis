import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

class PoissonGp():
    
    def __init__(self, T, W, sigma):
        
        self.T = T
        self.W = W
        self.sigma = sigma
            
    def CovarianceFunction(self, t1, t2, k):
        
        result = torch.exp(-1*((t1-t2)**2)/(2*(self.sigma[k]**2)))
        return result
    
    def CovarianceMatrix(self, k):
        
        Matrix = torch.ones((self.T, self.T))
        for i in range(self.T):
            for j in range(self.T):
                Matrix[i][j] = self.CovarianceFunction(i, j, k)
                       
        return Matrix
    
    def Data(self):
        
        X = torch.stack([MultivariateNormal(torch.zeros(100), self.CovarianceMatrix(i)).sample() for i in range(self.p)])
        C = torch.matmul(self.W, X)
        C = torch.exp(C)
        Y = torch.poisson(C)
        
        return Y 
        