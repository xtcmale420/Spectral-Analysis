import numpy as np
import torch
from torch import nn
from base import PoissonGp

class PointProcess(nn.Module):
    
    def __init__(self, Y, W=None, sigma=None):
        
        super(PointProcess, self).__init__()
        self.Y = Y
        if (W==None):
            W = torch.rand(4,3)
        if (sigma==None):
            sigma = torch.rand(3)
        self.sigma = nn.Parameter(sigma)
        self.W = nn.Parameter(W)
        self.T = 100
     
    def Chebyshev(self):
        
        for i in range(4):
            mean = torch.mean(self.Y[i])
            mean = np.array(mean)
            x = np.linspace(mean-2, mean+2, 1000)
            y = np.exp(x)
            Chebyshev = np.polynomial.chebyshev.Chebyshev.fit(x, y, 2)
            a, b, c = Chebyshev.convert().coef
            a = np.multiply(a, np.ones(self.T))
            b = np.multiply(b, np.ones(self.T))
            a = torch.tensor(a)
            b = torch.tensor(b)
            if (i==0):
                A = a
                B = b
            else:
                A = torch.concatenate((A,a), dim=0)
                B = torch.concatenate((B,b), dim=0)
        
        return A, B
            
        
    def SigmaInverse(self):
       
       GP = PoissonGp(self.T, self.W, self.sigma)
       K = torch.block_diag(*[GP.CovarianceMatrix(i) for i in range(3)]) 
       I = torch.eye(self.T)
       W = torch.kron(self.W, I)
       A = self.Chebyshev()[0]
       SigmaInverse = torch.matmul(torch.transpose(W, 0, 1), torch.diag(A).float())
       SigmaInverse = torch.matmul(SigmaInverse, W)
       SigmaInverse = torch.multiply(2, SigmaInverse)
       SigmaInverse = torch.add(SigmaInverse, torch.inverse(K))
       return SigmaInverse
   
    def Mu(self):
       
       B = self.Chebyshev()[1]
       W = torch.kron(self.W, torch.eye(self.T))
       y = torch.cat([self.Y[i] for i in range(4)])
       Sigma = self.SigmaInverse()
       Sigma = torch.inverse(Sigma)
       Mu = torch.matmul(Sigma, torch.transpose(W, 0, 1))
       Mu = torch.matmul(Mu, torch.subtract(y, B).float())
       return Mu
    
    def LogMarginalLikelihood(self):
        
        GP = PoissonGp(self.T, self.W, self.sigma)
        K = torch.block_diag(*[GP.CovarianceMatrix(i) for i in range(3)])
        SigmaInverse = self.SigmaInverse()
        Mu = self.Mu()
        term1 = torch.multiply(torch.linalg.slogdet(torch.inverse(SigmaInverse))[1], torch.tensor(0.5))
        term2 = torch.multiply(torch.tensor(0.5), torch.matmul(torch.matmul(Mu, SigmaInverse), Mu))
        term3 = torch.multiply(torch.tensor(0.5), torch.linalg.slogdet(K)[1])
        
        return term1+term2-term3
