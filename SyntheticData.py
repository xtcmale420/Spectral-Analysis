import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class GenerativeModel():
    
    def __init__(self):
        
        self.n = torch.tensor(4)
        self.p = torch.tensor(3)
        self.T = torch.tensor(100)
        sigma = [0.97035554, 0.9210223, 0.95637767]
        self.sigma = torch.tensor(sigma)
        W = [[0.22263762, 0.91836964, 0.91825529],
             [0.12453973, 0.40475435, 0.22164199],
             [0.38289004, 0.95335849, 0.17444501],
             [0.48565162, 0.49369852, 0.62337549]]
        self.W = torch.tensor(W)
        
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
        