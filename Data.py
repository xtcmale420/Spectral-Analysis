from SyntheticData import GenerativeModel
import torch
GP = GenerativeModel()
Y = GP.Data()


Z = torch.cat(([Y[i] for i in range(4)]), 0)
F = [Y[i] for i in range(4)]
print(Z.size)
print(F)