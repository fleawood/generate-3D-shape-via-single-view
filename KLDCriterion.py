import torch
import torch.nn as nn


class KLDCriterion(nn.Module):
    def __init__(self, coeff):
        super(KLDCriterion, self).__init__()
        self.coeff = coeff

    def forward(self, mean, var):
        square_mean = torch.pow(mean, 2)
        square_var = torch.pow(var, 2)
        return 0.5 * torch.sum(square_mean + square_var - torch.log(square_var) - 1)
