import torch


class ShiftedSoftplus(torch.nn.Module):
    def forward(self, x):
        return torch.nn.Softplus()(x) - torch.log(torch.tensor(2.0))
