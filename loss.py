import torch
import torch.nn.functional as F


class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, num, init_param=None):
        super().__init__()
        if init_param:
            self.params = torch.sqrt(0.5 / torch.Tensor(init_param))
        else:
            self.params = torch.nn.Parameter(torch.ones(num))

    def get_params(self):
        return 0.5 / (self.params ** 2)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i])
        return loss_sum


class ForceCosineLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def calc_angle(self, force_1, force_2):
        return torch.rad2deg(torch.acos(torch.cosine_similarity(force_1, force_2, dim=1)))

    def forward(self, force_1, force_2):
        return ((torch.cosine_similarity(force_1, force_2, dim=1) - 1) ** 2) * 1e3
        # return torch.rad2deg(torch.acos(torch.cosine_similarity(force_1, force_2, dim=1))) ** 2
