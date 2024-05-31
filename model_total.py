import math
import time
from abc import ABC
from collections import OrderedDict
from functools import wraps

from scipy.special import factorial
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_scatter import scatter
import numpy as np

from activation_fn import ShiftedSoftplus


def time_count(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(
            f"{func.__name__} time consume: {round(time.perf_counter() - start, 4)} s."
        )
        return res

    return wrapper


class EANNLayer(MessagePassing, ABC):
    def __init__(self, L, alpha, r_s, r_cutoff, max_number):
        super().__init__(aggr='add')
        self.L = L
        self.alpha = alpha
        self.r_s = r_s
        self.len_params = len(self.alpha) * len(self.r_s) * (self.L + 1)
        self.r_cutoff = r_cutoff
        # self.coeffs = torch.ones(10) / 10
        self.coeffs = torch.ones(max_number, max_number) / 10  # (119, 119) / 10
        # self.coeffs = torch.Tensor(np.loadtxt("./coeffs_45.dat"))
        # self.reset_parameters()
        self.coeffs = torch.nn.Parameter(self.coeffs)

        # self.element_coeffs = torch.nn.Parameter(torch.ones(119))

        self.L_list = self.generate_L()
        self.L_count = [len(l) for l in self.L_list]  # len of L_count: (L+1)
        self.L_total_count = sum(self.L_count)  # N_L
        self.L_list = torch.concat(self.L_list)  # (N_L, 3)
        self.L_preidx = (torch.Tensor(factorial(self.L_list.sum(-1))) /
                         torch.Tensor(factorial(self.L_list).prod(-1))).cuda()  # (N_L)
        self.L_list = self.L_list.cuda()

    def reset_parameters(self):
        # torch.nn.init.orthogonal_(self.coeffs)
        torch.nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))

    def generate_L(self):
        """
        generate series of (lx, ly, lz).
        :return: list of torch.Tensor: (L, N_Ls, 3)
        """
        L_total = []
        for single_l in range(self.L + 1):
            L_list = []
            for ii in range(single_l + 1):
                for jj in range(single_l + 1):
                    for kk in range(single_l + 1):
                        L_list.append([ii, jj, kk])
            L_list_tmp = []
            for l in L_list:
                if sum(l) == single_l:
                    L_list_tmp.append(l)
            L_total.append(torch.Tensor(L_list_tmp))
        return L_total

    def cos_cutoff(self, r_ij_norm):
        """
        cutoff function.
        :param r_ij_norm: torch.Tensor: (E, *).
        :return: torch.Tensor: (E, *).
        """
        # r_ij_norm[r_ij_norm > self.r_cutoff] = self.r_cutoff
        return (0.5 + 0.5 * torch.cos(torch.pi * r_ij_norm / self.r_cutoff)) ** 2

    def atomic_density(self, r_ij, r_ij_norm, L_list):
        """
        get atomic density.
        :param r_ij: torch.Tensor: (E, N_L, 3).
        :param r_ij_norm: torch.Tensor: (E, a, rs, N_L).
        :param L_list: (lx, ly, lz), torch.Tensor: (N_L, 3).
        :return: torch.Tensor: (E, a, rs, N_L).
        """
        left_res = (r_ij ** L_list).prod(-1)  # (E, N_L)

        # left_res = left_res.unsqueeze(1).unsqueeze(1).repeat(1, len(self.alpha), len(self.r_s), 1)  # (E, a, rs, N_L)
        # r_s = self.r_s.unsqueeze(-1).repeat(1, self.L_total_count).to(r_ij.device)  # (rs, N_L)
        # alpha = (self.alpha.unsqueeze(-1).unsqueeze(-1).
        #          repeat(1, len(self.r_s), self.L_total_count).to(r_ij.device))  # (a, rs, N_L)
        # return left_res * torch.exp(-alpha * (r_ij_norm - r_s) ** 2)

        res_1 = (r_ij_norm.unsqueeze(-1).expand(-1, len(self.r_s)) - self.r_s.unsqueeze(0).expand(-1, len(self.r_s))) ** 2  # (E, rs)
        exp_res = torch.exp(torch.einsum("j,ik->ijk", -self.alpha, res_1))  # (E, a, rs)
        return torch.einsum("il,ijk->ijkl", left_res, exp_res)  # (E, a, rs, N_L)

    def forward(self, x, edge_index, atomic_numbers):
        """
        calculate density.
        :param x: torch.Tensor: (num_atoms, 3).
        :param edge_index: torch.Tensor: (2, E).
        :param atomic_numbers:
        :return: torch.Tensor: (num_atoms, a * rs * L+1).
        """
        pair_idx = torch.concat([atomic_numbers[edge_index[0]].unsqueeze(0),
                                 atomic_numbers[edge_index[1]].unsqueeze(0)], dim=0)
        pair_idx = pair_idx.sort(dim=0).values  # (2, E)

        out = self.propagate(edge_index, x=x, L_list=self.L_list.cuda(), coeff_index=pair_idx)  # (num_atoms, a * rs * N_L)
        out = out.reshape(len(x), len(self.alpha), len(self.r_s), self.L_total_count)  # (num_atoms, a, rs, N_L)
        out = self.L_preidx.cuda() * out ** 2  # (num_atoms, a, rs, N_L)

        # all features of L are squeezed into last dimension, now separate them.
        out_total = scatter(out, self.L_list.sum(-1).long().cuda(), dim=-1)  # (num_atoms, a, rs, L+1)
        out_total = out_total.flatten(1, -1)  # (num_atoms, a * rs * L+1)

        # out_total = out_total * self.element_coeffs[atomic_numbers].unsqueeze(-1)  # (num_atoms, 1)

        return out_total

    def message(self, x_i, x_j, L_list, coeff_index):
        """
        calculate psai.
        :param x_i: torch.Tensor: (E, 3).
        :param x_j: torch.Tensor: (E, 3).
        :param L_list: torch.Tensor: (N_L, 3).
        :param coeff_index: torch.Tensor: (2, E).
        :return: torch.Tensor: (E, a * rs * N_L).
        """
        r_ij = x_i - x_j + 1e-6  # (E, 3)
        # r_ij = x_j - x_i  # (E, 3)

        # r_ij_norm = torch.norm(r_ij, dim=-1, keepdim=True)  # (E, 1)
        # r_ij = r_ij.unsqueeze(1).repeat(1, self.L_total_count, 1)  # (E, N_L, 3)
        # r_ij_norm = r_ij_norm.unsqueeze(-1).unsqueeze(-1).repeat(1, len(self.alpha), len(self.r_s), self.L_total_count)  # (E, a, rs, N_L)
        # density = self.atomic_density(r_ij=r_ij, r_ij_norm=r_ij_norm, L_list=L_list) * self.cos_cutoff(r_ij_norm)

        r_ij_norm = torch.norm(r_ij, dim=-1)  # (E)
        r_ij = r_ij.unsqueeze(1).expand(-1, self.L_total_count, 3)  # (E, N_L, 3)
        density = torch.einsum("ijkl,i->ijkl", self.atomic_density(r_ij=r_ij, r_ij_norm=r_ij_norm, L_list=L_list), self.cos_cutoff(r_ij_norm))

        density = density.flatten(1, 3)  # (E, a * rs * N_L)
        coeff_edge = self.coeffs[coeff_index.tolist()].unsqueeze(-1)  # (E, 1)
        density = density * coeff_edge
        return density


class PhysLayer(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim, r_cutoff, num_gaussian, embedding_size, num_res_block):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.r_cutoff = r_cutoff
        self.num_gaussian = num_gaussian
        self.embedding_size = embedding_size

        self.distance_expansion = GaussianSmearing(0.0, self.r_cutoff.item(), num_gaussians=self.num_gaussian)

        self.lin_nbr = torch.nn.Linear(self.in_dim + self.embedding_size, self.out_dim)
        self.lin_atom = torch.nn.Linear(self.in_dim + self.embedding_size, self.out_dim)
        self.lin_rbf = torch.nn.Linear(self.num_gaussian, self.out_dim)

        self.num_res_block = num_res_block
        self.lin_res_1 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.lin_res_2 = torch.nn.Linear(self.out_dim, self.out_dim)

        self.lin_out = torch.nn.Linear(self.out_dim, self.out_dim)

        # self.activation = torch.nn.Softplus()
        self.activation = ShiftedSoftplus()

    def residual_block(self, density):
        out = self.activation(density)
        out = self.lin_res_1(out)
        out = self.activation(out)
        out = self.lin_res_2(out)
        return out + density

    def forward(self, xx_pos_nuclei, edge_index):
        out_ij = self.propagate(edge_index, x=xx_pos_nuclei)
        out_ii = torch.concat((xx_pos_nuclei[:, :self.in_dim], xx_pos_nuclei[:, -self.embedding_size:]), dim=-1)
        out_ii = self.activation(self.lin_atom(self.activation(out_ii)))
        out = out_ij + out_ii

        for i in range(self.num_res_block):
            out = self.residual_block(out)

        out = self.activation(out)
        out = self.lin_out(out)

        return out

    def message(self, x_i, x_j):
        density_j = x_j[:, :self.in_dim]
        pos_i, pos_j = x_i[:, self.in_dim:(self.in_dim + 3)], x_j[:, self.in_dim:(self.in_dim + 3)]
        nuclei_j = x_j[:, -self.embedding_size:]

        r_ij = pos_i - pos_j + 1e-6
        # r_ij = pos_j - pos_i

        r_ij_norm = torch.norm(r_ij, dim=-1, keepdim=True)
        rbf_ij = self.distance_expansion(r_ij_norm)

        density_j = torch.concat((density_j, nuclei_j), dim=-1)
        density_j = self.activation(self.lin_nbr(self.activation(density_j)))
        rbf_ij = self.lin_rbf(rbf_ij)
        out = density_j * rbf_ij

        return out


class LinearNN(torch.nn.Module):
    def __init__(self, nn_shape):
        """
        construction of linear module

        Parameters
        ----------
        nn_shape : iterable[int]
            first element is input shape, the following are the output shapes of all layers,
            so the number of linear layers is len(nn_shape).
        """
        super().__init__()
        self.nn_shape = nn_shape
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(self.nn_shape[ii], self.nn_shape[ii + 1]) for ii in range(len(self.nn_shape) - 1)]
        )
        # self.activation = torch.nn.CELU(1.0)
        self.activation = ShiftedSoftplus()

    def forward(self, x):
        for idx, linear in enumerate(self.linears):
            x = self.activation(x)
            x = linear(x)
        return x


class HDNNP(torch.nn.Module):
    def __init__(self, L, r_cutoff):
        super().__init__()
        self.L = L
        self.r_cutoff = torch.Tensor([r_cutoff]).cuda()
        self.r_s = torch.linspace(0, 6, 64).cuda()  # torch.linspace(0, 6, 64)
        self.alpha = torch.Tensor([16.0]).cuda()  # 1.0
        self.len_params = len(self.alpha) * len(self.r_s) * (self.L + 1)

        self.embedding_size = 64  # 64
        self.num_gaussian = 32  # 32
        self.max_element_number = 7

        self.get_density = EANNLayer(L=self.L, alpha=self.alpha, r_s=self.r_s, r_cutoff=self.r_cutoff, max_number=self.max_element_number + 1)

        self.embedding = torch.nn.Embedding(self.max_element_number + 1, self.embedding_size)

        self.num_convs = 5  # 5
        self.convs_shape = [self.len_params, 256, 256, 128, 128, 128]
        self.num_res_block = 3  # 3
        self.convs = torch.nn.ModuleList([
            PhysLayer(in_dim=self.convs_shape[idx], out_dim=self.convs_shape[idx + 1], r_cutoff=self.r_cutoff,
                      num_gaussian=self.num_gaussian, embedding_size=self.embedding_size,
                      num_res_block=self.num_res_block)
            for idx in range(self.num_convs)
        ])

        self.ANN_type = [1, 6, 7]
        self.ANN_dict = torch.nn.ModuleDict()
        self.ANN_shape = [self.convs_shape[-1], 128, 128, 64]
        # for ii in range(1, self.max_element_number + 1):
        for ii in self.ANN_type:
            self.ANN_dict[str(ii)] = LinearNN(nn_shape=self.ANN_shape)

        self.output_shape = [self.ANN_shape[-1], 64, 64, 1]
        self.nn_ef = LinearNN(nn_shape=self.output_shape)
        self.nn_dip = LinearNN(nn_shape=self.output_shape)
        self.nn_pol = LinearNN(nn_shape=self.output_shape)
        self.nn_pol_3 = LinearNN(nn_shape=self.output_shape)

        from utils import MassCentre
        self.mass_centre = MassCentre()

        # self.ln_list = torch.nn.ModuleList([torch.nn.LayerNorm(ii, eps=1e-8) for ii in self.convs_shape])

    # @time_count
    def forward(self, data):
        data.pos.requires_grad_(True)
        xx = self.get_density(data.pos, data.edge_index, data.atomic_numbers)  # (batch_size * num_atoms, a * rs * L+1)
        nuclei = self.embedding(data.atomic_numbers)  # (batch_size * num_atoms, embed_size)

        # data.pos.requires_grad_(True)
        for idx, conv in enumerate(self.convs):
            # xx = self.ln_list[idx](xx)
            xx = torch.concat((xx, data.pos, nuclei), dim=-1)
            xx = conv(xx, data.edge_index)  # (batch_size * num_atoms, a * rs * L+1)
        # xx = self.ln_list[-1](xx)

        # for GPU and CPU
        yy = torch.zeros((len(xx)), self.ANN_shape[-1], requires_grad=False).cuda()  # (batch_size * N_atoms, 1)
        for number in set(data.atomic_numbers.tolist()):
            yy[data.atomic_numbers == number] = self.ANN_dict[str(int(number))](
                xx[data.atomic_numbers == number]
            )
        yy.requires_grad_(True)  # (batch_size * N_atoms, 1)
        # yy = xx

        # calc energy
        yy_energy = self.nn_ef(yy)  # (batch_size * N_atoms, 1)
        yy_sum = scatter(yy_energy, data.batch, dim=0)  # (batch_size, 1)
        yy_sum = yy_sum + data.energy_shift

        forces = -torch.autograd.grad(yy_sum,
                                      data.pos,
                                      grad_outputs=torch.ones(yy_sum.shape).cuda(),
                                      create_graph=True,
                                      retain_graph=True, )[0]

        # calc dipole
        partial_chg = self.nn_dip(yy)  # (batch_size * N_atoms, 1)

        # make total charge be 0
        # total_chg = scatter(partial_chg, data.batch, dim=0)[data.batch]  # (batch_size * N_atoms, 1)
        # num_atoms = scatter(torch.ones((len(data.batch), 1), device=data.batch.device), data.batch, dim=0)[data.batch]  # (batch_size * N_atoms, 1)
        # partial_chg = partial_chg - total_chg / num_atoms  # (batch_size * N_atoms, 1)

        mass_coord = self.mass_centre.centroid_coordinate(data.atomic_numbers, data.pos, data.batch)  # (batch_size * N_atoms, 3)
        dipole = partial_chg * mass_coord  # (batch_size * N_atoms, 3)
        dipole = scatter(dipole, data.batch, dim=0) * 4.803204544369458  # e*A to D, (batch_size, 3)

        # calc polar
        pol_coord = data.pos
        yy_pol = self.nn_pol(yy)  # (batch_size * N_atoms, 1)
        yy_sum_pol = scatter(yy_pol, data.batch, dim=0)  # (batch_size, 1)
        D_matrix = torch.autograd.grad(yy_sum_pol,
                                       data.pos,
                                       grad_outputs=torch.ones(yy_sum_pol.shape).cuda(),
                                       create_graph=True,
                                       retain_graph=True, )[0]

        n_atoms = int(len(data.pos) / len(data.energy))
        D_matrix = D_matrix.reshape(-1, n_atoms, 3)  # (batch_size, N_atoms, 3)
        D_matrix_T = D_matrix.transpose(1, 2).contiguous()
        polar_1 = torch.einsum('ijk,ikl->ijl', D_matrix_T, D_matrix)  # (batch_size, 3, 3)

        pol_coord = pol_coord.reshape(-1, n_atoms, 3)  # (batch_size, N_atoms, 3)
        pol_coord_T = pol_coord.transpose(1, 2).contiguous()

        polar_2 = (torch.einsum('ijk,ikl->ijl', pol_coord_T, D_matrix) +
                   torch.einsum('ijk,ikl->ijl', D_matrix_T, pol_coord))  # (batch_size, 3, 3)

        yy_pol_3 = self.nn_pol_3(yy) * 10  # (batch_size * N_atoms, 1)
        yy_sum_pol_3 = scatter(yy_pol_3, data.batch, dim=0)  # (batch_size, 1)
        polar_3 = (yy_sum_pol_3.unsqueeze(-1).repeat(1, 3, 3) *
                   torch.eye(3).cuda().unsqueeze(0).repeat(len(yy_sum_pol_3), 1, 1))

        polar = polar_1 + polar_2 + polar_3  # A**3, (batch_size, 3, 3)

        return yy_sum, forces, dipole, polar
