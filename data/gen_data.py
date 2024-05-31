import pickle

import pandas as pd
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import sys
from ase import Atom, Atoms
from ase.visualize import view
from sklearn import linear_model
from torch_scatter import scatter


class read_data:
    def __init__(self, xyz_path, energy_path, grad_path, dipole_path, polar_path):
        self.xyz_path = xyz_path
        self.energy_path = energy_path
        self.grad_path = grad_path
        self.dipole_path = dipole_path
        # self.chg_path = chg_path
        self.polar_path = polar_path

    def read_and_transform(self, r_cutoff, natom, out_file):
        with open(self.xyz_path, "r") as f:
            xyz_data = np.array([ii.strip() for ii in f.readlines() if ii]).reshape(-1, natom+2)

        with open(self.energy_path, "r") as f:
            energy_data = np.array([ii for ii in f.readlines() if ii], dtype=float)

        with open(self.grad_path, "r") as f:
            grad_data = np.array([ii.strip().split() for ii in f.readlines() if ii]).reshape((-1, natom, 3))

        with open(self.dipole_path, "r") as f:
            dipole_data = np.array([ii.strip().split() for ii in f.readlines() if ii]).reshape((-1, 3))

        # with open(self.chg_path, "r") as f:
        #     chg_data = np.array([ii.strip().split() for ii in f.readlines() if ii]).reshape((-1, natom))

        with open(self.polar_path, "r") as f:
            polar_data = np.array([ii.strip().split() for ii in f.readlines() if ii]).reshape((-1, 9))

        data_list = []
        count = 0

        for xyz, energy, grad, dipole, polar in zip(xyz_data, energy_data, grad_data,
                                                    dipole_data, polar_data):
            count += 1
            print(f"now: {count}") if count % 500 == 0 else 1

            xyz = xyz[2:]
            xyz = np.array([ii.split() for ii in xyz])
            positions = torch.Tensor(np.array(xyz[:, 1:], dtype=float))  # Angstrom

            atomic_numbers = torch.Tensor([Atom(ii.upper()).number for ii in xyz[:, 0]])

            gradients = torch.Tensor(np.array(grad, dtype=float)) * 1185.8212143783846  # Hartree/Bohr to kcal/mol/A

            dipole = torch.Tensor(np.array(dipole, dtype=float)).unsqueeze(0) * 2.541748485746558  # atomic unit to D

            # chg = (atomic_numbers - torch.Tensor(np.array(chg, dtype=float))).unsqueeze(-1)  # unit in e

            polar = torch.Tensor(np.array(polar, dtype=float)).reshape(3, 3).unsqueeze(0) * 0.529177249 ** 3  # a.u. to A**3

            atoms = Atoms(atomic_numbers, positions)  # unit in Angstrom
            # view(atoms)
            idx_i, idx_j = neighbor_list("ij", atoms, r_cutoff, self_interaction=False)
            edge_index = torch.Tensor(np.array([idx_i, idx_j])).long()
            # edge_attr = distances[mask].reshape(-1, 1)
            data = Data(
                x=None, y=None, pos=positions,
                edge_index=edge_index,
                energy=torch.Tensor([energy]).reshape(1, -1) * 627.5096080305927,  # Hartree to kcal/mol
                # edge_attr=edge_attr,
                atomic_numbers=atomic_numbers.long(),
                grads=gradients,
                dipole=dipole,
                # chg=chg,
                polar=polar,
            )
            data_list.append(data)

        torch.save(data_list, out_file)

        data_list = torch.load(out_file)
        for data in data_list:
            print(data)


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    data_dir = sys.argv[1]

    data = read_data(f"./{data_dir}/xyz.dat",
                     f"./{data_dir}/energy.dat",
                     f"./{data_dir}/grad.dat",
                     f"./{data_dir}/dipole.dat",
                     # "./ase_1000K/population.dat",
                     f"./{data_dir}/polarizability.dat")
    data.read_and_transform(r_cutoff=6.0, natom=10, out_file=f"./{data_dir}/data.pt")  # r_cutoff: 1.7, 3.0, 4.5, 6.0


