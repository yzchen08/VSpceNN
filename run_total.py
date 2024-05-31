import os
import time
from functools import wraps

import numpy as np
import torch
import torch_geometric as pyg
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model_total import HDNNP
# from model_total_test import HDNNP


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


@time_count
def get_pyrazine_data(start, end, data_path):
    import random
    data = torch.load(data_path)[start:end]
    random.seed(2024)
    random.shuffle(data)
    return data


@time_count
def train_one_epoch():
    model.train(mode=True)
    train_loss, train_loss_energy, train_loss_force = 0.0, 0.0, 0.0
    train_loss_dipole, train_loss_polar = 0.0, 0.0
    train_metric_energy, train_metric_force, train_metric_dipole, train_metric_polar = 0.0, 0.0, 0.0, 0.0
    for data in train_loader:
        data = data.cuda()
        energy, force, dipole, polar = model(data)

        loss_energy = loss_fn(energy, data.energy).mean()
        loss_force = loss_fn(force, -data.grads).mean()
        loss_dipole = loss_fn(dipole, data.dipole).mean()
        loss_polar = loss_fn(polar, data.polar).mean()
        loss = auto_weighted_loss(loss_energy, loss_force, loss_dipole, loss_polar)
        # loss = loss_energy + loss_force + loss_dipole + loss_polar

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_energy += loss_energy.item()
        train_loss_force += loss_force.item()
        train_loss_dipole += loss_dipole.item()
        train_loss_polar += loss_polar.item()

        train_metric_energy += metric_fn(energy, data.energy).mean().item()
        train_metric_force += metric_fn(force, -data.grads).mean().item()
        train_metric_dipole += metric_fn(dipole, data.dipole).mean().item()
        train_metric_polar += metric_fn(polar, data.polar).mean().item()

    train_loss /= len(train_loader)
    train_loss_energy /= len(train_loader)
    train_loss_force /= len(train_loader)
    train_loss_dipole /= len(train_loader)
    train_loss_polar /= len(train_loader)

    train_metric_energy /= len(train_loader)
    train_metric_force /= len(train_loader)
    train_metric_dipole /= len(train_loader)
    train_metric_polar /= len(train_loader)

    return (train_loss, train_loss_energy, train_loss_force, train_loss_dipole, train_loss_polar,
            train_metric_energy, train_metric_force, train_metric_dipole, train_metric_polar)


@time_count
def val_one_epoch():
    model.eval()
    val_metric_energy, val_metric_force, val_metric_dipole, val_metric_polar = 0.0, 0.0, 0.0, 0.0
    for data in val_loader:
        data = data.cuda()
        energy, force, dipole, polar = model(data)

        val_metric_energy += metric_fn(energy, data.energy).mean().item()
        val_metric_force += metric_fn(force, -data.grads).mean().item()
        val_metric_dipole += metric_fn(dipole, data.dipole).mean().item()
        val_metric_polar += metric_fn(polar, data.polar).mean().item()

    val_metric_energy /= len(val_loader)
    val_metric_force /= len(val_loader)
    val_metric_dipole /= len(val_loader)
    val_metric_polar /= len(val_loader)

    return val_metric_energy, val_metric_force, val_metric_dipole, val_metric_polar


@time_count
def test_one_epoch(best_model_filename):
    model = torch.load(best_model_filename, map_location=torch.device(f"cuda:{torch.cuda.current_device()}"))
    # print(model.get_density.coeffs[:9, :9])
    # np.savetxt("coeffs_44.dat", model.get_density.coeffs.detach().cpu(), fmt="%.5f")
    # exit()
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for k, v in model.named_parameters():
    #     print(k, v.shape)

    model.eval()
    test_metric_energy = 0.0
    pred_force_total = []
    true_force_total = []
    pred_dipole_total = []
    true_dipole_total = []
    pred_polar_total = []
    true_polar_total = []
    for count, data in enumerate(test_loader):
        print(f"now: {count}") if count % 10 == 0 else 1
        data = data.cuda()
        energy, force, dipole, polar = model(data)

        pred_force_total.append(force.data)
        true_force_total.append(-data.grads.data)
        pred_dipole_total.append(dipole.data)
        true_dipole_total.append(data.dipole.data)
        pred_polar_total.append(polar.data)
        true_polar_total.append(data.polar.data)

        test_metric_energy += metric_fn(energy, data.energy).mean().item()

    test_metric_energy /= len(test_loader)

    pred_force_total = torch.concat(pred_force_total)
    true_force_total = torch.concat(true_force_total)
    pred_dipole_total = torch.concat(pred_dipole_total)
    true_dipole_total = torch.concat(true_dipole_total)
    pred_polar_total = torch.concat(pred_polar_total)
    true_polar_total = torch.concat(true_polar_total)

    test_metric_force = metric_fn(pred_force_total, true_force_total).mean(dim=0).data
    test_metric_dip = metric_fn(pred_dipole_total, true_dipole_total).mean(dim=0).data
    test_metric_pol = metric_fn(pred_polar_total, true_polar_total).mean(dim=0).data

    relative_force = ((pred_force_total / true_force_total) - 1).abs().mean(dim=0) * 100
    relative_dip = ((pred_dipole_total / true_dipole_total) - 1).abs().mean(dim=0) * 100
    relative_pol = ((pred_polar_total / true_polar_total) - 1).abs().mean(dim=0) * 100

    rrmse_pol = loss_fn(pred_polar_total, true_polar_total).mean().data.sqrt() / true_polar_total.std()

    print(f"test MAE of energy: {test_metric_energy}")

    print(f"force MAE of each component:\n{test_metric_force}")
    print(f"force average MAE: {test_metric_force.mean()}")
    print(f"dipole MAE of each component:\n{test_metric_dip}")
    print(f"dipole average MAE: {test_metric_dip.mean()}")
    print(f"polar MAE of each component:\n{test_metric_pol}")
    print(f"polar average MAE: {test_metric_pol.mean()}")
    print(f"polar RRMSE: {rrmse_pol}")

    print(f"force relative:\n{relative_force}")
    print(f"dipole relative:\n{relative_dip}")
    print(f"polar relative:\n{relative_pol}")

    return None


def calc_energy_shift(data: list[Data]):
    print(f"len of energy shift data: {len(data)}")
    loader = DataLoader(data, batch_size=len(data),
                        shuffle=False, num_workers=0, pin_memory=False)

    element_count = []
    for data in loader:
        for idx in range(len(data)):
            numbers = data.atomic_numbers[data.batch == idx]
            numbers = numbers.bincount()[1:]
            numbers = torch.concat([numbers, torch.zeros(119 - len(numbers), dtype=torch.int32)])
            element_count.append(numbers.unsqueeze(0))
    element_count = torch.concat(element_count, dim=0)
    element_count = torch.hstack([element_count, torch.ones(len(data), 1, dtype=torch.int32)])

    shifts = np.linalg.lstsq(element_count, data.energy.flatten(), rcond=None)[0]

    return torch.Tensor(shifts)


def add_energy_shifts(data: list[Data], shifts: torch.Tensor):
    for idx in range(len(data)):
        numbers = data[idx].atomic_numbers.bincount()[1:]
        numbers = torch.concat([numbers, torch.zeros(119 - len(numbers), dtype=torch.int32), torch.ones(1)])
        data[idx].energy_shift = (numbers * shifts).sum().reshape(1, 1)
        # print((numbers * shifts).sum().reshape(1, 1))
        # print(shifts)
    return data


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    # torch.autograd.set_detect_anomaly(True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    TRAIN = True
    CUTOFF = 6.0  # 4.0 A for methanol, 6.0 A for pyrazine
    L = 2
    EPOCH = 200_000  # 5000
    TRAIN_VAL_SAMPLE = 2000  # 5000
    TRAIN_RATIO = 0.8
    BATCH_SIZE = 64  # 64
    LOG_FILE = "./log/EANN_ef_1.csv"
    MODEL_CKPT = "./log/EANN_ef_1.pt"
    # MODEL_CKPT = "./best/EANN_ef.pt"

    # train, val and test data sampled randomly from total dataset
    samples = get_pyrazine_data(0, 2500, f"./data/train/data.pt")

    samples_train = samples[:int(TRAIN_VAL_SAMPLE * TRAIN_RATIO)]
    energy_shifts = calc_energy_shift(samples_train)
    samples = add_energy_shifts(samples, energy_shifts)
    print(samples[0].energy_shift)
    # exit()

    # for idx in range(len(samples)):
    #     samples[idx].energy_shift = torch.Tensor([-165352.7812]).reshape(1, 1)  # 1000K: -165374.3438; 2000K: -165352.7812

    samples_train = samples[:int(TRAIN_VAL_SAMPLE * TRAIN_RATIO)]
    samples_val = samples[int(TRAIN_VAL_SAMPLE * TRAIN_RATIO):TRAIN_VAL_SAMPLE]
    samples_test = samples[TRAIN_VAL_SAMPLE:]

    print(f"length of train, val and test: {len(samples_train)}, {len(samples_val)}, {len(samples_test)}")

    train_loader = DataLoader(samples_train, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(samples_val,
                            batch_size=BATCH_SIZE * 16,
                            shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(samples_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=False)

    loss_fn = torch.nn.MSELoss(reduction="none")
    metric_fn = torch.nn.L1Loss(reduction="none")

    from loss import AutomaticWeightedLoss

    auto_weighted_loss = AutomaticWeightedLoss(4)
    # auto_weighted_loss = AutomaticWeightedLoss(4, init_param=[10, 10, 1500, 250])

    if TRAIN:
        model = HDNNP(L=L, r_cutoff=CUTOFF).cuda()
        # model = torch.load("./log/EANN_ef_45.pt").cuda()
        print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        WARNUP = 49  # 49
        INIT_LR = 1e-4  # 1e-4
        MAX_LR = 2e-3  # 5e-3
        MIN_LR = INIT_LR / 2
        DECAY_LR = -0.005  # -0.005

        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "weight_decay": 0, "lr": INIT_LR},
            {"params": auto_weighted_loss.parameters(), "weight_decay": 0, "lr": INIT_LR},
        ])
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.0)

        # scheduler = ReduceLROnPlateau(optimizer=optimizer,
        #                               factor=0.5,
        #                               patience=100,
        #                               min_lr=1e-05)

        def lr_lambda_1(epoch_lr):
            return (
                (INIT_LR + (MAX_LR - INIT_LR) / WARNUP * epoch_lr) / INIT_LR
                if epoch_lr <= WARNUP
                else max(MAX_LR * np.exp(DECAY_LR) ** (epoch_lr - WARNUP) / INIT_LR, MIN_LR / INIT_LR)
            )

        lr_lambda_2 = lambda epoch_lr: max(np.exp(DECAY_LR) ** epoch_lr, 1e-2)  # exponential decay
        lr_lambda_3 = lambda epoch_lr: 10  # constant
        scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda_1, lr_lambda_1])

        with open(LOG_FILE, "w") as f:
            f.write("epoch,train_loss,train_loss_energy,train_loss_force,train_loss_dipole,train_loss_polar,"
                    "train_metric_energy,train_metric_force,train_metric_dipole,train_metric_polar,"
                    "val_metric_energy,val_metric_force,val_metric_dipole,val_metric_polar")
            f.write("\n")

        best_metric = float("inf")
        for epoch in range(EPOCH):
            print(f"current learning rate: {scheduler.get_last_lr()}")
            print(f"current loss weight: {auto_weighted_loss.get_params().detach().numpy()}")

            (train_loss, train_loss_energy, train_loss_force, train_loss_dipole, train_loss_polar,
             train_metric_energy, train_metric_force, train_metric_dipole, train_metric_polar) = train_one_epoch()

            print(f"epoch: {epoch}, loss: {train_loss}, energy: {train_loss_energy}, force: {train_loss_force}, "
                  f"dipole: {train_loss_dipole}, polar: {train_loss_polar}, \n"
                  f"train metric energy: {train_metric_energy}, force: {train_metric_force}, "
                  f"dipole: {train_metric_dipole}, polar: {train_metric_polar}.")

            if epoch % 1 == 0:
                val_metric_energy, val_metric_force, val_metric_dipole, val_metric_polar = val_one_epoch()

                print(f"epoch: {epoch}, val metric energy: {val_metric_energy}, force: {val_metric_force}, "
                      f"dipole: {val_metric_dipole}, polar: {val_metric_polar}.")

                with open(LOG_FILE, "a") as f:
                    f.write(f"{epoch},{train_loss},{train_loss_energy},{train_loss_force},"
                            f"{train_loss_dipole},{train_loss_polar},"
                            f"{train_metric_energy},{train_metric_force},{train_metric_dipole},{train_metric_polar},"
                            f"{val_metric_energy},{val_metric_force},{val_metric_dipole},{val_metric_polar}")
                    f.write("\n")

                val_metric = val_metric_force + val_metric_energy
                if val_metric < best_metric:
                    torch.save(model, MODEL_CKPT)
                    print(f"save best model: {best_metric} -> {val_metric} to {MODEL_CKPT}")
                    best_metric = val_metric
                else:
                    print(f"current best model: {best_metric}")

            scheduler.step()

    test_one_epoch(MODEL_CKPT)

    print("DONE")
