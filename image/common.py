import glob
import json
import os
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn.functional as F
from pennylane.transforms import merge_amplitude_embedding
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import FilterOpNodes
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import (
    binary_f1_score,
    multiclass_accuracy,
    multiclass_f1_score,
)
from PIL import Image

log_interval = 1
tau = 1.0
grad_dist = []

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


def build_qlayer_config_list(configs):
    ret = []
    for config in configs:
        for l in config["layers"]:
            for i, c in enumerate(config["circuits"]):
                ret.append(
                    {
                        "circuits": c,
                        "qubits_per_circ": config["qubits_per_circ"][i],
                        "qubits_per_circ_ang": config["qubits_per_circ_ang"][i],
                        "measure_wires": None
                        if i >= len(config["measure_wires"])
                        else config["measure_wires"][i],
                        "embedding": config["embedding"],
                        "layers": l,
                        "layers_ang": config["layers_ang"][i],
                    }
                )
    return ret


def get_loader_circuit_tape(path):
    with open(path) as f:
        circ = qml.from_qasm(f.read())
    with qml.tape.QuantumTape() as tape:
        circ()
    return tape


def loader_circuit(params, tape):
    batched = qml.math.ndim(params) > 1
    features = qml.math.T(params) if batched else loader_circuit
    
    param_count = 0
    for op in tape:
        wires = op.wires
        gate = getattr(qml, op.name)
        if len(op.parameters) == 0:
            gate(wires=wires)
        else:
            gate(features[param_count], wires=wires)
            param_count += 1


class CustomImageDataset(Dataset):
    def __init__(self, path, class_names, filter_names = [], transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

        for class_label, class_name in enumerate(class_names):
            class_path = f'{path}/{class_name}'
            for filename in os.listdir(class_path):
                name, extension = filename.split(".")
                if (extension == "png" or extension == "jpg") and name in filter_names:
                    img_path = f"{class_path}/{filename}"
                    self.data.append([img_path, class_label, filename])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, img_name = self.data[idx]
        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, img_name


class CustomLoaderDataset(Dataset):
    def __init__(self, path, class_names, dim1_blocks = 2, dim2_blocks = 3, target_transform = None):
        self.target_transform = target_transform
        self.data = []

        for class_label, class_name in enumerate(class_names):
            class_path = f'{path}/{class_name}'
            for filename in sorted(os.listdir(class_path)):
                extension = filename.split(".")[-1]
                if "qasm" in extension:
                    base_name = "_".join(filename.split("_")[:-2])
                    if len(self.data) == 0 or (len(self.data) > 0 and self.data[-1][-1] != base_name):
                        block_paths = []
                        for dim1 in range(dim1_blocks):
                            for dim2 in range(dim2_blocks):
                                block_paths.append(f'{class_path}/{base_name}_{dim1}_{dim2}.{extension}')
                        self.data.append([block_paths, class_label, base_name])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        circ_paths, class_name, img_name = self.data[idx]
        label = class_name
        all_params = []
        
        for circ_path in circ_paths:
            qc = QuantumCircuit.from_qasm_file(circ_path)
            circ_params = []
            for inst in qc:
                circ_params.extend(inst.operation.params)

            all_params.append(circ_params)
        
        if self.target_transform:
            label = self.target_transform(label)

        return torch.tensor(np.array(all_params), requires_grad = False, dtype = torch.float), label, img_name


    def set_item(self, idx, datapoint):
        self.data[idx] = datapoint

def get_data_loaders(dataset, bs_train, bs_test, train_ratio):
    torch.manual_seed(42)
    d_size = len(dataset)
    print(f"loaded dataset size: {d_size}")

    indices = np.arange(d_size)
    np.random.shuffle(indices)
    train_size = int(train_ratio * d_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = None
    test_loader = None
    
    if bs_train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=bs_train,
            shuffle=True
        )

    if bs_test:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=bs_test,
            shuffle=True
        )
    return train_loader, test_loader


class TauCrossEntropyLoss(nn.Module):
    def __init__(self, tau=1.0, reduction='mean'):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.reduction = reduction

    def forward(self, logits, targets):
        scaled_logits = self.tau * logits
        loss = F.cross_entropy(scaled_logits, targets, reduction=self.reduction)
        return loss / self.tau

# criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 1.0]))
criterion = TauCrossEntropyLoss(tau = tau)

def train_step(net, opt, sched, epoch, loader, private = False):
    net.train()
    train_loss = 0

    per_sample_grad_list = []

    for batch_idx, (data, target, _) in enumerate(loader):
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        if batch_idx % log_interval == log_interval - 1:
            print(
                f"Train Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(loader.dataset)} ({100.0 * (batch_idx + 1) / len(loader):.0f}%)]\tLoss: {loss.item():.6f}",
                flush=True,
            )
        opt.step()

        if private:
            batch_per_sample_grad_list = []
            for key, param in dict(net.named_parameters()).items():
                batch_per_sample_grad_list.append(torch.reshape(param.grad_sample, (data.shape[0], -1)))
            per_sample_grad_list.append(torch.cat(batch_per_sample_grad_list, dim = 1))

        opt.zero_grad()
        if sched is not None:
            sched.step()

    if private:
        per_sample_grad_list = torch.cat(per_sample_grad_list, dim = 0)
        grad_dist.append(per_sample_grad_list.detach())
        
    train_loss /= len(loader)
    print(f"Train loss: {train_loss:.6f}", flush=True)
    return train_loss


def test(net, loader):
    test_loss = 0
    preds = torch.tensor([])
    targets = torch.tensor([])
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target, names) in enumerate(loader):
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            preds = torch.cat([preds, pred])
            targets = torch.cat([targets, target])

    test_loss /= len(loader)
    preds = torch.flatten(preds).long()
    targets = torch.flatten(targets).long()

    test_acc = multiclass_f1_score(preds, targets)

    print(
        "\nTest set: Avg. loss: {:.6f}, F1 Score: {:.5f}\n".format(
            test_loss,
            test_acc,
        ),
        flush = True
    )

    return test_loss, test_acc.item(), preds


@timing
def epoch(net, opt, sched, epoch, train_loader, test_loader, private):
    train_loss = train_step(net, opt, sched, epoch, train_loader, private)
    test_loss, test_acc, preds = test(net, test_loader)
    return train_loss, test_loss, test_acc


def train(net, opt, sched, n_epochs, train_loader, test_loader, eng = None):
    test_losses = []
    test_accs = []
    train_losses = []
    
    for ep in range(1, n_epochs + 1):
        train_loss, test_loss, test_acc = epoch(
            net, opt, sched, ep, train_loader, test_loader, private = eng is not None
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if eng is not None:
            print(eng.get_epsilon(1e-5))

    norms = [np.sqrt(torch.sum(ep_dist ** 2, axis = 1)) for ep_dist in grad_dist]
    print([n.mean().item() for n in norms])
    print([n.std().item() for n in norms])
    return train_losses, test_losses, test_accs


def remove_lines(file_content, prefixes_to_remove):
    lines = file_content.split("\n")
    filtered_lines = lines
    for prefix in prefixes_to_remove:
        filtered_lines = [
            line for line in filtered_lines if not line.strip().startswith(prefix)
        ]
    new_content = "\n".join(filtered_lines)
    return new_content
