import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import peftnet as pn
from peftnet.peft_module.rosalinear import RosaLinear
from peftnet.peft_module.loralinear import LoraLinear
from utils.utils import AverageMeter

import transformers

def generate_data(in_features, n_samples, sigma_x=1, sigma_y=1):
    x = sigma_x*torch.randn(n_samples, in_features)
    y_fn = lambda x: x**2 + 2*x + 1 + sigma_y*torch.randn(n_samples, in_features)
    y = y_fn(x)
    return x, y


def get_latency(model, input_size, device, iters=100, warmup=100):
    """Get latency mean/std of a model on a given input size."""
    model.eval()
    model = model.to(device)
    input = torch.randn(*input_size).to(device)
    # with torch.no_grad():

    # warmup
    for _ in range(warmup):
        model(input)


    latency = []
    for _ in range(iters):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model(input)
        end.record()

        torch.cuda.synchronize()
        latency.append(start.elapsed_time(end))

    return np.mean(latency), np.std(latency)


def timed_train_loop(model, loss_fn, optimizer, train_dataloader, epochs=10, warmup=100, factorize_freq=-1, device=None):
    """Train a model and return the time each iteration took."""

    latencies = []
    device = torch.device('cuda:0') if device is None else device
    model = model.to(device)

    for t in range(epochs):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if factorize_freq > 0 and t % factorize_freq == 0:
            model.factorize()

        for i, batch in enumerate(train_dataloader):

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end.record()

        if t >= warmup:
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            latencies.append(elapsed_time / 1000)

        if t % 10 == 0 and t > 0:
            curr_mean = np.mean(latencies)
            curr_std = np.std(latencies)
            print(f'Epoch {t}, Latency {curr_mean:.4f} +/- {curr_std:.4f}')

    mean = np.mean(latencies)
    std = np.std(latencies)
    return mean, std


def main():

    # Linear Model
    in_features = 768
    out_features = 768
    batch_size = 64

    # Data
    n_samples = 10000

    # Dataloader
    x, y = generate_data(in_features, n_samples)
    train_dataset = torch.utils.data.TensorDataset(x, y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    Linear = transformers.modeling_utils.Conv1D
    # Models
    # sequential_model = nn.Sequential(
    #     nn.Linear(in_features, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, out_features),
    # )
    sequential_model = nn.Sequential(
        Linear(nf=128, nx=in_features),
        nn.ReLU(),
        Linear(nf=128, nx=128),
        nn.ReLU(),
        Linear(nf=128, nx=128),
        nn.ReLU(),
        Linear(out_features, 128),
    )


    loss_fn = nn.MSELoss()

    # PEFT models
    peft_models = {
        "rosanet": pn.RosaNetDebug(copy.deepcopy(sequential_model), rank=1).factorize(),
        "loranet": pn.LoraNetDebug(copy.deepcopy(sequential_model), rank=1),
    }

    device = torch.device('cuda:0')
    for name, model in peft_models.items():
        # mean, std = get_latency(model, (batch_size, in_features), device, iters=iterations)
        # print(f'{name} latency: {mean:.4f} +/- {std:.4f}')

        timed_train_loop(
            model,
            loss_fn,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            train_dataloader,
            factorize_freq=1 if name == "rosanet" else -1,
            device=device,
            epochs=100,
            warmup=2,
        )





if __name__ == '__main__':
    main()