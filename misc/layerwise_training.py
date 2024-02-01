import torch
import torch.nn as nn
import copy


def forward_layerwise(layers, x, device='cpu', verbose=False):
    """ Forward pass through a list of layers.

    Description:
        For n = 0, 1, ..., N-1:
            Load layer n, forward pass using activation n, save activation n+1, unload layer n, load layer n+1,

    """
    activations = []
    x = x.to(device)
    for i, layer in enumerate(layers):
        if verbose:
            print(f"Forward pass through layer {i}...")
        layer = layer.to(device)
        x = layer(x)
        activations.append(x)
        layer.to('cpu')  # Unload layer from GPU
        del layer
    return activations


def backward_layerwise(layers, activations, loss, optimizers, device='cpu', verbose=True):
    """ Backward pass through a list of layers.

    Description:
        For n = N-1, N-2, ..., 1, 0:
            Load layer n, backward pass using activation n, save activation n-1, unload layer n, load layer n-1,

    """
    pass


def cuda_time_operation(func, func_kwargs, device='cuda:0', verbose=False):
    """ Time an operation on the GPU. """
    if verbose:
        print(f"Running operation on {device}...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(**func_kwargs)
    end.record()
    torch.cuda.synchronize()
    if verbose:
        print(f"Time elapsed: {start.elapsed_time(end)}ms")
    return start.elapsed_time(end)


if __name__ == '__main__':
    # Config
    input_size = 1024
    hidden_sizes = (1024, 1024, 1024)
    batch_size = 128

    # Create random TensorDataset
    dataset = torch.rand(1000, input_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create MLP
    input_layer = nn.Linear(input_size, hidden_sizes[0])
    hidden_layers = [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)]
    layers = [input_layer, *hidden_layers]

    # Create nn.Sequential
    mlp = nn.Sequential(*copy.deepcopy(layers))

    # Train Layer-wise
    loss_fn = nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(layers[i].parameters(), lr=0.001) for i in range(len(layers))]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    elapsed_sequential_samples = []
    elapsed_layerwise_samples = []
    iterations = 1000
    warmup = 100
    for t in range(iterations + warmup):

        # Forward pass (layer-wise)
        x = next(iter(dataloader))
        elapsed_layer_wise = cuda_time_operation(
            forward_layerwise, {'layers': layers, 'x': x, 'device': device}, device=device
        )
        if t >= warmup:
            elapsed_layerwise_samples.append(elapsed_layer_wise)

        # Forward pass (nn.Sequential)
        x = next(iter(dataloader))
        mlp_func = lambda x: mlp(x)
        mlp.to(device)
        x = x.to(device)
        elapsed_sequential = cuda_time_operation(
            mlp_func, {'x': x}, device=device
        )
        if t >= warmup:
            elapsed_sequential_samples.append(elapsed_sequential)

    # Compute mean and std
    elapsed_layer_wise = torch.tensor(elapsed_layerwise_samples)
    elapsed_sequential = torch.tensor(elapsed_sequential_samples)
    elapsed_layer_wise_mean = elapsed_layer_wise.mean().item()
    elapsed_sequential_mean = elapsed_sequential.mean().item()
    elapsed_layer_wise_std = elapsed_layer_wise.std().item()
    elapsed_sequential_std = elapsed_sequential.std().item()

    n_samples = len(elapsed_layer_wise)
    print(f"Layer-wise: {elapsed_layer_wise_mean:0.2f}ms ± {elapsed_layer_wise_std:0.2f}ms (n={n_samples})")
    print(f"Sequential: {elapsed_sequential_mean:0.2f}ms ± {elapsed_sequential_std:0.2f}ms (n={n_samples})")

