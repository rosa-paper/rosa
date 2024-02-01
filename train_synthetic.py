import copy
import os
import os.path as osp

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import set_seeds, refactorize, get_experiment_name
import peftnet as pn


class LinearModel(nn.Module):
    """A simple linear model."""
    def __init__(self, in_features=768, out_features=32, bias=False):
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """ Forward pass.

        Args:
            x: [batch_size, in_features]

        Returns:
            y: [batch_size, out_features]

        """
        return self.l1(x)


class MLP2Layer(nn.Module):
    """A 2-layer multi-layer perceptron."""
    def __init__(self, in_features=768, out_features=32, hidden=64, bias=False):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden, bias=bias)
        self.l2 = nn.Linear(hidden, out_features, bias=bias)

    def forward(self, x):
        """ Forward pass.

        Args:
            x: [batch_size, in_features]

        Returns:
            y: [batch_size, out_features]

        """
        return self.l2(nn.functional.relu(self.l1(x)))


def build_synthetic_dataset(model, n_samples=1000, n_dims=768):
    """Build a synthetic dataset from a given model."""
    with torch.no_grad():
        x = torch.randn(n_samples, n_dims)
        y = model(x)
        return x, y


def evaluate_model(model, dataloader):
    with torch.no_grad():
        loss_fn = torch.nn.MSELoss()
        losses = []
        for i, batch in enumerate(dataloader):
            x_train, y_train = batch
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            losses.append(loss.item())
        return sum(losses) / len(losses)


@hydra.main(version_base=None, config_path="configs", config_name="conf_synthetic.yaml")
def main(cfg: DictConfig):

    # *** Configure HPs ***
    args = OmegaConf.to_container(cfg, resolve=True)

    # True model parameters (model to be approximated)
    model_name = args["model"]["name"]  # Model type
    true_rank = args["exps"]["true_rank"]  # Rank of the true model
    in_f = args["data"]["in_f"]  # Input features
    out_f = args["data"]["out_f"]  # Output features

    # PEFT model parameters
    peft_rank_max = args['exps']['peft_rank_max']  # Maximum rank to test
    peft_rank_step = args['exps']['peft_rank_step']  # Rank step
    factorize_steps = args['exps']['factorize_steps']  # Number of steps between refactorizations
    factorize_warmup = args['exps']['factorize_warmup']  # Number of steps to warmup before refactorizing

    # Dataset parameters
    n_train_samples = args['data']['n_train_samples']  # Number of samples in the train_dataset
    n_valid_samples = args['data']['n_valid_samples']  # Number of samples in the valid_dataset

    # Train parameters
    n_epochs = args['train']['epochs']  # Number of epochs to train
    bs = args['train']['batch_size']  # Batch size
    lr = args['train']['lr']  # Learning rate

    # Logging parameters
    log_freq = args['log_freq']  # Logging frequency

    # *** End of Config ***

    # Get experiment name
    filename = get_experiment_name(
        {"train": args["train"], "exps": args["exps"], "model": args["model"]}, mode="str"
    )
    # Set seeds
    set_seeds(42)

    models = {"linear": LinearModel, "mlp2": MLP2Layer}
    init_model = models[model_name](in_features=in_f, out_features=out_f, bias=False)
    true_model = pn.LoraNet(copy.deepcopy(init_model), rank=true_rank, init_method="random")
    x_train, y_train = build_synthetic_dataset(true_model, n_samples=n_train_samples, n_dims=in_f)
    x_valid, y_valid = build_synthetic_dataset(true_model, n_samples=n_valid_samples, n_dims=in_f)

    # Dataloader
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False)

    # Select experiment to run
    if args['exps']['name'] == "rosa_vs_lora":
        # Compare LoRA and ROSA at different ranks

        peft_models = {
            **{
               "LoRA (r={}):".format(max(r, 1)): pn.LoraNet(copy.deepcopy(init_model), rank=max(r, 1))
               for r in range(0, peft_rank_max+1, peft_rank_step)
              },
            **{
                "ROSA (r={}):".format(max(r, 1)): pn.RosaNet(copy.deepcopy(init_model), rank=max(r, 1))
                for r in range(0, peft_rank_max+1, peft_rank_step)
              },
            "FT": copy.deepcopy(init_model)
        }
        def color_and_marker_func(name):
            rank_colors = {
                max(r, 1): plt.cm.get_cmap('tab10')(i) for i, r in
                enumerate(range(0, peft_rank_max + 1, peft_rank_step))
            }
            rank = int(name.split('=')[-1][:-2]) if "FT" not in name else 0
            color = rank_colors[rank] if "FT" not in name else "black"
            marker = {"LoRA": "--", "ROSA": "-", "FT": "-"}[name.split(" ")[0]]
            return color, marker

    elif args['exps']['name'] == "rosa_ablation_top_bottom":
        # Compare ROSA with different factorization modes

        peft_models = {
            "ROSA (r={}, f={})".format(max(r, 1), fmode): pn.RosaNet(
                copy.deepcopy(init_model), rank=max(r, 1), factorize_mode=fmode.lower()
            )
            for r in range(0, peft_rank_max + 1, peft_rank_step)
            for fmode in ["Top", "Bottom", "Random"]
        }

        def color_and_marker_func(name):
            rank_colors = {
                max(r, 1): plt.cm.get_cmap('tab10')(i) for i, r in
                enumerate(range(0, peft_rank_max + 1, peft_rank_step))
            }
            rank = int(name.split('r=')[-1].split(',')[0])
            color = rank_colors[rank]
            marker = {"Top": "^", "Bottom": ".", "Random": "x"}[name.split("f=")[-1][:-1]]
            return color, marker

    else:
        raise ValueError("Invalid experiment name: {}".format(args['exps']['name']))

    # Train
    total_steps = n_epochs
    factorize_freq = max(total_steps // factorize_steps, 1)
    print("Total steps: {} | Factorize freq: {}\n".format(total_steps, factorize_freq))

    for name, model in peft_models.items():

        print(f"\nTraining {name}")

        # Optimizer
        optimizer = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW
        }[args["train"]["optimizer"]['name']](model.parameters(), lr=lr)

        # Make loss function
        loss_fn = torch.nn.MSELoss()

        # Train
        val_losses = []
        for i_epoch in range(n_epochs):

            if i_epoch > max(factorize_warmup, 0) and "ROSA" in name and i_epoch % factorize_freq == 0:
                model, optimizer = refactorize(model, optimizer)
                print(f"Refactorized {name} at epoch {i_epoch+1}")

            for i_iter, batch in enumerate(train_dataloader):

                x_train, y_train = batch
                y_pred = model(x_train)
                loss = loss_fn(y_pred, y_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if i_iter % log_freq == 0:
                    val_loss = evaluate_model(model, valid_dataloader)
                    val_losses.append(val_loss)
                    print(f"Epoch [{i_epoch+1:03d}/{n_epochs:03d}] | Batch [{i_iter:03d}/{len(train_dataloader):03d}] | "
                          f"Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        # Plot
        color, marker = color_and_marker_func(name)
        plt.plot(val_losses, marker, label=name, color=color)

    plt.xlabel("Iterations")
    plt.ylabel("Validation Loss")
    plt.legend()
    outfile = osp.join(args['output'], "synthetic_{}.png".format(filename))
    if not os.path.exists(args['output']):
        os.makedirs(args['output'])
    plt.savefig(outfile, dpi=300)
    print("Saved figure to {}".format(outfile))


if __name__ == '__main__':
    main()

    # 1-layer
    # python train_synthetic.py model.name=linear exps.peft_rank_max=8 exps.peft_rank_step=2 exps.true_rank=24 train.epochs=500

    # 2-layer
    # python train_synthetic.py model.name=mlp2 exps.peft_rank_max=8 exps.peft_rank_step=2 exps.true_rank=24 train.epochs=1000 &&
    # python train_synthetic.py data.out_f=10 model.name=mlp2 exps.peft_rank_max=8 exps.peft_rank_step=2 exps.true_rank=24 train.epochs=1000
