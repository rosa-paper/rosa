import torch
import torch.nn as nn

# todo: make batch_size larger
# todo: try pip install torchviz


class AETLinear(nn.Module):
    def __init__(self, aet=True, in_features=10, out_features=10, rank=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a_trainable = aet
        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.a = torch.randn(in_features, rank)
        self.b = nn.Parameter(torch.randn(rank, out_features), requires_grad=True)
        # if not self.b_trainable else nn.Parameter(torch.randn(rank, out_features), requires_grad=self.b_trainable)

    def __repr__(self):
        # Show which parameters are trainable
        trainable_params = {
            'w': self.w.requires_grad,
            'a': self.a.requires_grad,
            'b': self.b.requires_grad
        }
        return f"{self.__class__.__name__}({', '.join([f'{k} trainable={v}' for k, v in trainable_params.items()])})"

    def forward(self, x):
        """ Forward pass of AET layer

        Args:
            x: [batch_size, in_features]

        Returns:

        """
        self.a = self.a.to(x.device)
        self.b = self.b.to(x.device)
        return x @ self.w + (x @ self.a) @ self.b

class Model(nn.Module):
    def __init__(self,  in_features=10, hidden_features=10, out_features=10, rank=1, aet=True):
        super().__init__()
        self.l1 = AETLinear(aet=aet, in_features=in_features, out_features=hidden_features, rank=rank)
        self.l2 = AETLinear(aet=aet, in_features=hidden_features, out_features=out_features, rank=rank)

    def forward(self, x):
        rlu = torch.nn.functional.relu
        return self.l2(rlu(self.l1(x)))


def report_memory_usage(message="", width=30):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # in MBs
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # in MBs
    print(f"{message.ljust(width)} --> Allocated memory: {allocated:.5f}MB, Reserved memory: {reserved:.5f}MB")


def main():
    # Define parameters
    params = {
        "batch_size": 512*32,
        "in_features": 1024,
        "out_features": 32,
        "rank": 1,
        "aet": False,
        "hidden_features": 100,
        "device": "cuda:0"
    }

    # Print parameters
    print("Experiment Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("-" * 50)

    device = torch.device(params["device"])

    report_memory_usage("Initial memory")

    model = Model(in_features=params["in_features"], hidden_features=params["hidden_features"], out_features=params["out_features"], rank=params["rank"], aet=params["aet"])
    report_memory_usage("After model creation")

    print(model)

    x_true = torch.randn(params["batch_size"], params["in_features"], device=device)
    y_true = torch.randn(params["batch_size"], params["out_features"], device=device)
    report_memory_usage("After data creation")

    # Move to GPU
    model.to(device)
    report_memory_usage("After moving model to GPU")

    # Forward pass
    loss = torch.nn.functional.mse_loss(model(x_true), y_true)
    print(f"Loss: {loss}")

    report_memory_usage("After forward pass")

if __name__ == "__main__":
    main()