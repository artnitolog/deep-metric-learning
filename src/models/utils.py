import torch
from transformers.activations import GELUActivation

class MLPHead(torch.nn.Module):
    def __init__(self, dims, dropout=0):
        super().__init__()
        if len(dims) == 2 and dims[1] is None:
            self.mlp = torch.nn.Identity()
            return
        layers = []
        for dim_from, dim_to in zip(dims[:-2], dims[1:-1]):
            layers.append(torch.nn.Linear(dim_from, dim_to))
            layers.append(GELUActivation())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    
class CombinedModel(torch.nn.Module):
    def __init__(self, trunk, pool_name, head, freeze_trunk=False):
        super().__init__()
        if freeze_trunk:
            for param in trunk.parameters():
                param.requires_grad = False
        self.trunk = trunk
        self.pool_name = pool_name
        self.head = head

    def forward(self, x):
        return self.head(getattr(self.trunk(x), self.pool_name))


class CombinedTextModel(torch.nn.Module):
    def __init__(self, trunk, head, freeze_trunk=False):
        super().__init__()
        if freeze_trunk:
            for param in trunk.parameters():
                param.requires_grad = False
        self.trunk = trunk
        self.head = head

    def forward(self, x):
        return self.head(self.trunk(*x).last_hidden_state[:, 0])

