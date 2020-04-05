import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class GNN_Policy(nn.Module):
    def __init__(self, d=2):
        super(GNN_Policy, self).__init__()
        self.conv1 = GCNConv(d,  16)
        self.conv2 = GCNConv(16, 1)
        self.activ    = nn.Linear(16, 1)
    
    def forward(self, graph):
        pass