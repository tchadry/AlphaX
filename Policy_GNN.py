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
        self.conv2 = GCNConv(16,  16)
        self.conv3 = GCNConv(16, 1)
        self.activ = nn.Linear(16, 1)
    
    def forward(self, data):
        x, edges, choices = data.x, data.edge_index, data.y
        
        x = self.conv1(x, edges)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edges)
        x = F.relu(x)
        
        c = self.conv3(x, edges)
        choice = torch.masked_select(c.squeeze(), choices)
        choice = F.softmax(choice, dim=0)
        
        v = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=data.long))
        value = self.activ(v)

        return choice, value