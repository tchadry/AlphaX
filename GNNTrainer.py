import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class GNN_Trainer:

    def __init__(self, model, example_queue):
        self.model = model
        self.example_queue = example_queue
        self.losses = []

    def train_example(self):
        self.model.train()

        example = self.example_queue.get()
        graph, choice_probs, value = example["graph"], example["choice_probs"], example["pred_value"]

        pred_choices, pred_value = self.model(graph)
        loss = loss_fn(pred_choices, choice_probs) + (0.2 * loss_fn(pred_value, value))

        self.losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train_all(self):
        self.model.train()
        while len(self.example_queue)!=0:

            example = self.example_queue.pop(0)
            graph, choice_probs, value = example["graph"], example["choice_probs"], example["pred_value"]

            pred_choices, pred_value = self.model(graph)
            loss = loss_fn(pred_choices, choice_probs) + (0.2 * loss_fn(pred_value, value))

            self.losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
