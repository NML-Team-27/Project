import torch_geometric as pyg
import torch
from torch import Tensor
from torch import nn
from typing import Optional


class ConvBlock(nn.Module):
    """Graph convolution"""

    def __init__(self, nb_conv: int = 2, in_channels: int = -1, out_channels: int = 1) -> None:
        super().__init__()

        layers = []

        for _ in range(nb_conv):
            layers.append(
                (
                    pyg.nn.GraphConv(in_channels=in_channels, out_channels=out_channels),
                    "x, edge_index -> x",
                )
            )

            layers.append(nn.ReLU())

            in_channels = out_channels

        self.layers = pyg.nn.Sequential("x, edge_index", layers[:-1])

    def forward(self, x: Tensor, edge_index: Tensor):
        """Forward pass

        Args:
            x (Tensor): tensor (batch size x channels x in_channels)
            edge_index (Tensor): tensor (2 x nb edges)
        """
        return self.layers(x.float(), edge_index)


class GNN(nn.Module):
    """GNN model in two blocks:

    - temporal processing : Conv1D
    - spatial processing : GraphConv
    """

    def __init__(
        self,
        nb_graphconv: int = 1,
        out_channels_graph: int = 1,
        in_channels_graph: int = 1,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            nb_conv=nb_graphconv, in_channels=in_channels_graph, out_channels=out_channels_graph
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(out_channels_graph, out_channels_graph // 2)
        self.fc2 = nn.Linear(out_channels_graph // 2, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor
    ):
        x = self.conv(x, edge_index)
        x = self.relu(self.fc1(x))
        
        return self.fc2(x).squeeze(1)


class GAT(nn.Module):
    """GAT architecture"""

    def __init__(
        self,
        in_channels_graph: int,
        out_channels_graph: int,
        heads: int,
        nb_graph_conv: int,
        dropout: float,
    ) -> None:
        super().__init__()

        layers = []

        for _ in range(nb_graph_conv):
            layers.append(
                (
                    pyg.nn.GATv2Conv(
                        in_channels=in_channels_graph,
                        out_channels=out_channels_graph,
                        heads=heads,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
            )

            layers.append(nn.LeakyReLU())

            in_channels_graph = out_channels_graph * heads

        self.layers = pyg.nn.Sequential("x, edge_index", layers[:-1])
        self.relu = nn.LeakyReLU() 
        input_linear_size = out_channels_graph * heads
        self.fc1 = nn.Linear(input_linear_size, input_linear_size // 2)
        self.fc2 = nn.Linear(input_linear_size // 2, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor
    ):
        x = self.layers(x.float(), edge_index=edge_index)
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)