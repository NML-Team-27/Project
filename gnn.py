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
        return self.layers(x, edge_index)


class GNN(nn.Module):
    """GNN model in two blocks:

    - temporal processing : Conv1D
    - spatial processing : GraphConv
    """

    def __init__(
        self,
        nb_graphconv: int=1,
        out_channels_graph: int=1,
        in_channels_graph: int = 1,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            nb_conv=nb_graphconv, in_channels=in_channels_graph, out_channels=out_channels_graph
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(out_channels_graph, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor
    ):
        x = self.conv(x, edge_index)

        return self.fc1(x).squeeze(1)


class GAT(nn.Module):
    """GAT architecture"""

    def __init__(
        self,
        in_channels_graph: int,
        out_channels_graph: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.gat_layers = pyg.nn.Sequential(
            "x, edge_index",
            [
                (
                    pyg.nn.GATv2Conv(
                        in_channels=in_channels_graph,
                        out_channels=out_channels_graph,
                        heads=heads,
                        edge_dim=1,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),
                (
                    pyg.nn.GATv2Conv(
                        in_channels=out_channels_graph * heads,
                        out_channels=out_channels_graph,
                        heads=heads,
                        edge_dim=1,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),
                (
                    pyg.nn.GATv2Conv(
                        in_channels=out_channels_graph * heads,
                        out_channels=out_channels_graph,
                        heads=heads,
                        edge_dim=1,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU()
            ],
        )

        self.fc = nn.Linear(out_channels_graph * heads, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor
    ):
        x = self.gat_layers(x, edge_index=edge_index)

        return self.fc(x).squeeze(1)
