import torch_geometric as pyg
import torch
from torch import Tensor
from torch import nn
from typing import Optional

from src.constants import NB_CHANNELS



class ConvBlock(nn.Module):
    """Graph convolution"""

    def __init__(self, nb_conv: int = 2, in_channels: int = -1, out_channels: int = 1) -> None:
        super().__init__()

        layers = []

        for _ in range(nb_conv):
            layers.append(
                (
                    pyg.nn.GraphConv(in_channels=in_channels, out_channels=out_channels),
                    "x, edge_index, edge_weight -> x",
                )
            )

            layers.append(nn.ReLU())

            in_channels = out_channels

        self.layers = pyg.nn.Sequential("x, edge_index, edge_weight", layers[:-1])

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        """Forward pass

        Args:
            x (Tensor): tensor (batch size x channels x in_channels)
            edge_index (Tensor): tensor (2 x nb edges)
        """
        return self.layers(x, edge_index, edge_weight)


class GNN(nn.Module):
    """GNN model in two blocks:

    - temporal processing : Conv1D
    - spatial processing : GraphConv
    """

    def __init__(
        self,
        nb_conv1d: int,
        nb_graphconv: int,
        kernel_size: int,
        stride: int,
        out_channels_temp: int,
        out_channels_graph: int,
        in_channels_graph: int = 1,
    ) -> None:
        super().__init__()

        self.temporal_block = TemporalConvBlock(
            nb_conv=nb_conv1d,
            kernel_size=kernel_size,
            stride=stride,
            out_channels=out_channels_temp,
            in_channels=1,  # NB_CHANNELS
        )

        self.spatial_block = SpatialConvBlock(
            nb_conv=nb_graphconv, in_channels=in_channels_graph, out_channels=out_channels_graph
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(out_channels_graph * NB_CHANNELS, 1)  # out_channels_graph // 2)
        # self.fc2 = nn.Linear(out_channels_graph // 2, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch: Optional[Tensor] = None
    ):
        batch_size = torch.unique(batch).shape[0]
        x = self.relu(self.temporal_block(x.double()))
        x = self.relu(self.spatial_block(x, edge_index, edge_weight))

        # Aggregate features of all nodes
        # x = pyg.nn.global_add_pool(x, batch=batch)
        x = x.reshape(batch_size, -1)  # Concatenate features of all nodes

        # x = self.relu(self.fc1(x))
        return self.fc1(x).squeeze(1)


class GAT(nn.Module):
    """GAT architecture"""

    def __init__(
        self,
        nb_conv1d: int,
        kernel_size: int,
        stride: int,
        out_channels_temp: int,
        in_channels_graph: int,
        out_channels_graph: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv_layer = TemporalConvBlock(
            nb_conv=nb_conv1d,
            kernel_size=kernel_size,
            stride=stride,
            in_channels=1,
            out_channels=out_channels_temp,
        )

        self.gat_layers = pyg.nn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    pyg.nn.GATv2Conv(
                        in_channels=in_channels_graph,
                        out_channels=out_channels_graph,
                        heads=heads,
                        edge_dim=1,
                        dropout=dropout,
                    ),
                    "x, edge_index, edge_attr -> x",
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
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
            ],
        )

        self.fc = nn.Linear(out_channels_graph * heads, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Optional[Tensor] = None
    ):
        batch_size = torch.unique(batch).shape[0]
        x = self.conv_layer(x)
        x = self.gat_layers(x.double(), edge_index=edge_index, edge_attr=edge_attr.double())

        x = pyg.nn.global_add_pool(x, batch=batch)
        # x = x.reshape(batch_size, -1) # Concatenate features of all nodes

        return self.fc(x).squeeze(1)


class GlobalGAT(nn.Module):
    """Global GAT architecture inspired from "A META-GNN APPROACH TO PERSONALIZED SEIZURE DETECTION AND
    CLASSIFICATION" : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094957&tag=1
    """

    def __init__(
        self,
        nb_conv1d: int,
        kernel_size: int,
        stride: int,
        out_channels_temp: int,
        in_channels_graph: int,
        out_channels_graph: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv_layer = TemporalConvBlock(
            nb_conv=nb_conv1d,
            kernel_size=kernel_size,
            stride=stride,
            in_channels=1,
            out_channels=out_channels_temp,
        )

        self.gat_layers = pyg.nn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    pyg.nn.GATv2Conv(
                        in_channels=in_channels_graph,
                        out_channels=out_channels_graph,
                        heads=heads,
                        edge_dim=1,
                        dropout=dropout,
                    ),
                    "x, edge_index, edge_attr -> x",
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
                    "x, edge_index, edge_attr -> x",
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
                    "x, edge_index, edge_attr -> x",
                ),
                nn.ReLU(),
            ],
        )

        self.fc = nn.Linear(out_channels_graph * heads, 1)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Optional[Tensor] = None
    ):
        # batch_size = torch.unique(batch).shape[0]
        # x = self.conv_layer(x)
        x = self.gat_layers(x.double(), edge_index=edge_index, edge_attr=edge_attr.double())

        x = pyg.nn.global_add_pool(x, batch=batch)
        # x = x.reshape(1, -1)  # Concatenate features of all nodes

        # return torch.sigmoid(self.fc(x).squeeze(1)).double()
        return self.fc(x).squeeze(1)
