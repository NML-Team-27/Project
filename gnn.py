import torch_geometric as pyg
import torch
from torch import Tensor
from torch import nn
from typing import Optional


class ConvBlock(nn.Module):
    """class represent a Graph convolutio block """

    def __init__(self, nb_conv: int = 2, in_channels: int = -1, out_channels: int = 1) -> None:
        """
        Args:
            nb_conv  (int): the number of convolution in the convolution layer
            in_channels (int): the number of channels as input of the convolution layer
            out_channels (int): the number of channels as output of the convolution layer
        """
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
    """GNN model
    """

    def __init__(
        self,
        nb_graphconv: int = 1,
        out_channels_graph: int = 1,
        in_channels_graph: int = 1,
    ) -> None:
        """
        Args:
            nb_graph_conv (int): the number of convolution in the convolution layer
            out_channels_graph (int): the number of channels as output of the convolution layer
            in_channels_graph (int): the number of channels as input of the convolution layer
        """
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
        """Forward pass

        Args:
            x (Tensor): tensor (batch size x channels x in_channels)
            edge_index (Tensor): tensor (2 x nb edges)
        """
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
        """
        Args:
            in_channels_graph (int): the number of channels as input of the convolution layer
            out_channels_graph (int): the number of channels as output of the convolution layer
            heads (int): the number of heads (in the GAT network)
            nb_graph_conv (int): the number of convolution in the convolution layer
            dropout (float): the dropout rate in the GAT architecture 
        """
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
        """Forward pass

        Args:
            x (Tensor): tensor (batch size x channels x in_channels)
            edge_index (Tensor): tensor (2 x nb edges)
        """
        x = self.layers(x.float(), edge_index=edge_index)
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)