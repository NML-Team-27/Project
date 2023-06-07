"""Script to train models.
"""
from copy import deepcopy
import logging
from typing import Any, Tuple, List
from sklearn import metrics
from sklearn.metrics import classification_report
from numpy.typing import NDArray
import numpy as np
from torch import nn
from torch.optim import AdamW
import torch_geometric as pyg
import torch
from tqdm import tqdm
import gnn
import os 
import networkx as nx
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
from torch_geometric.data import Data


def instantiate_model(model_name:str, out_channels_graph:int, in_channels_graph:int, nb_graph_conv:int, heads:int = None, dropout: float = 0.0) -> Any:
    """Instantiate the model given the configuration.

    Args:
        model_name (Str): the name of the model we need to instantiate
        out_channels_graph (int): the number of channels as output of the convolution layer
        in_channels_graph (int): the number of channels as input of the convolution layer
        nb_graph_conv (int): the number of convolution in the convolution layer
        heads (int): the number of heads (in the GAT network)
        dropout (float): the dropout rate in the GAT architecture 

    Raises:
        ValueError: if model is not supported

    Returns:
        Any: instance of the model (untrained)
    """
    # Return the model based on the name
    if model_name == "gnn":
        return gnn.GNN(
            out_channels_graph=out_channels_graph,
            in_channels_graph=in_channels_graph,
            nb_graphconv=nb_graph_conv,
        )
    elif model_name == "gat":
        return gnn.GAT(
            in_channels_graph=in_channels_graph,
            out_channels_graph=out_channels_graph,
            heads=heads,
            dropout=dropout,
            nb_graph_conv=nb_graph_conv
        )

    else:
        raise ValueError(f'Model "{model_name}" is not supported.')
        

def validation_step(
    dataloader: pyg.loader.DataLoader, val_ids: List[int], model: Any, criterion: Any, device: str
) -> float:
    """Perform a validation step and compute the F1-score.

    Args:
        dataloader (pyg.loader.DataLoader): validation dataset
        model (Any): fitted model
        criterion (Any): loss function
        ep (int): epoch index
        device (str): device (cpu or cuda)
    """
    # Put the model on evaluation to avoid gradient update and set dropout to work correctly
    model.eval()
    preds = []
    labels_gt = []

    with torch.no_grad():
        running_loss = 0
        nb_nodes = 0

        for batch in dataloader:
            # Get model outputs and compute loss 
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            val_outputs = outputs[val_ids]
            loss = criterion(val_outputs, batch.y[val_ids])
            running_loss += loss.item()
            
            # Get the prediction for the batch
            pred = torch.sigmoid(val_outputs) > 0.5
            preds += pred.tolist()
            labels_gt += batch.y[val_ids].tolist()
            nb_nodes += val_outputs.shape[0]

        # print(f'Validation loss : {running_loss / nb_nodes}')
    # Put the model in train mode again
    model.train()
    # Compute the score of the model on the given data
    f1 = metrics.f1_score(labels_gt, preds)
    recall = metrics.recall_score(labels_gt, preds)
    precision = metrics.precision_score(labels_gt, preds)
    # print(f'Validation F1-score : {f1}')
    # print(f'Validation recall-score : {recall}')
    # print(f'Validation precision-score : {precision}')


    return f1


def fit(
    model: nn.Module,
    dataset,
    train_ids,
    val_ids,
    pos_weight: float,
    device: str,
    lr,
    epochs
):
    """Train the given model.

    Args:
        model (nn.Module): model to train
        config (DataConf): config
        dataset (Dataset): train dataset
        val_dataset (Dataset): validation dataset
        pos_weight (float): weight for label 1 in loss function
        device (str): device (cpu or cuda)
        lr (float): the learning rate for the training
        epochs (int): the number of epochs to train the models
    """
    # Create model, optimizers, loss and move them to the correct device
    model = model.train()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(device)
    dataloader = pyg.loader.DataLoader(
        dataset, batch_size=1
    )

    best_val_f1 = -1
    best_params = None

    for ep in range(epochs):
        running_loss = 0
        nb_nodes = 0
        # for i, batch in tqdm(
        #     enumerate(dataloader), desc=f"Epoch {ep + 1} / {epochs}"
        # ):
        for i, batch in enumerate(dataloader):
            # Get prediction and comput eloss
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            train_outputs = outputs[train_ids]
            loss = criterion(train_outputs, batch.y[train_ids])

            # Update gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss count
            running_loss += loss.item()
            nb_nodes += train_outputs.shape[0]
        # Compute average lsos
        avg_loss = running_loss / nb_nodes
      
        #print(f"Avg. train loss of epoch {ep + 1} : {avg_loss:.4f}")

        # Compute the f1 score on the validation to determine which model is the best
        val_f1_score   = validation_step(dataset, val_ids, model, criterion, device)

        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            best_params = deepcopy(model.state_dict())

    # Load best params based on validation F1-score
    model.load_state_dict(best_params)

    # print('=============================')
    # print(f'Best validation F1-score : {best_val_f1}')
    # print('=============================')

    return model, best_val_f1


def predict(
    model: nn.Module, dataset, device: str
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """ predict on the given dataloader

    Args:
        model (nn.Module): model to train
        dataset (Dataset) : prediction dataset
        device (str): device (cpu or cuda)
    """
    preds = []
    labels_gt = []
    # put the model in eval mode
    model = model.eval()

    # Create data loader
    dataloader = pyg.loader.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        batch = batch.to(device)
        # Create prediction on the given set 
        pred = torch.sigmoid(model(batch.x, batch.edge_index)) > 0.5
        preds += pred.long().tolist()
        labels_gt += batch.y.long().tolist()

    return np.array(preds), np.array(labels_gt)


def train_gnn(
    dataset:np.array,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    pos_weight: float,
    model_name: str,
    lr: float,
    epochs: int,
    out_channels_graph: int,
    in_channels_graph: int,
    nb_graph_conv:int ,
    heads:int = None,
    dropout:float = 0.0,
) -> Any:
    """Train the model defined by the parameteres.

    Args:
        dataset (Dataset): dataset containing all data
        train_ids (List[int]): list of all train indices
        val_ids (List[int]): list of all val indices
        test_ids (List[int]): list of all test indices
        pos_weights (float): rescaling weights as data is un balanced
        model_name (str): the name of the model we should load
        lr (float): the learning rate for the training
        epochs (int): the number of epochs to train the models
        out_channels_graph (int): the number of channels as output of the convolution layer
        in_channels_graph (int): the number of channels as input of the convolution layer
        nb_graph_conv (int): the number of convolution in the convolution layer
        heads (int): the number of heads (in the GAT network)
        dropout (float): the dropout rate in the GAT architecture 

    Returns: 
        - Dictionnary with the classification report of the train and test set
        - The validation f1-score 
    """

    if torch.cuda.is_available():
        device = "cuda"
        logging.info("Device : GPU")
    else:
        device = "cpu"
    # Create the model
    model = instantiate_model(model_name, out_channels_graph, in_channels_graph, nb_graph_conv, heads, dropout)
    # Count the number of parameters 
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'Number of parameters in model : {nb_params}')

    # Fit the model 
    model, best_val_f1 = fit(
        model, dataset, train_ids, val_ids, pos_weight, device, lr, epochs
    )

    # Make the predictions
    preds, label_gt = predict(model, dataset, device=device)

    # Compute the classification report
    train_label_np = label_gt[train_ids]
    train_preds = preds[train_ids]
    test_label_np = label_gt[test_ids]
    test_preds = preds[test_ids]

    results = {}
    results['train'] = classification_report(train_label_np, train_preds, output_dict=True)
    results['test'] = classification_report(test_label_np, test_preds, output_dict=True)

    return results, best_val_f1



def cities_loop_gnn(data_path: str, seed: int = 42) -> None:
    """
    Create one model of GNN and GAT for each city
    Args
        data_path (str): the path to the folder with all cities
        seed (int): the seed to fix the randomness
    """
    for entry in os.scandir(data_path):
        if entry.is_dir():
            city = entry.name
            
            # Get the feature of each city
            df_features = pd.read_csv(os.path.join(data_path,"handcrafted_features.csv"))
            df_features = df_features[df_features["city"]==city]
            targets_handcrafted = df_features["city_center"].values
            features_handcrafted = df_features.drop(["stop_I", "name", "city_center", "city","Unnamed: 0"],axis=1).values
            # Create the graph
            graph = nx.read_edgelist(os.path.join(entry.path, 'adj_mat.edg'), create_using=nx.DiGraph)

            # Create adjcacncey matrix
            print(f'number of edges in nx graph : {graph.number_of_edges()}')
            adj_mat = nx.adjacency_matrix(graph, weight=None) # not weighted
            print(f'adj mat shape : {adj_mat.shape}')
            edge_index, _ = pyg.utils.from_scipy_sparse_matrix(adj_mat)
            print(f'adj mat shape sparse tensor : {edge_index.shape}')

            # Create data
            d = Data(
                x=torch.from_numpy(features_handcrafted),
                y=torch.tensor(targets_handcrafted).clone(),
                edge_index=edge_index.clone(),
            )

            # Split the dataset into train, val, test
            train_ids, test_ids = train_test_split(
                np.arange(features_handcrafted.shape[0]), test_size=0.2, stratify=targets_handcrafted, random_state=seed
            )

            train_ids, val_ids = train_test_split(
                train_ids, test_size=0.2, stratify=targets_handcrafted[train_ids], random_state=seed
            )
            # Rescaling factor as we have unbalanced
            pos_weight = np.sum(targets_handcrafted[train_ids] == 0) / np.sum(targets_handcrafted[train_ids] == 1)
            print(f'Pos weight : {pos_weight}')
            # train the two models 
            results_gnn = train_gnn(
                [d],
                train_ids,
                val_ids,
                test_ids,
                pos_weight=pos_weight,
                model_name='gnn',
                lr=1e-3,
                epochs=20,
                out_channels_graph=32,
                in_channels_graph=18,
                nb_graph_conv=3,
                heads=5,
                dropout=0.0
            )
            results_gat = train_gnn(
                [d],
                train_ids,
                val_ids,
                test_ids,
                pos_weight=pos_weight,
                model_name='gat',
                lr=1e-3,
                epochs=20,
                out_channels_graph=32,
                in_channels_graph=18,
                nb_graph_conv=3,
                heads=8,
                dropout=0.0
            )
            # Save the results 
            with open(os.path.join(entry.path, 'results', 'results.json'), 'r') as f:
                res_dict = json.load(f)
            print()
            res_dict['gnn'] = results_gnn
            res_dict["gat"] = results_gat

            with open(os.path.join(entry.path, 'results', 'results.json'), 'w') as f:
                json.dump(res_dict, f)