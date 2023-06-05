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


def instantiate_model(model_name:str, out_channels_graph, in_channels_graph, nb_graph_conv, heads = None, dropout = 0.0) -> Any:
    """Instantiate the model given the configuration.

    Args:
        config (DataTrainConf): configuration

    Raises:
        ValueError: if model is not supported

    Returns:
        Any: instance of the model (untrained)
    """
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
    model.eval()
    preds = []
    labels_gt = []

    with torch.no_grad():
        running_loss = 0
        nb_nodes = 0

        for batch in dataloader:
            batch = batch.to(device)

            outputs = model(batch.x, batch.edge_index)

            val_outputs = outputs[val_ids]

            loss = criterion(val_outputs, batch.y[val_ids])

            running_loss += loss.item()

            pred = torch.sigmoid(val_outputs) > 0.5
            preds += pred.tolist()
            labels_gt += batch.y[val_ids].tolist()
            nb_nodes += val_outputs.shape[0]

        # print(f'Validation loss : {running_loss / nb_nodes}')

    model.train()
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
        dataset (EEGDatasetGeom): train dataset
        val_dataset (EEGDatasetGeom): validation dataset
        pos_weight (float): weight for label 1 in loss function
    """
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
            batch = batch.to(device)

            outputs = model(batch.x, batch.edge_index)

            train_outputs = outputs[train_ids]

            loss = criterion(train_outputs, batch.y[train_ids])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            nb_nodes += train_outputs.shape[0]

        avg_loss = running_loss / nb_nodes
      
        #print(f"Avg. train loss of epoch {ep + 1} : {avg_loss:.4f}")

        val_f1_score = validation_step(dataset, val_ids, model, criterion, device)

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
    preds = []
    labels_gt = []

    model = model.eval()

    dataloader = pyg.loader.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        batch = batch.to(device)

        pred = torch.sigmoid(model(batch.x, batch.edge_index)) > 0.5
        preds += pred.long().tolist()
        labels_gt += batch.y.long().tolist()

    return np.array(preds), np.array(labels_gt)


def train_gnn(
    dataset,
    train_ids,
    val_ids,
    test_ids,
    pos_weight,
    model_name,
    lr,
    epochs,
    out_channels_graph,
    in_channels_graph,
    nb_graph_conv,
    heads = None,
    dropout = 0.0,
) -> Any:
    if torch.cuda.is_available():
        device = "cuda"
        logging.info("Device : GPU")
    else:
        device = "cpu"

    model = instantiate_model(model_name, out_channels_graph, in_channels_graph, nb_graph_conv, heads, dropout)

    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'Number of parameters in model : {nb_params}')

    model, best_val_f1 = fit(
        model, dataset, train_ids, val_ids, pos_weight, device, lr, epochs
    )

    preds, label_gt = predict(model, dataset, device=device)

    train_label_np = label_gt[train_ids]
    train_preds = preds[train_ids]
    test_label_np = label_gt[test_ids]
    test_preds = preds[test_ids]

    results = {}
    results['train'] = classification_report(train_label_np, train_preds, output_dict=True)
    results['test'] = classification_report(test_label_np, test_preds, output_dict=True)

    return results, best_val_f1