"""Script to train models.
"""
from copy import deepcopy
import logging
from typing import Any, Tuple
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm  # , GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from numpy.typing import NDArray
import numpy as np
from torch import nn
from torch.optim import AdamW
import torch_geometric as pyg
import torch

from src.utils import classification_report2dict, number_parameters
from src.models import gnn


def instantiate_model(model_name:str, out_channels_graph, in_channels_graph, heads, dropout) -> Any:
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
        )
    elif model_name == "gat":
        return gnn.GAT(
            in_channels_graph=in_channels_graph,
            out_channels_graph=out_channels_graph,
            heads=heads,
            dropout=dropout,
        )

    else:
        raise ValueError(f'Model "{model_name}" is not supported.')


def _log_return_artifacts(
    label_np: NDArray[np.int_],
    preds: NDArray[np.int_],
    verbose: bool,
    return_report: bool,
    return_predictions: bool,
    model,
    split: str = "train",
) -> Any:
    # Training scores
    report = classification_report(label_np, preds, output_dict=True)
    auroc = roc_auc_score(label_np, preds)

    if verbose:
        print(f"{split} report :")
        print(report)
        print(f"auroc : {auroc}")

    if return_report:
        if return_predictions:
            return model, (report, preds, label_np)
        else:
            return model, report
    else:
        if return_predictions:
            return model, preds, label_np
        else:
            return model
        

def validation_step(
    dataloader: pyg.loader.DataLoader, model: Any, criterion: Any, ep: int, device: str
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
        nb_graphs = 0

        for batch in dataloader:
            batch = batch.to(device)

            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y.double())

            running_loss += loss.item()
            nb_graphs += batch.num_graphs

            pred = torch.sigmoid(outputs)
            preds.append(int(pred > 0.5))
            labels_gt.append(batch.y.item())

    model.train()
    f1 = metrics.f1_score(labels_gt, preds)

    return f1


def fit(
    model: nn.Module,
    dataset,
    val_dataset,
    pos_weight: float,
    device: str,
    lr,
    batch_size, 
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
        dataset, batch_size=batch_size, shuffle=True
    )

    val_dataloader = pyg.loader.DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_val_f1 = -1
    best_params = None
    best_epoch = -1

    for ep in range(epochs):
        running_loss = 0
        nb_graphs = 0
        for i, batch in tqdm(
            enumerate(dataloader), desc=f"Epoch {ep + 1} / {epochs}"
        ):
            batch = batch.to(device)

            outputs = model(batch.x, batch.edge_index,batch.batch)
            # print(f'outputs : {outputs}')
            loss = criterion(outputs, batch.y.double())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            nb_graphs += batch.num_graphs

        avg_loss = running_loss / nb_graphs
      
        print(f"Avg. train loss of epoch {ep + 1} : {avg_loss:.4f}")

        val_f1_score = validation_step(val_dataloader, model, criterion, ep, device)

        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            best_params = deepcopy(model.state_dict())
            best_epoch = ep

    # Load best params based on validation F1-score
    model.load_state_dict(best_params)

    return model


def predict(
    model: nn.Module, dataset, device: str
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    preds = []
    labels_gt = []

    model = model.eval()

    dataloader = pyg.loader.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        batch = batch.to(device)

        pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch)).item()
        preds.append(int(pred > 0.5))
        labels_gt.append(batch.y.item())

    return np.array(preds), np.array(labels_gt)


def main_nn(
    dataset,
    val_dataset,
    pos_weight,
    model_name,
    lr,
    batch_size,
    epoch,
    mode="train",
    return_predictions: bool = False,
    verbose: bool = True,
    weight_path=None
) -> Any:
    """Instantiate and train a GNN model according to the config.

    Args:
        config (DataConf): config to use
        dataset (EEGDatasetGeom): train dataset
        val_dataset (EEGDatasetGeom): validation dataset
        return_report (bool, optional): whether to return the classification report. Defaults to False.
        return_predictions (bool, optional): whether to return the predictions. Defaults to False.
    """
    if torch.cuda.is_available():
        device = "cuda"
        logging.info("Device : GPU")
    else:
        device = "cpu"

    model = instantiate_model(model_name).double()
    nb_params = number_parameters(model)
    logging.info(f"Number of trainable parameters in model : {nb_params}")

    model = fit(
        model, dataset, val_dataset, pos_weight, device,lr, batch_size, epoch
    )
    preds, train_label_np = predict(model, dataset, device=device)

    return _log_return_artifacts(
        train_label_np, preds, verbose, return_predictions, model
    )
