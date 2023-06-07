from pecanpy import pecanpy as pp
import utils
import networkx as nx
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import json
from numpy.typing import NDArray
from typing import Dict
import pandas as pd


def init_result_folder(path: str) -> nx.Graph:
    """
    Initialize the result folder and return the graph
    :param str: the path in which we should save the results
    :return: the created graph
    """
    # Check if the 'results' folder exists in the given path
    if not os.path.exists(os.path.join(path, 'results')):
        # Create the 'results' folder if it doesn't exist
        os.mkdir(os.path.join(path, 'results'))

    # Create an empty 'results.json' file inside the 'results' folder
    with open(os.path.join(path, 'results', 'results.json'), 'w') as f:
        json.dump(dict(), f)

    # Extract the city name
    path_split = path.split('/')
    city_name = path_split[-1]

    # Get the data directory by excluding the last part of the path
    data_dir = os.path.join(*path_split[:-1])

    # Create the city graph
    graph, _ = utils.create_city_graph(city_name, data_dir)

    # Convert the graph to a directed graph (DiGraph)
    graph = nx.DiGraph(graph)

    # Create a mapping to relabel the nodes of the graph
    mapping = {k: i for i, k in enumerate(sorted(graph.nodes()))}

    # Relabel the nodes of the graph using the mapping
    graph = nx.relabel_nodes(graph, mapping)

    # Save the graph as an edge list
    path_edg_file = os.path.join(path, 'adj_mat.edg')
    nx.write_edgelist(graph, path_edg_file, data=False, delimiter='\t')

    return graph


def save_embeddings(path: str, p: float, q: float, num_walks: int, dim: int = 256) -> NDArray[np.float_]:
    """
    Create embeddings, save and return them
    :param path: the path to the city folder
    :param p: the parameter p of Node2Vec
    :param q: the parameter q of Node2Vec
    :param num_walks: the number of random walks from each node
    :param dim: the dimensionality of the embeddings
    :return: the embeddings for each node in the graph
    """
    path_edg_file = os.path.join(path, 'adj_mat.edg')

    # Initialize the SparseOTF object for node2vec with the given parameters
    g = pp.SparseOTF(p=p, q=q, verbose=False, workers=16)

    # Read the graph from the edge list
    g.read_edg(path_edg_file, weighted=False, directed=True)

    # Generate node embeddings using node2vec 
    embeddings = g.embed(dim=dim, num_walks=num_walks)

    # Save the embeddings
    np.save(os.path.join(path, 'results', 'node2vec_embeddings.npy'), embeddings)

    # Return the saved embeddings
    return embeddings


def train_svm(features: NDArray[np.float_], targets: NDArray[np.float_], seed: int = 42,scale_features=False,X_test=None, y_test=None) -> Dict:
    """
    Trains an SVM classifier and returns the classification results.

    Args:
        features (NDArray[np.float_]): The input features for training the classifier.
        targets (NDArray[np.float_]): The target labels for training the classifier.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        scale_features (bool, optional): Indicates whether to scale the input features. Defaults to False.
        X_test (NDArray[np.float_], optional): The input features for testing the classifier. Defaults to None.
        y_test (NDArray[np.float_], optional): The target labels for testing the classifier. Defaults to None.

    Returns:
        Dict: A dictionary containing the classification results for the training and testing sets.

    """
    # Split the features and targets into training and testing sets if the test set is not provided
    if X_test is None:
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2,
                                                            stratify=targets, random_state=seed)
    else:
        X_train, y_train = features, targets

    # Initialize the SVM model 
    model = SVC(class_weight='balanced', C=1)

    # Scale the features if required
    if scale_features:
        standard_scaler = StandardScaler()
        standard_scaler.fit(X_train)
        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

    # Fit the SVM model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the training and testing data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Compute classification reports for the training and testing sets
    results = {}
    results['train'] = classification_report(y_train, train_preds, output_dict=True)
    results['test'] = classification_report(y_test, test_preds, output_dict=True)

    # Return the results
    return results

def cities_loop(data_path: str, p: float, q: float, num_walks: int, dim: int) -> None:
    """
    :param data_path: the folder in which we have all data
    :param p: the parameter p for Node2Vec
    :param q: the parameter q for Node2Vec
    :param num_walks: the number of random walks from each node
    :param dim: the dimensionality of the embeddings
    """
    # Iterate over all cities in the data path
    for entry in os.scandir(data_path):
        if entry.is_dir():
            # Get the name of the city from the directory entry
            city = entry.name
            
            print(f"Start training for city {city}")
            
            # Initialize the result folder for the city and obtain the graph
            graph = init_result_folder(entry.path)
            
            # Generate embeddings
            embeddings = save_embeddings(entry.path, p, q, num_walks=num_walks, dim=dim)
            
            # Get the classes from the graph
            targets = dict(graph.nodes(data='city_center'))
            targets = np.array([targets[v] for v in range(len(targets))])
            
            # Train an SVM classifier using the node embeddings
            results = train_svm(embeddings, targets)
            
            # Read the handcrafted features for the city from a CSV file
            df_features = pd.read_csv(os.path.join(data_path, "handcrafted_features.csv"))
            df_features = df_features[df_features["city"] == city]
            
            # Extract the classes and features
            targets_handcrafted = df_features["city_center"].values
            features_handcrafted = df_features.drop(["stop_I", "name", "city_center", "city", "Unnamed: 0"], axis=1)
            
            # Train an SVM classifier using the handcrafted features
            results_handcrafted = train_svm(features_handcrafted, targets_handcrafted, scale_features=True)
            
            # Update the 'results.json' file with the classification results
            with open(os.path.join(entry.path, 'results', 'results.json'), 'r') as f:
                res_dict = json.load(f)
            res_dict['baseline_svm'] = results
            res_dict["handcrafted_svm"] = results_handcrafted
            with open(os.path.join(entry.path, 'results', 'results.json'), 'w') as f:
                json.dump(res_dict, f)
