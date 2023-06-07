import numpy as np
import pandas as pd
import networkx as nx
from os.path import isdir, join
from os import listdir
import networkx as nx
from typing import Dict, List,Tuple
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(rc={'figure.max_open_warning': 0})

def get_all_cities_in_directory(directory:str)->List[str]:
    """
    Get the list of all cities in the directory
    :param str: the directory in which we should find the cities
    :return: the list of cities
    """
    if(isdir(directory)):
        return sorted([d for d in listdir(directory) if isdir(join(directory, d))])
    return None


def gather_all_data(path: str, nodes: bool = True):
    """
    Gather the data from all cities in the given path 
    :param path: the path in which we should load the data
    :param nodes: boolean telling if we load nodes or edges.
    :return: a dataframe with all the nodes / edges data. 
    """
    # get all cities in the directory
    directories = get_all_cities_in_directory(path)
    data = []

    # Load the edges or nodes with the correct separator
    file = 'network_nodes_labeled.csv' if nodes else 'network_combined.csv'
    sep = ',' if nodes else ';'

    # Iterate over all cities and load data for each city
    for city in directories:
        df = pd.read_csv(join(path, city, file), sep=sep)
        df['city'] = city
        data.append(df)
    # Concatenate all data together
    return pd.concat(data)



def create_city_graph(city_name:str, directory:str) -> nx.MultiDiGraph:
    """
    Create the graph of the city based on the information
    :param city_name: the name of the city we are loading
    :param directory: the path in which we load data 
    :return: the networkX graph object
    """
    # Load node and edge dataframe
    edge_list_df = pd.read_csv(join(directory, city_name, "network_combined.csv"), sep=";").drop(["route_I_counts"],axis=1).rename({"from_stop_I":"source","to_stop_I":"target"},axis=1)
    node_info_df = pd.read_csv(join(directory, city_name, "network_nodes_labeled.csv"), sep=",").drop(["lat","lon"],axis=1)

    # Add edges from the edge list DataFrame to the graph with attributes
    graph = nx.from_pandas_edgelist(edge_list_df, 'source', 'target', True,create_using=nx.MultiDiGraph)

    # Add node information from the node info DataFrame to the graph
    node_info = node_info_df.set_index('stop_I')
    # Set node attributes in the graph
    nx.set_node_attributes(graph, node_info.to_dict("index"))
    return  graph, node_info

def get_all_city_graph(directory: str)-> Tuple[List[nx.MultiDiGraph], List[pd.DataFrame],List[str]]:
    """
    Get all the cities graph 
    :param directory: the path in which we load data 
    :return: a tuple containing
        - a list of all cities graph
        - the nodes information with all node pandas datagrame
        - list of all cities directory
    """
    # Initialize list
    all_graphs = []
    all_nodes = []
    directories = get_all_cities_in_directory(directory)

    # Iterate over all cities 
    for city in directories:
        # Create the city graph using the function
        graph, nodes = create_city_graph(city_name=city, directory=directory)
        # add the city name to the nodes informations
        nodes["city"] = nodes.apply(lambda x: city,axis=1)
        # Add the graph and dataframe to the lists
        all_graphs.append(graph)
        all_nodes.append(nodes)

    return all_graphs, all_nodes,directories

def add_attribute_to_name(df, attribute_name, fct,graph):
    """
    Add the computed attribute to the dataframe
    :param df: the dataframe 
    :param attribute_name: the name of the attributes we need 
    :param fct: the function to use to compute the properties
    :param graph: the graph in which we want to compute the properties 
    """
    if "degree" in attribute_name:
        # For the degree distribution, we compute the property and convert it to a dictionnary
        if "in" in attribute_name:
            dico_to_value = {k:v for (k,v) in graph.in_degree(list(graph.nodes()))}
        elif "out" in attribute_name:
            dico_to_value = {k:v for (k,v) in graph.out_degree(list(graph.nodes()))}
    else:
        # All the other properties (except the degree) returns a dictionnarywith node as key and the property as value
        dico_to_value = fct(graph)
    # Add the property to the dataframe
    df[attribute_name] = df.apply(lambda x: dico_to_value[x.name],axis=1)
    return df