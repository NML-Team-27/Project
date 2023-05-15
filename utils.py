import numpy as np
import pandas as pd
import networkx as nx
from os.path import isdir, join
from os import listdir
import networkx as nx
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(rc={'figure.max_open_warning': 0})

def get_all_cities_in_directory(directory:str)->List[str]:
    if(isdir(directory)):
        return sorted([d for d in listdir(directory) if isdir(join(directory, d))])
    return None


def gather_all_data(path: str, nodes: bool = True):
    
    directories = get_all_cities_in_directory(path)
    data = []
    file = 'network_nodes_labeled.csv' if nodes else 'network_combined.csv'
    sep = ',' if nodes else ';'

    for city in directories:
        df = pd.read_csv(join(path, city, file), sep=sep)
        df['city'] = city

        data.append(df)

    return pd.concat(data)



def create_city_graph(city_name:str, directory:str) -> nx.Graph:
    
    import pandas as pd
    import networkx as nx
    edge_list_df = pd.read_csv(join(directory, city_name, "network_combined.csv"), sep=";").drop(["route_I_counts","d"],axis=1).rename({"from_stop_I":"source","to_stop_I":"target"},axis=1)
    
    node_info_df = pd.read_csv(join(directory, city_name, "network_nodes_labeled.csv"), sep=",").drop(["lat","lon"],axis=1)


    # Add edges from the edge list DataFrame to the graph with attributes
    graph = nx.from_pandas_edgelist(edge_list_df, 'source', 'target', True)

    #print(graph.get_edge_data(10924,10920))

    # Add node information from the node info DataFrame to the graph
    node_info = node_info_df.set_index('stop_I')
    nx.set_node_attributes(graph, node_info.to_dict("index"))
    return  graph, node_info

def get_all_city_graph(directory: str)-> Dict[str, nx.Graph]:
    all_graphs = []
    all_nodes = []
    directories = get_all_cities_in_directory(directory)

    for city in directories:
        graph, nodes = create_city_graph(city_name=city, directory=directory)
        nodes["city"] = nodes.apply(lambda x: city,axis=1)
        all_graphs.append(graph)
        all_nodes.append(nodes)

    return all_graphs, all_nodes,directories

def add_attribute_to_name(df, attribute_name, fct,graph):
    dico_to_value = fct(graph)
    df[attribute_name] = df.apply(lambda x: dico_to_value[x.name],axis=1)
    return df




    
