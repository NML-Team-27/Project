import numpy as np
import pandas as pd
import networkx as nx
from os.path import isdir, join
from os import listdir
from pandera.typing import DataFrame


def gather_all_data(path: str, nodes: bool = True) -> DataFrame:
    if(isdir(path)):
        directorys = sorted([d for d in listdir(path) if isdir(join(path, d))])
    
    data = []
    file = 'network_nodes_labeled.csv' if nodes else 'network_combined.csv'
    sep = ',' if nodes else ';'

    for city in directorys:
        df = pd.read_csv(join(path, city, file), sep=sep)
        df['city'] = city

        data.append(df)

    return pd.concat(data)