# Project

In this project, we aims at building models that can predict if one node is part of the city center or not. Our dataset comes from this link and we extract all the information from this dataset only.

The repository is structured as follows 

```
├── README.md
├── centers.json						            # City center coordinates 
├── data
│   ├── all_metrics.csv					        # CSV file containing all the graph properties for all the nodes in all cities
│   ├── one folder per city
│   │   ├── adj_mat.edg					        # The adjacency matrix
│   │   ├── license.txt					        # license file for the city
│   │   ├── network_combined.csv		    # Network edges with attributes from the original dataset
│   │   ├── network_nodes.csv			      # Stop names and coordinates
│   │   ├── network_nodes_labeled.csv	  # Stop names, coordinates and labels
│   │   ├── results
│   │   │   ├── node2vec_embeddings.npy	# Node2Vec embeddings used in the best Node2Vec model
│   │   │   └── results.json			      # The experiment results for the current city
│   │   └── stats.csv					          # Statistics about data collection in the city (from the dataset)
│   ├── handcrafted_features.csv		    # All the features (both from the dataset and graph properties) that will be used during the training of the baseline models
├── eda_distributions.ipynb				      # The notebook containing all the data exploration
├── GNN.ipynb							              # notebook with everything related to GNN
├── gnn.py								              # Definintion of GNN mdoel
├── handcrafted_features.ipynb			    # Merged all data in a single dataframe
├── labelling.ipynb						          # Notebook to label our data
├── node2vec.ipynb						          # Notebook with everything related to the baseline models (handcrafted and Node2vec)
├── radii.json							            # The radii used to determine the city center stops (distance from the coordinates of the center) 
├── requirements.txt					          # File containing all the libraries to run our code
├── save_gnns.json						          # Cross validation for the parameters of the GNN
├── save_p_q.json						            # Cross validation for the parameters  p and q for Node2Vec parameters. 
├── training_baseline.py				        # All the functions to train the baseline
├── training_gnn.py						          # Util file with all the code to train the GNNs
└── utils.py							              # Util script to gather data and compute properties value

```

### Data
To get the data, everything is stored on Git-lfs. You can run the following command to get the data

```
git-lfs pull
```

### Experiments
Below we recally the best parameters that we obtained in our experiments. 

- Node2Vec: p=5.0, q=0.1, num_walks=100, hidden-dim = 256.
- Conv GNN: nb_conv = 3, dim = 32. 
- GAT: nb_conv = 3, dim = 32, heads = 8

The detailled results can be found in [this file for baseline](save_p_q.json) and [this file for GNN](save_gnns.json)

To run experiments, you can use the following two files: 

- `cities_loop` in [training_baseline](training_baseline.py).
- `cities_loop_gnn` in [training_gnn](training_gnn.py).


### Report
The report can be found in the `report.pdf` file.

### File
The data exploration and feature extraction can found in [labelling.ipynb](labelling.ipynb), [handcrafted_features.ipynb](handcrafted_features.ipynb) and   [eda_distributions.ipynb](eda_distributions.ipynb).

The training code is in [GNN.ipynb](GNN.ipynb) and [node2vec.ipynb](node2vec.ipynb). 
