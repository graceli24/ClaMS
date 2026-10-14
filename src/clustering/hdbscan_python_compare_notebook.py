# %%

import time
import os
import ast
import numpy as np
import scipy as sp
import pandas as pd
import sklearn.metrics as skmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.patches as patches
import matplotlib.colors as mpl_colors
import hdbscan
import networkx as nx
from pathlib import Path

from hdbscan._hdbscan_tree import compute_stability

write_outputs = False
min_cluster_size = 200

work_dir = "/p/lustre1/li85/clams_test/mnist_debug"
output_dir = f"{work_dir}/hdbscan_test/python_hdbscan"
mst_file = f"{work_dir}/mst_edges.txt"

if not os.path.exists(output_dir):
     os.makedirs(output_dir)

# %% Read the tree

edge_df = pd.read_csv(mst_file, header=None)
edge_df.columns = ["edge_id", "node1", "node2", "dist"]
edge_df = edge_df.drop(columns='edge_id')
n = len(edge_df) + 1

# nx_edge_df = edge_df.copy()
# nx_edge_df.columns = ["source", "target", "dist"]
# # T = nx.from_pandas_edgelist(nx_edge_df)
# del nx_edge_df

dist_to_edge_id = {float(edge_df["dist"].iloc[i]): i for i in range(len(edge_df))}

# %% Run HDBSCAN

# Convert the MST to a distance matrix to input into HDBSCAN
data = []
rows = []
cols = []
max_dist = 0
for i in range(len(edge_df)):
    row = edge_df.iloc[i]
    node1, node2 = int(row["node1"]), int(row["node2"])
    data.append(row["dist"])
    data.append(row["dist"])
    rows.append(node1)
    rows.append(node2)
    cols.append(node2)
    cols.append(node1)
    if row["dist"] > max_dist:
        max_dist = row["dist"]
dist_matrix = sp.sparse.coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(n, n))
print("The max distance in the MST is:", max_dist)

clusterer = hdbscan.HDBSCAN(metric='precomputed',
                            min_cluster_size = min_cluster_size,
                            min_samples = 1)
clusterer.fit(dist_matrix)
single_linkage_df = clusterer.single_linkage_tree_.to_pandas()

num_clusters = clusterer.labels_.max()  + 1
print("Number of clusters = ", num_clusters)

# %% Write the point labels

label_df = pd.DataFrame({"point_id": list(range(n)), 
                        "cluster_label": clusterer.labels_})

if write_outputs:
    output_file = f"{output_dir}/cluster_labels-m{min_cluster_size}.txt"
    label_df.to_csv(f"{output_dir}/cluster_labels-m{min_cluster_size}.txt", 
                    sep='\t', index=False)
    print(output_file)

# %% Check the stabilities

condensed_tree_df = clusterer.condensed_tree_.to_pandas()
condensed_tree_df = condensed_tree_df[condensed_tree_df["child_size"] > 1]

stability_dict = compute_stability(clusterer.condensed_tree_._raw_tree)

_temp = {"child":[], "stability":[]}
for key, value in stability_dict.items():
    _temp["child"].append(int(key))
    _temp["stability"].append(value)
stability_df = pd.DataFrame(_temp)
del _temp

cluster_df = pd.merge(condensed_tree_df, stability_df, on="child")
cluster_df["birth_distance"] = 1 / cluster_df["lambda_val"]
# cluster_df["birth_distance"] = cluster_df["birth_distance"].astype(int)
cluster_df["stability"] = cluster_df["stability"].round(5)

cluster_df = cluster_df.sort_values("birth_distance")
# cluster_df = cluster_df.sort_values("stability")
cluster_df.reset_index(drop=True, inplace=True)

if write_outputs:
    output_file = f"{output_dir}/cluster_info-m{min_cluster_size}.csv"
    cluster_df.to_csv(output_file, index=False)
    print(output_file)

child_cluster_ids = set(cluster_df.child)
parent_cluster_ids = set(cluster_df.parent)
leaf_cluster_ids = child_cluster_ids - parent_cluster_ids
leaf_clusters = cluster_df[cluster_df["child"].isin(leaf_cluster_ids)]
leaf_clusters = leaf_clusters.sort_values("birth_distance")

# Match clusters to labels

cluster_df["cluster_id"] = [-1]*len(cluster_df)

selected_cluster_ids = clusterer.condensed_tree_._select_clusters()
selected_cluster_ids = [int(x) for x in selected_cluster_ids]
# internal_to_label = {cid: i for i, cid in enumerate(selected_cluster_ids)}

for cluster_id in range(num_clusters):
    cluster_df.loc[cluster_df["child"] == selected_cluster_ids[cluster_id], "cluster_id"] = cluster_id

cluster_df = cluster_df.rename(columns={"child": "cluster_name", "child_size": "size"})

# %% Read our hdbscan output

n_nodes = 1
n_tasks = 112
ygm_dir = Path(f"{work_dir}/hdbscan_test/distributed_hdbscan/cluster_data/{n_nodes}nodes-{n_tasks}tasks/cluster_data-m{min_cluster_size}")

files = [str(f.resolve()) for f in ygm_dir.glob('*.csv')]
dfs = []
for f in files:
    df = pd.read_csv(f,dtype={"stability": float, "stability_traversing_up": float, "non_chain_child1": str, "non_chain_child2": str})
    # df = df[df["valid_cluster"] == 1]
    # df = df[df["size"] >= min_cluster_size]
    dfs.append(df)

ygm_clustering = pd.concat(dfs, ignore_index=True)
del dfs

ygm_clustering = ygm_clustering.sort_values("birth_distance")# ygm_clustering = ygm_clustering.sort_values("stability_traversing_up")
ygm_clustering.reset_index(drop=True, inplace=True)
ygm_clustering["stability"] = ygm_clustering["stability"].round(5)
ygm_clustering["stability_traversing_up"] = ygm_clustering["stability_traversing_up"].round(5)

ygm_clustering = ygm_clustering[ygm_clustering["size"] >= min_cluster_size]
ygm_clustering["split_edge_id"] = ygm_clustering["cluster_name"].apply(lambda name: ast.literal_eval(name)[1])

ygm_leaf_clusters = ygm_clustering[ygm_clustering["child1"] == "((0, 0), 0)"]
ygm_leaf_clusters = ygm_leaf_clusters[ygm_leaf_clusters["child2"] == "((0, 0), 0)"]
ygm_leaf_clusters = ygm_leaf_clusters.sort_values("birth_distance")

print("Number of clusters: ", len(ygm_clustering))
print("Number of selected clusters:", len(ygm_clustering[ygm_clustering["selected"] == 1]))
print("Number of leaf clusters: ", len(ygm_leaf_clusters))
print("Number of selected leaf clusters:", len(ygm_leaf_clusters[ygm_leaf_clusters["selected"] == 1]))
print("Number of clusters with birth distance 0: ", len(ygm_clustering[ygm_clustering["birth_distance"] == 0]))

# compare_df = pd.merge(cluster_df, ygm_clustering, on=["size", "birth_distance"])
compare_df = pd.merge(cluster_df, ygm_clustering, on=["size"])

# # %% Read serial hdbscan output

# print("Serial clustering")

# serial_file = f"{work_dir}/hdbscan_test/serial_hdbscan/cluster_data-m{min_cluster_size}.txt"
# column_names = ["cluster_id", "size", "birth_distance", "stability_traversing_up", "child1", "child2", "selected", "final_cluster_id", "split_edge_id"]
# serial_clustering = pd.read_csv(serial_file, header=0, names=column_names)

# serial_clustering = serial_clustering.sort_values("birth_distance")
# serial_clustering.reset_index(drop=True, inplace=True)

# serial_clustering["stability_traversing_up"] = serial_clustering["stability_traversing_up"].astype(float)
# serial_clustering["stability_traversing_up"] = serial_clustering["stability_traversing_up"].round(5)

# serial_leaf_clusters = serial_clustering[serial_clustering["child1"] == 4294967295]
# serial_leaf_clusters = serial_leaf_clusters[serial_leaf_clusters["child2"] == 4294967295]
# serial_leaf_clusters = serial_leaf_clusters.sort_values("birth_distance")

# print("Number of clusters: ", len(serial_clustering))
# print("Number of selected clusters:", len(serial_clustering[serial_clustering["selected"] == 1]))
# print("Number of leaf clusters: ", len(serial_leaf_clusters))
# print("Number of selected leaf clusters:", len(serial_leaf_clusters[serial_leaf_clusters["selected"] == 1]))
# print("Number of clusters with birth distance 0: ", len(serial_clustering[serial_clustering["birth_distance"] == 0]))