# %%

import time
import os
import ast
import numpy as np
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

min_cluster_size = 1000
work_dir = "/p/lustre1/li85/deep1b"

# %% Read our hdbscan output

ygm_dir = Path(f"{work_dir}/hdbscan_test/clams_test/cluster_data/cluster_data-m{min_cluster_size}-2nodes-v10")

files = [str(f.resolve()) for f in ygm_dir.glob('*.csv')]
dfs = []
for f in files:
    df = pd.read_csv(f,dtype={"stability": float, "stability_traversing_up": float, "non_chain_child1": str, "non_chain_child2": str})
    # df = df[df["valid_cluster"] == 1]
    # df = df[df["size"] >= min_cluster_size]
    dfs.append(df)

ygm_clustering = pd.concat(dfs, ignore_index=True)
del dfs

# %%
ygm_clustering = ygm_clustering.sort_values("birth_distance")
# ygm_clustering = ygm_clustering.sort_values("stability_traversing_up")
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

# %% Read serial hdbscan output

print("Serial clustering")

serial_file = f"{work_dir}/hdbscan_test/serial_hdbscan/cluster_data-m{min_cluster_size}-v2.txt"
column_names = ["cluster_id", "size", "birth_distance", "stability_traversing_up", "child1", "child2", "selected", "final_cluster_id", "split_edge_id"]
serial_clustering = pd.read_csv(serial_file, header=0, names=column_names)

serial_clustering = serial_clustering.sort_values("birth_distance")
serial_clustering.reset_index(drop=True, inplace=True)

serial_clustering["stability_traversing_up"] = serial_clustering["stability_traversing_up"].astype(float)
serial_clustering["stability_traversing_up"] = serial_clustering["stability_traversing_up"].round(5)

serial_leaf_clusters = serial_clustering[serial_clustering["child1"] == 4294967295]
serial_leaf_clusters = serial_leaf_clusters[serial_leaf_clusters["child2"] == 4294967295]
serial_leaf_clusters = serial_leaf_clusters.sort_values("birth_distance")

print("Number of clusters: ", len(serial_clustering))
print("Number of selected clusters:", len(serial_clustering[serial_clustering["selected"] == 1]))
print("Number of leaf clusters: ", len(serial_leaf_clusters))
print("Number of selected leaf clusters:", len(serial_leaf_clusters[serial_leaf_clusters["selected"] == 1]))
print("Number of clusters with birth distance 0: ", len(serial_clustering[serial_clustering["birth_distance"] == 0]))

# %%
leaf_df = pd.merge(serial_leaf_clusters, ygm_leaf_clusters, on=["size", "birth_distance"])
leaf_compare_df = pd.merge(serial_leaf_clusters, ygm_leaf_clusters, on=["size", "birth_distance", "selected"])

problem_leaf_clusters = leaf_df[~leaf_df["cluster_name"].isin(leaf_compare_df["cluster_name"])]

# %%

compare_df = pd.merge(serial_clustering, ygm_clustering, on=["size", "birth_distance"])
problem_clusters = ygm_clustering[~ygm_clustering["cluster_name"].isin(compare_df["cluster_name"])]
problem_clusters2 = serial_clustering[~serial_clustering["cluster_id"].isin(compare_df["cluster_id_x"])]
problem_clusters = problem_clusters.sort_values("size")
problem_clusters2 = problem_clusters2.sort_values("size")

size_issue_df = pd.merge(problem_clusters2, problem_clusters, on=["split_edge_id"])
size_issue_df = size_issue_df.sort_values("split_edge_id")
size_issue_df["size_diff"] = size_issue_df["size_x"] - size_issue_df["size_y"]

test_df = compare_df[compare_df["stability_traversing_up_x"] != compare_df["stability_traversing_up_y"]]
test_df = test_df[["cluster_id_x", "cluster_name", "size", "stability_traversing_up_x", "stability_traversing_up_y"]]
test_df = test_df.sort_values("size")

# test = pd.merge(ygm_clustering, serial_clustering, on=["split_edge_id"])
# check_serial = serial_clustering[~serial_clustering["cluster_name"].isin(test_df["split_edge_id"])]

# %% Check one problem leaf cluster

# cluster_name = "((991586135, 6), 1063592)"
# cluster_name = "((999984804, 6), 1766493)"
# df = compare_df[compare_df["cluster_name"] == cluster_name]
# df = pd.DataFrame()
# for i in range(20000):
#     row = compare_df[compare_df["cluster_name"] == cluster_name]
#     if len(row) > 0:
#         cluster_name = row.iloc[0]["parent"]
#         df = pd.concat([df, row])
#     else:
#         break

#     if row.iloc[0]["selected_y"] == 1:
#         break

# %% Check root chain size issue

correct_num_points_added = []
for i in range(len(size_issue_df)):
    row = size_issue_df.iloc[i]
    size = row["size_x"] 
    child_size1 = serial_clustering[serial_clustering["cluster_id"] == row["child1_x"]].iloc[0]["size"] 
    child_size2 = serial_clustering[serial_clustering["cluster_id"] == row["child2_x"]].iloc[0]["size"]
    num_points_added = size - child_size1 - child_size2
    correct_num_points_added.append(num_points_added)
size_issue_df["correct_num_points_added"] = correct_num_points_added

problem_merges_df = size_issue_df[size_issue_df["num_points_added"] != size_issue_df["correct_num_points_added"]]

# %%

test_df = pd.merge(problem_clusters2, problem_clusters, on=["split_edge_id"])
test_df = test_df.sort_values("size_y")
diff = test_df["size_y"] - test_df["size_x"]
diff.value_counts()

# %%
parsed = ygm_clustering["cluster_name"].map(ast.literal_eval)
edge_ids = parsed.map(
    lambda cluster_name: cluster_name[1] if cluster_name[0] == (0, 11) else None
)
smallest_edge_ids = edge_ids.dropna().nsmallest(10)

bottom_cluster_names = [f"((0, 11), {int(edge_id)})" for edge_id in smallest_edge_ids]
bottom_cluster_df = ygm_clustering[ygm_clustering["cluster_name"].isin(bottom_cluster_names)]
bottom_cluster_df = bottom_cluster_df.sort_values("size")
bottom_cluster_df
# print(serial_clustering[serial_clustering["size"] == bottom_cluster_df.iloc[0]["size"]])

# %%
parsed = problem_clusters["cluster_name"].map(ast.literal_eval)
edge_ids = parsed.map(
    lambda cluster_name: cluster_name[1] if cluster_name[0] == (0, 11) else None
)
smallest_edge_id = edge_ids.dropna().min()

bottom_problem_cluster_name = f"((0, 11), {int(smallest_edge_id)})"
bottom_cluster_df = ygm_clustering[ygm_clustering["cluster_name"] == bottom_problem_cluster_name]
print(bottom_cluster_df)
print(serial_clustering[serial_clustering["birth_distance"] == bottom_cluster_df.iloc[0]["birth_distance"]])