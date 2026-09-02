#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=clmshls
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 8
#SBATCH --time=00:05:00
#SBATCH --partition=pdebug
#SBATCH --exclusive

OUT_DIR="/p/lustre1/li85/oas_clustering/test"
echo "Command: mkdir -p $OUT_DIR"
mkdir -p $OUT_DIR

echo "Running DNND with distance on ANARCI sequences"
# echo "Command: mkdir -p /p/lustre1/li85/oas_clustering/test/tmp/knng"
# mkdir -p /p/lustre1/li85/oas_clustering/test/tmp/knng

BUILD_DIR="/usr/workspace/li85/hpc_clustering/ClaMS/build"
DNND_EXE="${BUILD_DIR}/custom_distance_example/build_knng_neo_anarci_dist"

INPUT_POINT_PATH="/p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/50germlines/100-150seqs/no_duplicates_anarci/anarci_numberings.txt"

# cmd="$DNND_EXE -k 20 -o $OUT_DIR/tmp/knng -p cdrs -f substitution -a 1.0 -M ${BUILD_DIR}/../custom_distance_example/BLOSUM62.txt -v ${INPUT_POINT_PATH}"
cmd="$DNND_EXE -k 20 -o $OUT_DIR/tmp/knng -p cdrs -f identity -H -v ${INPUT_POINT_PATH}"
echo "Command: srun $cmd"
srun $cmd

if [ $? -ne 0 ]; then
  echo "Error executing command."
  exit 1
fi

# echo "Generate CC bridge edges"
# echo "Command: srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/protein-clust/build/src/connect_ccs_random_ygm -i /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cc_bridge_edges"
# srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/protein-clust/build/src/connect_ccs_random_ygm -i /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cc_bridge_edges

# if [ $? -ne 0 ]; then
#   echo "Error executing command."
#   exit 1
# fi

# KNNG_DIR="/p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng"
# BRIDGE_EDGE_FILE="/p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cc_bridge_edges/cc_connect_edges.txt"
# if [ -e ${BRIDGE_EDGE_FILE} ]; then
# echo "Merge bridge edges with original kNNG"
# echo "Command: srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/protein-clust/build/src/merge_connect_edges_anarci_substitution -g /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng/knng -c /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cc_bridge_edges/cc_connect_edges.txt -a 1.0 -A 1 -M /usr/workspace/li85/hpc_clustering/protein-clust/build/../substitution_matrix/BLOSUM62.txt -G /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng_merged /p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/50germlines/100-150seqs/no_duplicates_anarci/anarci_numberings.txt "
# srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/protein-clust/build/src/merge_connect_edges_anarci_substitution -g /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng/knng -c /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cc_bridge_edges/cc_connect_edges.txt -a 1.0 -A 1 -M /usr/workspace/li85/hpc_clustering/protein-clust/build/../substitution_matrix/BLOSUM62.txt -G /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng_merged /p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/50germlines/100-150seqs/no_duplicates_anarci/anarci_numberings.txt 

# if [ $? -ne 0 ]; then
#   echo "Error executing command."
#   exit 1
# fi

# KNNG_DIR="/p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/knng_merged"
# fi

# echo "Running AMST"
# echo "Command: mkdir -p /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst"
# mkdir -p /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst
# echo "Command: srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/ClaMS/build/src/mst/build_amst -e 1.1 -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst/mst.txt -i ${KNNG_DIR}"
# srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/ClaMS/build/src/mst/build_amst -e 1.1 -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst/mst.txt -i ${KNNG_DIR}

# if [ $? -ne 0 ]; then
#   echo "Error executing command."
#   exit 1
# fi

# echo "Running HPC Clustering"
# MIN_CLUSTER_SIZE=2
# echo "Min clustering size ${MIN_CLUSTER_SIZE}"
# echo "Command: /usr/workspace/li85/hpc_clustering/ClaMS/build/src/clustering/run_hdbscan_clustering -i /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst/mst.txt  -m ${MIN_CLUSTER_SIZE} -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt"
# /usr/workspace/li85/hpc_clustering/ClaMS/build/src/clustering/run_hdbscan_clustering -i /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst/mst.txt  -m ${MIN_CLUSTER_SIZE} -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt

# if [ $? -ne 0 ]; then
#   echo "Error executing command."
#   exit 1
# fi

# echo "Evaluating Clustering using YGM"
# echo "Command: srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/clams-cc/build/src/clustering_metrics_no_mi -v -p -g /p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/50germlines/100-150seqs/no_duplicates_anarci/labels.txt /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt"
# srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/clams-cc/build/src/clustering_metrics_no_mi -v -p -g /p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/50germlines/100-150seqs/no_duplicates_anarci/labels.txt /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt

# if [ $? -ne 0 ]; then
#   echo "Error executing command."
#   exit 1
# fi

# echo "" 
# MIN_CLUSTER_SIZE=5
# echo "Min clustering size ${MIN_CLUSTER_SIZE}"
# echo "Command: /usr/workspace/li85/hpc_clustering/ClaMS/build/src/clustering/run_hdbscan_clustering -i /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst/mst.txt  -m ${MIN_CLUSTER_SIZE} -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt"
# /usr/workspace/li85/hpc_clustering/ClaMS/build/src/clustering/run_hdbscan_clustering -i /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/mst/mst.txt  -m ${MIN_CLUSTER_SIZE} -o /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt

# if [ $? -ne 0 ]; then
#   echo "Error executing command."
#   exit 1
# fi

# echo "Evaluating Clustering using YGM"
# echo "Command: srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/clams-cc/build/src/clustering_metrics_no_mi -v -p -g /p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/50germlines/100-150seqs/no_duplicates_anarci/labels.txt /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt"
# srun --ntasks-per-node=112 /usr/workspace/li85/hpc_clustering/clams-cc/build/src/clustering_metrics_no_mi -v -p -g /p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/50germlines/100-150seqs/no_duplicates_anarci/labels.txt /p/lustre1/li85/oas_clustering/50germlines/100-150seqs/no_duplicates_anarci/neo_dnnd_substitution/cdrs/BLOSUM62/inverse-power1.0/amst_err1.1/k20-s1/tmp/cluster_labels-m${MIN_CLUSTER_SIZE}.txt

# if [ $? -ne 0 ]; then
#   echo "Error executing command."
#   exit 1
# fi

# echo "" 
