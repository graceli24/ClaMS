source /usr/workspace/li85/python_tools/grace_env/bin/activate

NUM_NODES=4
NUM_TASKS_PER_NODE=112

BUILD_DIR="/usr/workspace/li85/hpc_clustering/ClaMS/build"

# INPUT_POINT_PATH="/p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/no_max_germlines/100-150seqs/no_duplicates_anarci/anarci_numberings.txt"
# OUT_DIR="/p/lustre3/li85/clams_test/anarci_clustering_outputs/no_max_germ-100-150seqs"

INPUT_POINT_PATH="/p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/no_max_germlines/100-1000seqs/no_duplicates_anarci/anarci_numberings"
OUT_DIR="/p/lustre3/li85/clams_test/anarci_clustering_outputs/no_max_germ-100-1kseqs"

# INPUT_POINT_PATH="/p/lustre1/li85/oas_data/oas_labeled_with_germline_anarci/no_max_germlines/100-1000seqs/no_duplicates_anarci/anarci_numberings"
# OUT_DIR="/p/lustre3/li85/clams/oas"

DIST_TYPE="substitution"
k=10
AMST_ERR=1.1
MIN_CLUSTER_SIZE=5

identity_distance_cmd="-d identity --use_hamming_dist"
substitution_distance_cmd="-d substitution --inverse_power 1.0"

JOB_NAME="${DIST_TYPE}/k${k}-amst_err${AMST_ERR}"

echo "Command: mkdir -p $OUT_DIR"
mkdir -p $OUT_DIR
echo "Command: rm -r ${OUT_DIR}/${JOB_NAME}"
rm -r ${OUT_DIR}/${JOB_NAME}

options="-b ${BUILD_DIR} \
-p ${INPUT_POINT_PATH}
-o ${OUT_DIR}
-j ${JOB_NAME}
${identity_distance_cmd}
--region_names full
-k ${k}
-R
-a ${AMST_ERR}
-m ${MIN_CLUSTER_SIZE}
--distributed_hdbscan
-N ${NUM_NODES}
-T ${NUM_TASKS_PER_NODE}
-v
-S
"

echo "python3 run_anarci_dist_pipeline.py ${options} -B \"-- --partition=pdebug --time=00:30:00 --account=clmshls --exclusive\""
python3 run_anarci_dist_pipeline.py ${options} -B "-- --partition=pdebug --time=00:30:00 --account=clmshls --exclusive"
