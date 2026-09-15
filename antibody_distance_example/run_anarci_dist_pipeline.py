# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.

import os
import sys
import argparse
import time
from datetime import datetime

sys.path.append("./../script/benchmark")
from bench_utilities import *



def parse_options():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate a batch shell script'
                                                 'for ClaMS HPC clustering with ANARCI distance.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cwd = os.getcwd() #Assumes we are running from the build directory

    # Build directory to run things from
    parser.add_argument('-b', '--build_dir', type=str, 
                        default=f'{cwd}',
                        help='Path to directory to run everything from. '
                             'This should be set to the path to the build directory. '
                             'Default is the current working directory.')

    # For input point (feature) data
    parser.add_argument('-p', '--point_path',
                        required=True,
                        help='Path to the point data file')

    # For distance function
    parser.add_argument('-d', '--distance_name', type=str, default='identity',
                        help='Distance name for ANARCI sequences. Options are identity or substitution.')
    parser.add_argument('--region_names', type=str, default='full',
                        help = 'Comma separated list of antibody sequence region names or positions '
                               'to consider in ANARCI distance calculation.')
    parser.add_argument('--substitution_matrix', type=str, default=None,
                        help = 'Path to an amino acid substitution matrix to use for substitution '
                                'distance calculation.')
    parser.add_argument('--inverse_power', type=float, default=1.0,
                        help = 'For substitution distance, similarity s is converted to '
                               'distance = s^(-inverse_power).')
    parser.add_argument('--dont_include_ends', action='store_true',
                        help = 'Don\'t include ends (start/end of ANARCI positions before there are amino acids).'
                               ' Only applies if using identity distance.')
    parser.add_argument('--use_hamming_dist', action='store_true',
                        help = 'Use unnormalized Hamming distance instead of identity distance. '
                               'Only applies if using identity instead of substitution distance.')


    # For NN-Descent (DNND)
    parser.add_argument('-k', '--nng_k', type=int, default=50,
                        help='#of neighbors to have when constructing'
                             'a KNNG value.')
    parser.add_argument('--neodnnd_threads', type=int, default=2,
                        help='Number of threads to use for NEO-DNND. Ignored if --neodnnd is not specified.')
    parser.add_argument('--nng_r', type=float, default=0.5,
                        help='The r (sampling) parameter for KNNG construction.')
    parser.add_argument('--nng_delta', type=float, default=0.0001,
                        help='The delta parameter (termination condition) for KNNG construction.')

    parser.add_argument('-G', '--input_dnnd_ds_path', default='',
                        help='Path to an already constructed DNND PM datastore. If specified, skip the DNND step.')

    # KNNG backup
    parser.add_argument('-R', '--backup_knng', action='store_true',
                        help='Backup the KNNG datastore after DNND step.')

    # Approximate Minimum Spanning Tree (AMST)
    parser.add_argument('-a', '--amst_approx_bound', default='1.1',
                        dest='amst_approx_bound_range', action='store', type=str,
                        help='Approximation bound for AMST. Single value or comma separated list of values (e.g., 1.1,1.2,1.4).')

    # For HDBSCAN
    # Min cluster size, conmma separated list of min cluster sizes, or range of min cluster sizes
    parser.add_argument('-m', '--min_cluster_size', default='5',
                        dest='min_cluster_size_range',
                        action='store', type=str,
                        help='Minimum cluster size. Single value or, comma separated list of values or range of values (e.g., 5,10,15 or 5-15)')
    # Min samples
    parser.add_argument('-s', '--min_samples', type=int, default=-1,
                        dest='min_samples',
                        help='min_sample value for calculating core distance. If -1 is given, core distance is not calculated.')
    # Use distributed HDBSCAN
    parser.add_argument('--distributed_hdbscan', action='store_true',
                        help='Use distributed HDBSCAN (run_distributed_hdbscan_clustering) instead of serial version (run_hdbscan_clustering).')

    # For evaluation
    parser.add_argument('-g', '--ground_truth_path', default=None,
                        help='Path to the ground truth data file for evaluating clustering quality.')
    # By default (with no -y flag), will keep using the python script for cluster evaluation,
    # Use the -y flag to use a YGM calculation instead
    parser.add_argument('-y', '--ygm_cluster_eval', action='store_true',
                        help='Use YGM cluster evaluation calculation.')

    # Batch Job configurations for DNND, AMST, Distributed HDBSCAN, and YGM partition comparison
    parser.add_argument('-N', '--num_nodes', type=int, default=1,
                        help='Number of nodes to use for running DNND and AMST')
    parser.add_argument('-T', '--num_tasks_per_node', type=int, default=32,
                        help='Number of tasks per node to use for running DNND and AMST')

    # Submit the job
    parser.add_argument('-S', '--submit_job', action='store_true',
                        help='Submit the batch job to the scheduler')
    parser.add_argument('-B', '--sbatch_opts', type=str,
                        default='-- --time=01:00:00 --account=clmshls',
                        help='Additional options to pass to sbatch (e.g., time limit and bank name) when submit a batch job. Must start with \'--\' to avoid a parse error, e.g., -B "-- --time=01:00:00 --account=clmshls"')

    # For output
    parser.add_argument('-o', '--output_root_dir',
                        default=f'{cwd}/anarci_clustering_outputs',
                        help='Path to the root of output directories. '
                             'Outputs will write to output_root_dir/job_name.')
    parser.add_argument('-j', '--job_name', type=str, default=None,
                        help='Job name. Outputs will write to output_root_dir/job_name. '
                             'If none provided, the current time will be used.')
    parser.add_argument('-w', '--work_dir', type=str, default=None,
                        help='Work directory for intermediate files.'
                            'This must be a location shared by all jobs and MPI processes.'
                            'If none provided, will use output_root_dir/job_name/temp.')

    # Executables
    parser.add_argument('-D', '--dnnd_exe', type=str, default=None,
                        help='Path to the DNND executable.')
    parser.add_argument('-M', '--mfc_exe', type=str, default=None,
                        help='Path to the MFC executable.')
    parser.add_argument('-A', '--amst_exe', type=str, default=None,
                        help='Path to the AMST executable.')
    parser.add_argument('-C', '--clustering_exe', type=str, default=None,
                        help='Path to the HPC clustering executable.')
    parser.add_argument('-E', '--evaluator_exe', type=str, default=None,
                        help='Path to the clustering evaluation executable.')

    # Etc
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose outputs.')

    options = parser.parse_args()

    # Default directories and executables if none provided
    if options.dnnd_exe is None:
        options.dnnd_exe = f'{options.build_dir}/antibody_distance_example/build_knng_neo_anarci_dist'
    if options.mfc_exe is None:
        options.mfc_exe = f'{options.build_dir}/antibody_distance_example/connect_mfc_anarci_dist'
    if options.amst_exe is None:
        options.amst_exe = f'{options.build_dir}/src/mst/build_amst'
    if options.clustering_exe is None:
        if options.distributed_hdbscan:
            options.clustering_exe = f'{options.build_dir}/src/clustering/run_distributed_hdbscan_clustering'
        else:
            options.clustering_exe = f'{options.build_dir}/src/clustering/run_hdbscan_clustering'
    if options.evaluator_exe is None:
        if options.ygm_cluster_eval:
            options.evaluator_exe = f'{options.build_dir}/tpls/clams-cc/build/src/clustering_metrics'
        else:
            options.evaluator_exe = f'{options.build_dir}/../script/benchmark/evaluate_clustering_quality.py'
    if options.substitution_matrix is None:
        options.substitution_matrix = f'{options.build_dir}/../antibody_distance_example/BLOSUM62.txt'

    return options


def generate_job_name():
    time.sleep(2)
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def add_clustering_evaluation(job_script, cluster_label_file, amst_ds_path,
                              ground_truth_path, evaluator,
                              ygm_cluster_eval,
                              num_tasks_per_node, verbose,
                              singleton_cluster_to_noise_points):
    job_script.write("echo\n")
    job_script.write("date\n")
    job_script.write(f"echo ================================\n")
    job_script.write(f"echo \"Evaluating Clustering\"\n")
    job_script.write(f"echo ================================\n")

    if ygm_cluster_eval:
        job_script.write("echo \"Evaluating Clustering using YGM\"\n")
        verbose_flag = " -v" if verbose else ""
        evaluation_command = (f"{evaluator} {verbose_flag} -p"
                              f"-g {ground_truth_path} "
                              f"{cluster_label_file}")
        add_srun_cmd(num_tasks_per_node, evaluation_command, job_script)
    else:
        job_script.write(
            "echo \"Evaluating Clustering using python script\"\n")
        evaluation_command = (f"python3 {evaluator} "
                              f"-c {cluster_label_file} "
                              f"-g {ground_truth_path}")
        if singleton_cluster_to_noise_points:
            evaluation_command += " -s"
        add_cmd(evaluation_command, job_script)


# Function to generate the batch script
def gen_anarci_dist_bench_script(job_name, job_dir, work_dir,
                                    num_nodes, num_tasks_per_node,
                                    point_path,
                                    distance_name, region_names,
                                    substitution_matrix, inverse_power,
                                    dont_include_ends, use_hamming_dist,
                                    dnnd_exe, nng_k,
                                    nng_r, nng_delta,
                                    backup_knng,
                                    mfc_exe, 
                                    amst_exe, amst_approx_bound_list,
                                    clustering_exe,
                                    min_cluster_size_set_cmnds,
                                    min_samples,
                                    distributed_hdbscan, 
                                    verbose=False,
                                    ground_truth_path=None,
                                    evaluator_exe=None,
                                    ygm_cluster_eval=False,
                                    input_dnnd_ds_path=''):


    create_dir(job_dir)
    job_script_path = f'{job_dir}/job.sh'

    # Open the output shell script file for writing
    with open(job_script_path, 'w') as job_script:

        # Set up the batch script header
        set_up_batch_header(job_script, job_name, job_dir, num_nodes)

        add_cmd(f'mkdir -p {work_dir}', job_script, True, False)

        # Run the DNND step
        job_script.write("echo\n")
        job_script.write("date\n")
        if len(input_dnnd_ds_path) == 0:
            job_script.write(f"echo ================================\n")
            job_script.write(f"echo \"Building KNNG\"\n")
            job_script.write(f"echo ================================\n")
            dnnd_ds_path = f"{work_dir}/knng_pm_datastore"
            dnnd_batch_size = 2 ** 25
            verbose_flag = '-v' if verbose else ''
            dnnd_command = f"{dnnd_exe} {verbose_flag} -k {nng_k} -r {nng_r} -d {nng_delta} -o {dnnd_ds_path} -b {dnnd_batch_size} -f {distance_name} -p {region_names}"
            if distance_name == "substitution":
                dnnd_command = f"{dnnd_command} -a {inverse_power} -M {substitution_matrix}"
            else:
                dont_use_ends_flag = '-x' if dont_include_ends else ''
                hamming_dist_flag = '-H' if use_hamming_dist else ''
                dnnd_command = f"{dnnd_command} {dont_use_ends_flag} {hamming_dist_flag}"
            if backup_knng:
                dnnd_ds_path_backup = f"{dnnd_ds_path}_backup"
                job_script.write(f"echo \"Backing up KNNG datastore to {dnnd_ds_path_backup}\"\n")
                dnnd_command = f"{dnnd_command} -B {dnnd_ds_path_backup}"
            dnnd_command = f"{dnnd_command} {point_path}"
            add_srun_cmd(num_tasks_per_node, dnnd_command, job_script)
            # if backup_knng:
            #     dnnd_ds_path_backup = f"{dnnd_ds_path}_backup"
            #     job_script.write(f"echo\n")
            #     job_script.write("date\n")
            #     job_script.write(f"echo \"Backing up KNNG datastore\"\n")
            #     backup_knng_command = f"cp -r {dnnd_ds_path} {dnnd_ds_path_backup}"
            #     add_cmd(backup_knng_command, job_script)
        else:
            job_script.write(
                f"Using existing DNND datastore at {input_dnnd_ds_path}\n")
            dnnd_ds_path = input_dnnd_ds_path

        # Connect the CCs
        job_script.write("echo\n")
        job_script.write("date\n")
        job_script.write(f"echo ================================\n")
        job_script.write(f"echo \"Running MFC\"\n")
        job_script.write(f"echo ================================\n")
        mfc_command = f"{mfc_exe} -d {dnnd_ds_path} -f {distance_name}"
        if distance_name == "substitution":
            mfc_command = f"{mfc_command} -a {inverse_power} -M {substitution_matrix}"
        else:
            dont_use_ends_flag = '-x' if dont_include_ends else ''
            hamming_dist_flag = '-H' if use_hamming_dist else ''
            mfc_command = f"{mfc_command} {dont_use_ends_flag} {hamming_dist_flag}"
        add_srun_cmd(num_tasks_per_node, mfc_command, job_script)

        try_no = 0
        for amst_approx_bound in amst_approx_bound_list:
            # Run the AMST step
            job_script.write("echo\n")
            job_script.write("date\n")
            job_script.write(f"echo ================================\n")
            job_script.write(f"echo \"Running AMST, approx bound = {amst_approx_bound}\"\n")
            job_script.write(f"echo ================================\n")
            amst_ds_path = f"{work_dir}/amst_pm_datastore_a{amst_approx_bound}"
            amst_command = f"{amst_exe} -d {dnnd_ds_path} -p {amst_ds_path} -e {amst_approx_bound}"
            add_srun_cmd(num_tasks_per_node, amst_command, job_script)

            # Run the HPC Clustering step
            for set_cmd in min_cluster_size_set_cmnds:
                # Set the min cluster size environment variable for this run
                add_cmd(set_cmd, job_script, False, False)

                job_script.write("echo\n")
                job_script.write("date\n")
                job_script.write(f"echo ================================\n")
                job_script.write(f"echo \"Running CLAMS-HDBSCAN, min cluster size = ${{MIN_CLUSTER_SIZE}}\"\n")
                job_script.write(f"echo ================================\n")

                job_script.write(
                    f"echo \"Min cluster size ${{MIN_CLUSTER_SIZE}}\"\n")
                if distributed_hdbscan:
                    cluster_label_file = f"{work_dir}/cluster_labels_a{amst_approx_bound}_m${{MIN_CLUSTER_SIZE}}/"
                    cluster_tree_file = f"{work_dir}/cluster_tree_a{amst_approx_bound}_m${{MIN_CLUSTER_SIZE}}/"
                    verbose_flag = '-v' if verbose else ''
                    hpc_clustering_command = (f"{clustering_exe} {verbose_flag} -i {amst_ds_path} -M "
                                            f" -m ${{MIN_CLUSTER_SIZE}} "
                                            f" -o {cluster_label_file} "
                                            f" -c {cluster_tree_file}")
                    add_srun_cmd(num_tasks_per_node, hpc_clustering_command, job_script)
                else:
                    cluster_label_file = f"{work_dir}/cluster_labels_a{amst_approx_bound}_m${{MIN_CLUSTER_SIZE}}.txt"
                    cluster_tree_file = f"{work_dir}/cluster_tree_a{amst_approx_bound}_m${{MIN_CLUSTER_SIZE}}.txt"
                    hpc_clustering_command = (f"{clustering_exe} -i {amst_ds_path} -M "
                                            f" -m ${{MIN_CLUSTER_SIZE}} "
                                            f" -o {cluster_label_file} "
                                            f" -c {cluster_tree_file} "
                                            f" -P ")
                    add_cmd(hpc_clustering_command, job_script)

                # Run the evaluation step
                if ground_truth_path is not None and ground_truth_path != '':

                    add_clustering_evaluation(job_script, cluster_label_file, 
                                                amst_ds_path,
                                                ground_truth_path, evaluator_exe,
                                                ygm_cluster_eval,
                                                num_tasks_per_node, verbose,
                                                singleton_cluster_to_noise_points=False)

                    # job_script.write("echo\n")
                    # job_script.write("date\n")
                    # job_script.write(f"echo ================================\n")
                    # job_script.write(f"echo \"Assign clusters to noise points\"\n")
                    # job_script.write(f"echo ================================\n")
                    # # Remove the .txt extension and add .noise_assigned.txt
                    # cluster_label_file_no_noise = f"{cluster_label_file[:-4]}.noise_assigned.txt"
                    # cluster_assign_command = (f"{noise_point_assigner_exe} -M "
                    #                           f"-m {amst_ds_path} "
                    #                           f"-c {cluster_label_file} "
                    #                           f"-o {cluster_label_file_no_noise}")
                    # add_cmd(cluster_assign_command, job_script)

                job_script.write(f"echo \"\" \n")
            try_no += 1

    # If the file was not created, return an error
    if not os.path.exists(job_script_path):
        print(f"Error: Could not create the shell script {job_script_path}")
        exit(1)

    # Make the output script executable
    os.chmod(job_script_path, 0o755)

    return job_script_path


def main():
    # Parse the command line arguments
    opts = parse_options()
    # Remove '--' from opts.sbatch_opts if necessary
    if opts.sbatch_opts.startswith('--'):
        opts.sbatch_opts = opts.sbatch_opts[2:]

    if opts.job_name == "":
        opts.job_name = generate_job_name()
    print(f"Job name: {opts.job_name}")

    # Create the output directory
    job_dir = os.path.abspath(opts.output_root_dir + '/' + opts.job_name)
    print(f"Output directory: {job_dir}")

    # Scratch directory for intermediate files
    if opts.work_dir is None:
        opts.work_dir = os.path.abspath(job_dir + '/temp')
    print(f"Work directory: {opts.work_dir}")

    min_cluster_size_list = parse_range(opts.min_cluster_size_range)
    min_cluster_size_set_cmnds = [f'MIN_CLUSTER_SIZE={x}' for x in
                                  min_cluster_size_list]

    amst_approx_bound_list = parse_float_list(opts.amst_approx_bound_range)

    # Generate a benchmark batch script
    job_script = gen_anarci_dist_bench_script(opts.job_name, job_dir, 
                                                opts.work_dir,
                                                opts.num_nodes, 
                                                opts.num_tasks_per_node,
                                                opts.point_path,
                                                opts.distance_name, 
                                                opts.region_names,
                                                opts.substitution_matrix, 
                                                opts.inverse_power,
                                                opts.dont_include_ends, 
                                                opts.use_hamming_dist,
                                                opts.dnnd_exe, opts.nng_k,
                                                opts.nng_r, opts.nng_delta,
                                                opts.backup_knng,
                                                opts.mfc_exe, opts.amst_exe, 
                                                amst_approx_bound_list,
                                                opts.clustering_exe,
                                                min_cluster_size_set_cmnds,
                                                opts.min_samples,
                                                opts.distributed_hdbscan,
                                                opts.verbose,
                                                opts.ground_truth_path,
                                                opts.evaluator_exe,
                                                opts.ygm_cluster_eval,
                                                opts.input_dnnd_ds_path)

    print(f"Generated batch script: {job_script}")

    job_submission_cmd = f"sbatch {opts.sbatch_opts} {job_script}"
    # Write job execution commands log
    with open(f'{job_dir}/info.txt', 'w') as f:
        f.write(f"Command executed by user:\n")
        f.write(' '.join(sys.argv))
        f.write('\n\n')
        f.write(f"To submit the job, run:\n{job_submission_cmd}\n")

    # Submit the job
    if opts.submit_job:
        print(f"Submit job: {job_script}")
        os.system(job_submission_cmd)
    else:
        print(f"To submit the job, run:\n{job_submission_cmd}\n")


if __name__ == '__main__':
    main()
