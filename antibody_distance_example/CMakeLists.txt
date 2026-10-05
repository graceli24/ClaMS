# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


add_basic_executable(build_knng_neo_anarci_dist build_knng_neo_anarci_dist.cpp)
setup_saltatlas_target(build_knng_neo_anarci_dist)

add_basic_executable(connect_mfc_anarci_dist connect_mfc_anarci_dist.cpp)
setup_saltatlas_target(connect_mfc_anarci_dist)

add_basic_executable(bridge_edges_anarci_dist bridge_edges_anarci_dist.cpp)
setup_saltatlas_target(bridge_edges_anarci_dist)

# add_basic_executable(process_oas_raw_data process_oas_raw_data.cpp)
# setup_saltatlas_target(process_oas_raw_data)

# add_basic_executable(assign_sequence_ids assign_sequence_ids.cpp)
# setup_saltatlas_target(assign_sequence_ids)

