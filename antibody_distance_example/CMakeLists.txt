# Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
# Project Developers. See the top-level COPYRIGHT file for details.


add_basic_executable(build_knng_neo_anarci_dist build_knng_neo_anarci_dist.cpp)
setup_saltatlas_target(build_knng_neo_anarci_dist)

add_basic_executable(connect_mfc_anarci_dist connect_mfc_anarci_dist.cpp)
setup_saltatlas_target(connect_mfc_anarci_dist)