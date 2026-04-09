// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#define CLAMS_USE_SALTATLAS
#define METALL_DISABLE_CONCURRENCY

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <metall/metall.hpp>

#include "../common.hpp"

using namespace clams;

void parse_option(int argc, char *argv[], std::filesystem::path &pm_knng_path,
                  std::filesystem::path &output_path) {
  int opt_char;
  while ((opt_char = getopt(argc, argv, "i:o:h")) != -1) {
    switch (opt_char) {
      case 'i':
        pm_knng_path = std::filesystem::path(optarg);
        break;
      case 'o':
        output_path = std::filesystem::path(optarg);
        break;
      case 'h':
        std::cout << "Usage: " << argv[0]
                  << " -i <pm_knng_path> -o <output_path>" << std::endl;
        std::exit(0);
      default:
        std::cerr << "Unknown option: " << static_cast<char>(opt_char)
                  << std::endl;
        std::exit(1);
    }
  }

  if (pm_knng_path.empty()) {
    std::cerr << "No input PM KNNG path is specified (-i)." << std::endl;
    std::exit(1);
  }

  if (output_path.empty()) {
    std::cerr << "No output path is specified (-o)." << std::endl;
    std::exit(1);
  }
}

int main(int argc, char *argv[]) {
  ygm::comm comm(&argc, &argv);

  std::filesystem::path pm_knng_path;
  std::filesystem::path output_path;
  parse_option(argc, argv, pm_knng_path, output_path);

  dnnd_t     dnnd(saltatlas::open_read_only, pm_knng_path, comm);
  const auto knng_id = dnnd.get_index_ids().front();
  dnnd.dump_graph(knng_id, output_path, true);
  comm.cf_barrier();
  comm.cout0("Finished dumping KNNG to: ", output_path.string());

  return 0;
}