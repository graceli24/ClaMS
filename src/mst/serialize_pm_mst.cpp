// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

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

#include <spdlog/spdlog.h>
#include <metall/metall.hpp>

#include "../common.hpp"

using namespace clams;

void parse_option(int argc, char *argv[], std::filesystem::path &pm_mst_path,
                  std::filesystem::path &output_path) {
  int opt_char;
  while ((opt_char = getopt(argc, argv, "i:o:h")) != -1) {
    switch (opt_char) {
      case 'i':
        pm_mst_path = std::filesystem::path(optarg);
        break;
      case 'o':
        output_path = std::filesystem::path(optarg);
        break;
      case 'h':
        std::cout << "Usage: " << argv[0]
                  << " -i <pm_mst_path> -o <output_path>" << std::endl;
        std::exit(0);
      default:
        std::cerr << "Unknown option: " << static_cast<char>(opt_char)
                  << std::endl;
        std::exit(1);
    }
  }

  if (pm_mst_path.empty()) {
    std::cerr << "No input Metall MST path is specified (-i)." << std::endl;
    std::exit(1);
  }

  if (output_path.empty()) {
    std::cerr << "No output path is specified (-o)." << std::endl;
    std::exit(1);
  }
}

int main(int argc, char *argv[]) {
  std::filesystem::path pm_mst_path;
  std::filesystem::path output_path;
  parse_option(argc, argv, pm_mst_path, output_path);

  metall::manager   metall_manager(metall::open_read_only, pm_mst_path);
  const auto *const input_mst_edges =
      metall_manager.find<weighted_edge_list_t>(metall::unique_instance).first;
  if (!input_mst_edges) {
    spdlog::critical("Failed to find MST edges in Metall datastore at {}",
                     pm_mst_path.string());
    std::abort();
  }
  std::cout << "#of MST edges: " << input_mst_edges->size() << std::endl;

  std::cout << "Serializing MST edges to: " << output_path << std::endl;
  std::ofstream ofs(output_path);
  if (!ofs) {
    std::cerr << "Failed to open " << output_path << std::endl;
    std::abort();
  }

  for (const auto &edge : *input_mst_edges) {
    ofs << edge << "\n";
  }
  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to write to file: " << output_path << std::endl;
    std::abort();
  }

  std::cout << "Successfully serialized MST edges" << std::endl;

  return 0;
}