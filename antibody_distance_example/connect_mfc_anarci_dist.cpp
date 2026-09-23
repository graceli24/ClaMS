// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#include <algorithm>
#include <iterator>
#include <vector>
#define CLAMS_USE_SALTATLAS

#include <unistd.h>

#include <filesystem>
#include <iostream>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/utility/timer.hpp>

#include "../src/common.hpp"
#include "../src/mfc/mfc.h"
#include "anarci_common.hpp"
// #include "mfc_sample_lcc.h"

namespace cls = clams;

struct option_t {
  std::filesystem::path pm_knng_datastore_path;
  std::string           bridge_edge_dump_file;
  std::filesystem::path component_size_dump_file;
  // Distance name options are identity or substitution
  std::string              distance_name        = "identity";
  std::vector<std::string> region_names         = {"full"};
  bool                     dont_include_ends    = false;
  bool                     use_hamming_distance = false;
  // Assumes we run the program from custom_distance_example
  std::filesystem::path substitution_matrix_path = "./BLOSUM62.txt";
  // Convert substitution similarity score to a distance as
  // distance = similarity^(-a)
  cls::distance_t inverse_power = 1;
};

void show_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name
            << " -d <pm_knng_datastore_path> [-f <distance_name>] [-b "
               "<bridge_edge_dump_file>] [distance options]"
            << std::endl;
  std::cout << "  -d <path>: Path to the distributed PM kNNG datastore."
            << std::endl;
  std::cout << "  -b <path>: (Optional) File to dump the bridge edges."
            << std::endl;
  std::cout << "  -f <str>   Distance name. Options are 'identity' or "
               "'substitution' (default: identity)"
            << std::endl;
  std::cout << "  -p <str>   Region names. Regions of the antibody "
               "ANARCI sequence to use. Options are 'full', 'cdrs', 'cdr{i}' "
               "where i is 1 2 or 3, or a comma-separated list of positions "
               "(default: full)"
            << std::endl;
  std::cout << "  -x         Don't include ends (start/end of ANARCI positions "
               "before there's amino acids). Only applies if using identity "
               "distance."
            << std::endl;
  std::cout << "  -H         Use unnormalized Hamming distance instead of "
               "identity distance."
            << std::endl;
  std::cout << "  -M <path>  Path to substitution matrix file. Only applies if "
               "using substitution distance."
            << std::endl;
  std::cout << "  -a <float>   Power a used to convert substitution similarity "
               "to distance. distance = similarity^(-a)."
            << std::endl;
}

bool parse_option(int argc, char *argv[], option_t &opt) {
  int n;
  opt.pm_knng_datastore_path.clear();
  std::string provided_region_option = "";

  while ((n = getopt(argc, argv, "d:f:b:c:p:M:a:xH")) != -1) {
    switch (n) {
      case 'd':
        opt.pm_knng_datastore_path = std::filesystem::path(optarg);
        break;

      case 'f':
        opt.distance_name = optarg;
        break;

      case 'b':
        opt.bridge_edge_dump_file = optarg;
        break;

      case 'c':
        opt.component_size_dump_file = std::filesystem::path(optarg);
        break;

      case 'p':
        provided_region_option = optarg;
        break;

      case 'x':
        opt.dont_include_ends = true;
        break;

      case 'H':
        opt.use_hamming_distance = true;
        break;

      case 'M':
        opt.substitution_matrix_path = optarg;
        break;

      case 'a':
        opt.inverse_power = std::stof(optarg);
        break;

      default:
        return false;
    }
  }

  if (opt.pm_knng_datastore_path.empty()) {
    std::cerr << "Datastore path is required (-d)." << std::endl;
    return false;
  }
  if (!std::filesystem::exists(opt.pm_knng_datastore_path)) {
    std::cerr << "Datastore does not exist: " << opt.pm_knng_datastore_path
              << std::endl;
    return false;
  }

  // Parse the provided region option for calculating identity if it exists
  if (provided_region_option != "") {
    std::vector<std::string> region_names;
    if (provided_region_option == "cdrs") {
      region_names = {"cdr1", "cdr2", "cdr3"};
    } else if (provided_region_option.find(',') != std::string::npos) {
      std::stringstream ss(provided_region_option);
      std::string       substr;
      while (std::getline(ss, substr, ',')) {
        region_names.push_back(substr);
      }
    } else {
      region_names.push_back(provided_region_option);
    }
    opt.region_names = region_names;
  }

  return true;
}

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);

  option_t   opt;
  const bool opt_parse_ret = parse_option(argc, argv, opt);
  if (!opt_parse_ret) {
    if (comm.rank0()) {
      show_usage(argv[0]);
    }
    return EXIT_FAILURE;
  }

  comm.cout0() << "PM datastore path\t" << opt.pm_knng_datastore_path
               << std::endl;
  comm.cout0() << "Distance name\t" << opt.distance_name << std::endl;

  ygm::utility::timer root_timer;
  {
    cls::dist_pm_knng_t pm_knng(comm.get_mpi_comm());
    pm_knng.open(opt.pm_knng_datastore_path);

    static auto &knng   = pm_knng.get_knng();
    static auto &pstore = pm_knng.get_point_store();

    comm.cout0() << "Opened" << std::endl;
    std::cout.flush();

    ygm::utility::timer cc_timer;

    ygm::container::disjoint_set<cls::id_t> disjoint_set(comm);
    for (const auto &[source, neighbors] : knng) {
      for (const auto &neighbor : neighbors) {
        disjoint_set.async_union(source, neighbor.id);
      }
    }

    comm.cout0() << "Read all points" << std::endl;
    std::cout.flush();

    comm.barrier();

    disjoint_set.all_compress();
    comm.cout0() << "Debug: disjoint_set.num_sets() = "
                 << disjoint_set.num_sets() << std::endl;
    std::cout.flush();

    std::set<cls::id_t> rep_set;
    disjoint_set.for_all(
        [&](const auto &p, const auto &rep) { rep_set.insert(rep); });
    rep_set = ygm::all_reduce(
        rep_set,
        [](std::set<cls::id_t> a, std::set<cls::id_t> b) {
          std::set<cls::id_t> res;
          std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                         std::inserter(res, res.end()));
          return res;
        },
        comm);

    comm.cout0() << "Found reps" << std::endl;
    comm.cout0() << "Debug: number of reps = " << rep_set.size() << std::endl;
    std::cout.flush();

    std::vector<std::shared_ptr<
        ygm::container::bag<std::pair<cls::id_t, cls::point_t>>>>
        bags;
    std::map<cls::id_t, std::shared_ptr<ygm::container::bag<
                            std::pair<cls::id_t, cls::point_t>>>>
        bag_map;
    static std::map<cls::id_t, std::shared_ptr<ygm::container::bag<
                                   std::pair<cls::id_t, cls::point_t>>>>
        *s_bag_map;
    s_bag_map = &bag_map;

    comm.cout0() << "Debug: bag definitions done" << std::endl;  // debug
    std::cout.flush();

    bags.reserve(rep_set.size());
    for (auto &rep : rep_set) {
      bags.emplace_back(
          new ygm::container::bag<std::pair<cls::id_t, cls::point_t>>(comm));
      bag_map.emplace(rep, bags.back());
    }

    comm.cout0() << "Debug: created bag of bags" << std::endl;  // debug
    std::cout.flush();

    disjoint_set.for_all([&](const auto &p, const auto &rep) {
      comm.async(
          cls::dist_pm_knng_t::get_owner(p, comm.size()),
          [](cls::id_t p, cls::id_t rep) {
            auto point = pstore.find(p);
            if (point == pstore.end()) {
              std::cerr << "Cannot find point for ID: " << p << std::endl;
              std::abort();
            }
            s_bag_map->at(rep)->local_insert({p, point->second});
          },
          p, rep);
    });

    comm.barrier();

    comm.cout0() << "Debug: assign points to correct bags"
                 << std::endl;  // debug
    std::cout.flush();

    for (auto &b : bags) {
      if (b->size() > (size_t)comm.size()) b->rebalance();
    }

    comm.cout0() << "CC took (s): " << cc_timer.elapsed() << std::endl;
    comm.cout0() << "#of CCs: " << bags.size() << std::endl;

    // Get the LCC component id and size on all ranks
    id_t        lcc_component_id = 0;
    std::size_t lcc_size         = 0;
    id_t        i                = 0;
    for (auto &b : bags) {
      std::size_t size = b->size();
      if (size > lcc_size) {
        lcc_size         = size;
        lcc_component_id = i;
      }
      ++i;
    }
    comm.barrier();
    comm.cout0() << "Largest connected component (LCC) size: " << lcc_size
                 << std::endl;
    comm.cout0() << "LCC index: " << lcc_component_id << std::endl;

    // DEBUG write component sizes to file
    try {
      if (opt.component_size_dump_file.empty()) {
        opt.component_size_dump_file =
            opt.pm_knng_datastore_path / "../component_sizes.csv";
      }
      if (!opt.component_size_dump_file.empty()) {
        cc_timer.reset();

        comm.cout0() << "Getting component sizes" << std::endl;
        // (Component id, rank, component size, local size)
        static std::vector<std::tuple<cls::id_t, int, cls::id_t, cls::id_t>>
             all_component_sizes;
        id_t i = 0;
        for (auto &b : bags) {
          std::size_t size = b->size();
          if (comm.rank() == 0) {
            std::tuple<cls::id_t, int, cls::id_t, cls::id_t> component_tuple{
                i, comm.rank(), size, b->local_size()};
            all_component_sizes.push_back(component_tuple);
          }
          ++i;
        }
        comm.barrier();

        // Print component sizes
        if (comm.rank() == 0) {
          std::cout << "Write component sizes to "
                    << opt.component_size_dump_file << std::endl;
          std::filesystem::path parent_dir =
              opt.component_size_dump_file.parent_path();
          std::filesystem::create_directories(parent_dir);
          std::ofstream ofs(opt.component_size_dump_file);
          if (!ofs.is_open()) {
            comm.cerr0() << "Cannot open file: " << opt.component_size_dump_file
                         << std::endl;
            return EXIT_FAILURE;
          } else {
            ofs << "component_id,size,local_size\n";
            for (const auto &row : all_component_sizes) {
              ofs << std::get<0>(row) << "," << std::get<2>(row) << ","
                  << std::get<3>(row) << "\n";
            }
            ofs.flush();
            ofs.close();
          }
        }

        comm.barrier();
        comm.cout0() << "Writing CC sizes took (s): " << cc_timer.elapsed()
                     << std::endl;
      }
    } catch (...) {
      // pass
    }

    // Define distance function
    // Get the position to index map for the indicies in our distance
    // calculation. We will only bother reading these positions.
    std::map<std::string, uint16_t> position_to_idx_map =
        get_position_to_idx_map_from_region_name(opt.region_names);

    // Get a sorted vector of indices used in distance calculation
    std::stringstream            positions_in_calc_ss;
    static std::vector<uint16_t> idxs_for_distance_calculation;
    for (auto &[position, idx] : position_to_idx_map) {
      idxs_for_distance_calculation.push_back(idx);
      positions_in_calc_ss << position << " ";
    }
    std::sort(idxs_for_distance_calculation.begin(),
              idxs_for_distance_calculation.end());

    comm.cout0() << "ANARCI distance type: " << opt.distance_name << std::endl;
    comm.cout0() << "Positions used in distance calculation: "
                 << positions_in_calc_ss.str() << std::endl;
    if (opt.distance_name == "identity") {
      comm.cout0() << "Don't include edges: " << opt.dont_include_ends << "\n"
                   << "Use unnormalized Hamming distance: "
                   << opt.use_hamming_distance << std::endl;
    }

    std::vector<std::vector<sim_t>> substitution_matrix;
    std::vector<amino_acid_t>       char_to_int_converter;
    if (opt.distance_name == "substitution") {
      comm.cout0() << "Distance calculation uses substitution matrix at: "
                   << opt.substitution_matrix_path << std::endl;
      comm.cout0() << "Distance is calculated distance = similarity^(-a) with "
                   << "a = " << opt.inverse_power << std::endl;

      auto ret = read_substitution_matrix(opt.substitution_matrix_path);
      substitution_matrix   = ret.first;
      char_to_int_converter = ret.second;
    }

    // Get the correct anarci distance function
    static auto anarci_identity_distance = get_anarci_identity_distance_lambda(
        opt.dont_include_ends, opt.use_hamming_distance);
    static auto anarci_substitution_distance =
        get_anarci_substitution_distance_lambda(substitution_matrix,
                                                opt.inverse_power);

    static bool use_substitution_distance;
    if (opt.distance_name == "substitution") {
      use_substitution_distance = true;
    } else {
      use_substitution_distance = false;
    }

    static auto ref_distance_func = [](const auto &seq1, const auto &seq2) {
      if (use_substitution_distance) {
        return anarci_substitution_distance(seq1, seq2);
      } else {
        return anarci_identity_distance(seq1, seq2);
      }
    };

    // for (auto &b : bags) {
    //   size_t s = b.size();
    //   comm.cout0() << "    CC: size " << s << " local: " << b.local_size() <<
    //   std::endl;
    // }

    static std::vector<std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
        local_bridge_edges;

    ygm::utility::timer connect_timer;
    // auto                ref_distance_func =
    //     saltatlas::distance::distance_function<cls::point_t,
    //     cls::distance_t>(
    //         opt.distance_name);
    mfc(
        comm, bags,
        [&](std::pair<cls::id_t, cls::point_t> a,
            std::pair<cls::id_t, cls::point_t> b) {
          return ref_distance_func(a.second, b.second);
        },
        [&](std::pair<cls::id_t, cls::point_t> a,
            std::pair<cls::id_t, cls::point_t> b) {
          auto dist = ref_distance_func(a.second, b.second);
          comm.async(
              cls::dist_pm_knng_t::get_owner(a.first, comm.size()),
              [](cls::id_t a, cls::id_t b, cls::distance_t dist) {
                local_bridge_edges.emplace_back(a, b, dist);
                knng[a].emplace_back(b, dist);
              },
              a.first, b.first, dist);
        });

    comm.barrier();
    comm.cout0() << "Connected CCs took (s): " << connect_timer.elapsed()
                 << std::endl;
    comm.cout0() << "Entire algorithm took (s): " << root_timer.elapsed()
                 << std::endl;

    if (!opt.bridge_edge_dump_file.empty()) {
      comm.cout0() << "Gather bridge edges to rank 0" << std::endl;
      // Gather bridge edges from all ranks to rank 0
      static std::vector<std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
          all_bridge_edges;
      comm.cf_barrier();
      comm.async(
          0,
          [](const auto &edges) {
            all_bridge_edges.insert(all_bridge_edges.end(), edges.begin(),
                                    edges.end());
          },
          local_bridge_edges);
      comm.barrier();

      // Print bridge edges
      if (comm.rank0()) {
        comm.cout0() << "Dump bridge edges to " << opt.bridge_edge_dump_file
                     << std::endl;
        std::ofstream ofs(opt.bridge_edge_dump_file);
        if (!ofs.is_open()) {
          comm.cerr0() << "Cannot open file: " << opt.bridge_edge_dump_file
                       << std::endl;
          return EXIT_FAILURE;
        }
        for (const auto &edge : all_bridge_edges) {
          ofs << std::get<0>(edge) << "\t" << std::get<1>(edge) << "\t"
              << std::get<2>(edge) << std::endl;
        }
      }
    }
    comm.cf_barrier();
  }
  comm.cout0() << "Done" << std::endl;

  return EXIT_SUCCESS;
}
