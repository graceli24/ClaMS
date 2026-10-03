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

namespace cls = clams;

struct option_t {
  std::filesystem::path pm_knng_datastore_path;
  std::string           bridge_edge_dump_file;
  std::filesystem::path component_size_dump_file;
  float                 lcc_sample_fraction = 1.0;
};

void show_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name
            << "  -d <pm_knng_datastore_path> [-f <distance_name>] [-b "
               "<bridge_edge_dump_file>] [distance options]"
            << std::endl;
  std::cout << "  -d <path>: Path to the distributed PM kNNG datastore."
            << std::endl;
  std::cout << "  -b <path>: (Optional) File to dump the bridge edges."
            << std::endl;
  std::cout << "  -s <float>:(Optional) Sample the largest connected "
               "component to only consider s fraction of points. Requires "
               "0 < s < 1 and if no valid is s provided, defaults to s = 1, no "
               "sampling of the LCC"
            << std::endl;
}

bool parse_option(int argc, char *argv[], option_t &opt) {
  int n;
  opt.pm_knng_datastore_path.clear();
  std::string provided_region_option = "";

  while ((n = getopt(argc, argv, "d:s:f:b:c:M:a:xH")) != -1) {
    switch (n) {
      case 'd':
        opt.pm_knng_datastore_path = std::filesystem::path(optarg);
        break;

      case 's':
        opt.lcc_sample_fraction = std::stof(optarg);
        break;

      case 'b':
        opt.bridge_edge_dump_file = optarg;
        break;

      case 'c':
        opt.component_size_dump_file = std::filesystem::path(optarg);
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
  if (opt.lcc_sample_fraction < 0 | opt.lcc_sample_fraction > 1) {
    std::cerr << "Provided LCC sample fraction " << opt.lcc_sample_fraction
              << " is not strictly between 0 and 1. Defaulting to 1 and not "
                 "sampling the LCC."
              << std::endl;
    opt.lcc_sample_fraction = 1.0;
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

  ygm::utility::timer root_timer;
  {
    ygm::utility::timer step_timer;
    cls::dist_pm_knng_t pm_knng(comm.get_mpi_comm());
    pm_knng.open(opt.pm_knng_datastore_path);

    static auto &knng   = pm_knng.get_knng();
    static auto &pstore = pm_knng.get_point_store();

    comm.cout0() << "Opened" << std::endl;

    ygm::utility::timer cc_timer;

    ygm::container::disjoint_set<cls::id_t> disjoint_set(comm);
    for (const auto &[source, neighbors] : knng) {
      for (const auto &neighbor : neighbors) {
        disjoint_set.async_union(source, neighbor.id);
      }
    }

    comm.cout0() << "Read all points" << std::endl;
    comm.cout0() << "  Time to read points and add to disjoint set (s): "
                 << step_timer.elapsed() << std::endl;
    step_timer.reset();

    comm.barrier();

    disjoint_set.all_compress();

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
    comm.cout0() << "  Number of reps = " << rep_set.size() << std::endl;
    comm.cout0()
        << "  Time to compress disjoint set and get reps on all ranks (s): "
        << step_timer.elapsed() << std::endl;
    step_timer.reset();

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

    bags.reserve(rep_set.size());
    for (auto &rep : rep_set) {
      bags.emplace_back(
          new ygm::container::bag<std::pair<cls::id_t, cls::point_t>>(comm));
      bag_map.emplace(rep, bags.back());
    }

    comm.cout0() << "  Time to initialize bags (s): " << step_timer.elapsed()
                 << std::endl;
    step_timer.reset();

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

    comm.cout0() << "  Time to add points to correct component bag (s): "
                 << step_timer.elapsed() << std::endl;
    step_timer.reset();

    for (auto &b : bags) {
      if (b->size() > (size_t)comm.size()) b->rebalance();
    }

    comm.cout0() << "  Time to rebalance bags (s): " << step_timer.elapsed()
                 << std::endl;

    comm.cout0() << "CC took (s): " << cc_timer.elapsed() << std::endl;
    comm.cout0() << "#of CCs: " << bags.size() << std::endl;
    step_timer.reset();

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
    if (!opt.component_size_dump_file.empty()) {
      step_timer.reset();

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
        std::cout << "Write component sizes to " << opt.component_size_dump_file
                  << std::endl;
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
      comm.cout0() << "Writing CC sizes took (s): " << step_timer.elapsed()
                   << std::endl;
    }

    // Sample from the LCC if needed
    if ((opt.lcc_sample_fraction < 1.0) & (lcc_size > comm.size() * 10)) {
      ygm::utility::timer sample_lcc_timer;
      comm.cout0() << "Sampling from the LCC. Sample fraction: "
                   << opt.lcc_sample_fraction << std::endl;

      // Get which point indicies in the component to sample
      uint64_t num_samples =
          static_cast<uint64_t>(std::ceil(opt.lcc_sample_fraction * lcc_size));
      std::vector<uint64_t> idxs(num_samples);
      comm.cout0() << "Number of samples: " << num_samples << std::endl;

      if (comm.rank() == 0) {
        // Create a vector with index from 0 to (lcc_size - 1)
        idxs.resize(lcc_size);
        std::iota(idxs.begin(), idxs.end(), 0);

        // Partial Fischer Yates shuffle
        uint64_t                   seed = std::random_device{}();
        std::default_random_engine random(seed);
        for (int64_t i = 0; i < num_samples; ++i) {
          std::uniform_int_distribution<int64_t> dist(i, lcc_size);
          int64_t                                j = dist(random);
          std::swap(idxs[i], idxs[j]);
        }

        idxs.resize(num_samples);
      }
      MPI_Bcast(idxs.data(), num_samples, MPI_UINT64_T, 0, comm.get_mpi_comm());

      // Get the prefix sum of local LCC size
      uint64_t local_lcc_size = bags[lcc_component_id]->local_size();
      uint64_t prefix_sum     = 0;
      MPI_Exscan(&local_lcc_size, &prefix_sum, 1, MPI_UINT64_T, MPI_SUM,
                 comm.get_mpi_comm());
      if (comm.rank() == 0) {
        prefix_sum = 0;
      }

      // Give every rank vector of all consecutive prefix sums
      std::vector<uint64_t> prefix_sums(comm.size());
      MPI_Allgather(&prefix_sum, 1, MPI_UINT64_T, prefix_sums.data(), 1,
                    MPI_UINT64_T, comm.get_mpi_comm());
      // Append the lcc size as the last entry to prefix sums
      prefix_sums.push_back(lcc_size);

      // Get the sampled points on each rank
      std::vector<std::pair<cls::id_t, cls::point_t>> sampled_points;
      int                                             rank = comm.rank();
      auto local_lcc_begin = bags[lcc_component_id]->local_begin();
      for (uint64_t &idx : idxs) {
        if ((prefix_sums.at(rank) <= idx) & (idx < prefix_sums.at(rank + 1))) {
          uint64_t local_idx = idx - prefix_sums.at(rank);
          sampled_points.push_back(*std::next(local_lcc_begin, local_idx));
        }
      }
      comm.barrier();

      // Clear the old LCC component bag, insert the sampled component points
      // and rebalance
      bags[lcc_component_id]->clear();
      for (auto &item : sampled_points) {
        bags[lcc_component_id]->local_insert(item);
      }
      comm.barrier();
      if (bags[lcc_component_id]->size() > comm.size()) {
        bags[lcc_component_id]->rebalance();
      }

      comm.cout0() << "Time to sample LCC (s): " << sample_lcc_timer.elapsed()
                   << std::endl;
    }

    static std::vector<std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
        local_bridge_edges;

    ygm::utility::timer connect_timer;
    mfc(
        comm, bags,
        [&](std::pair<cls::id_t, cls::point_t> a,
            std::pair<cls::id_t, cls::point_t> b) {
          return anarci_substitution_distance(a.second, b.second);
        },
        [&](std::pair<cls::id_t, cls::point_t> a,
            std::pair<cls::id_t, cls::point_t> b) {
          auto dist = anarci_substitution_distance(a.second, b.second);
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
