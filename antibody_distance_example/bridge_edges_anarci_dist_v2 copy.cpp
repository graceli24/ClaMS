// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#define CLAMS_USE_SALTATLAS

#include <unistd.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <vector>

#include <boost/type_index.hpp>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/container/map.hpp>
#include <ygm/utility/timer.hpp>

#include "../src/common.hpp"
// #include "../src/mfc/mfc.h"
#include "anarci_common.hpp"

/*
Hash function for std::pair
Syntax taken from: https://lists.isocpp.org/std-discussion/2020/12/0937.php
Using boost's hash_combine:
https://www.boost.org/doc/libs/1_51_0/doc/html/hash/combine.html

*/
namespace std {
template <typename T1, typename T2>
struct hash<pair<T1, T2>> {
  size_t operator()(const pair<T1, T2> &input_pair) const {
    size_t seed = 0;
    boost::hash_combine(seed, input_pair.first);
    boost::hash_combine(seed, input_pair.second);
    return seed;
  }
};
}  // namespace std

namespace cls        = clams;
using point_t        = saltatlas::pm_feature_vector<amino_acid_t>;
using dist_pm_knng_t = cls::dist_pm_knng<cls::id_t, point_t, cls::distance_t>;
using pstore_t =
    cls::dist_pm_knng<cls::id_t, point_t, cls::distance_t>::point_store_type;

struct option_t {
  std::filesystem::path pm_knng_datastore_path;
  std::string           bridge_edge_dump_file;
  std::filesystem::path component_size_dump_file;
  float                 lcc_sample_fraction          = 1.0;
  bool                  skip_small_component_bridges = false;
};

void show_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name
            << "  -d <pm_knng_datastore_path> [-b "
               "<bridge_edge_dump_file>] [-c <component_size_dump_file>] "
               "[sampling options]"
            << std::endl;
  std::cout << "  -d <path>: Path to the distributed PM kNNG datastore."
            << std::endl;
  std::cout << "  -b <path>: (Optional) File to dump the bridge edges."
            << std::endl;
  std::cout << "  -s <float>: (Optional) Sample the largest connected "
               "component to only consider s fraction of points. Requires "
               "0 <= s <= 1 and if no valid is s provided, defaults to "
               "s = 1, no sampling of the LCC. If s = 0, then we sample "
               "a single representative from the LCC."
            << std::endl;
  std::cout << "  -r <path>: (Optional) Skip checking bridge edges between "
               "small components and just connect each small component "
               "representative to the LCC. Instead of finding MFC bridge "
               "edges, this is like randomly connecting the small "
               "components to the LCC."
            << std::endl;
}

bool parse_option(int argc, char *argv[], option_t &opt) {
  int n;
  opt.pm_knng_datastore_path.clear();
  std::string provided_region_option = "";

  while ((n = getopt(argc, argv, "d:s:f:b:c:rxH")) != -1) {
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

      case 'r':
        opt.skip_small_component_bridges = true;
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
  if ((opt.lcc_sample_fraction < 0) | (opt.lcc_sample_fraction > 1)) {
    std::cerr << "Provided LCC sample fraction " << opt.lcc_sample_fraction
              << " is not between 0 and 1. Defaulting to 1 and not "
                 "sampling the LCC."
              << std::endl;
    opt.lcc_sample_fraction = 1.0;
  }

  return true;
}

// Translates id_t to MPI type
MPI_Datatype mpi_id_type() {
  if constexpr (std::is_same_v<cls::id_t, uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<cls::id_t, uint64_t>) {
    return MPI_UINT64_T;
  } else {
    static_assert(std::is_same_v<cls::id_t, uint32_t> ||
                      std::is_same_v<cls::id_t, uint64_t>,
                  "id_t must be uint32_t or uint64_t");
  }
}

// Translates distance_t to MPI type
MPI_Datatype mpi_distance_type() {
  if constexpr (std::is_same_v<cls::distance_t, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<cls::distance_t, double>) {
    return MPI_DOUBLE;
  } else {
    static_assert(std::is_same_v<cls::distance_t, float> ||
                      std::is_same_v<cls::distance_t, double>,
                  "distance_t must be float or double");
  }
}

// Get connected components
void get_connected_components(
    ygm::container::disjoint_set<cls::id_t> &disjoint_set,
    std::vector<uint64_t>                   &component_sizes,
    std::vector<std::vector<cls::id_t>>     &local_component_members,
    std::vector<cls::id_t>                  &local_lcc_members) {
  ygm::comm          &comm = disjoint_set.comm();
  ygm::utility::timer step_timer;

  disjoint_set.all_compress();

  std::map<cls::id_t, uint64_t> local_component_size_map;
  disjoint_set.for_all([&local_component_size_map](
                           [[maybe_unused]] const auto &p, const auto &rep) {
    ++local_component_size_map[rep];
  });
  std::map<cls::id_t, uint64_t> component_size_map = ygm::all_reduce(
      local_component_size_map,
      [](std::map<cls::id_t, uint64_t> &a, std::map<cls::id_t, uint64_t> &b) {
        std::map<cls::id_t, uint64_t> result = a;
        for (auto &[key, value] : b) {
          result[key] += value;
        }
        return result;
      },
      comm);
  comm.barrier();
  uint64_t num_components = component_size_map.size();

  comm.cout0() << "  Time to compress disjoint set and get disjoint set reps "
                  "and component sizes on all ranks (s): "
               << step_timer.elapsed() << std::endl;  // debug
  step_timer.reset();

  // Get the representative and size of the largest connected component (LCC)
  // Find the iterator pointing to the maximum value
  auto maxIt = std::max_element(
      component_size_map.begin(), component_size_map.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });
  cls::id_t lcc_rep  = maxIt->first;
  uint64_t  lcc_size = maxIt->second;
  comm.cout0() << "  Largest connected component (LCC) size: " << lcc_size
               << std::endl;

  // Assign an index to each component other than the LCC
  std::map<cls::id_t, cls::id_t> component_rep_to_idx;
  std::vector<cls::id_t>         djset_reps;
  djset_reps.reserve(num_components);
  component_sizes.reserve(num_components);

  cls::id_t idx  = 0;
  auto      hint = component_rep_to_idx.end();
  for (auto &[rep, size] : component_size_map) {
    if (rep != lcc_rep) {
      component_rep_to_idx.emplace_hint(hint, rep, idx);
      djset_reps.push_back(rep);
      component_sizes.push_back(size);
      ++idx;
    }
  }
  component_size_map.clear();  // no longer needed

  // Put the point ids in each component into vectors, the vector in the ith
  // position is the members for component i. LCC members are stored
  // separately
  local_component_members.resize(num_components - 1);
  disjoint_set.for_all([&local_component_members, &local_lcc_members,
                        &component_rep_to_idx,
                        &lcc_rep](const auto &point_id, const auto &rep) {
    if (rep == lcc_rep) {
      local_lcc_members.push_back(point_id);
    } else {
      local_component_members[component_rep_to_idx.at(rep)].push_back(point_id);
    }
  });
  comm.barrier();

  comm.cout0() << "  Time to add point ids to correct component vector (s): "
               << step_timer.elapsed() << std::endl;  // debug
}

// Sample LCC
void sample_largest_connected_component(
    ygm::comm &comm, std::vector<cls::id_t> &local_lcc_members,
    uint64_t num_samples) {
  uint64_t lcc_size = ygm::sum(local_lcc_members.size(), comm);

  // Get which point indicies in the component to sample
  std::vector<uint64_t> idxs(num_samples);
  if (comm.rank() == 0) {
    // Create a vector with indices from 0 to (lcc_size - 1)
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
  uint64_t local_lcc_size = local_lcc_members.size();
  uint64_t prefix_sum     = 0;
  MPI_Exscan(&local_lcc_size, &prefix_sum, 1, MPI_UINT64_T, MPI_SUM,
             comm.get_mpi_comm());
  if (comm.rank() == 0) {
    prefix_sum = 0;
  }

  // Get the sampled points on each rank
  std::vector<cls::id_t> local_sampled_points;
  local_sampled_points.clear();
  for (uint64_t &idx : idxs) {
    if ((prefix_sum <= idx) & (idx < prefix_sum + local_lcc_size)) {
      uint64_t local_idx = idx - prefix_sum;
      local_sampled_points.push_back(local_lcc_members.at(local_idx));
    }
  }

  // Clear the old LCC member vector and replace it with the sampled
  // points
  local_lcc_members.clear();
  local_lcc_members = local_sampled_points;
}

// Get the points in the (possibly sampled) LCC from pstore
std::vector<std::pair<cls::id_t, point_t>> get_lcc_point_data(
    ygm::comm &comm, pstore_t &_pstore,
    std::vector<cls::id_t> &local_lcc_members) {
  static std::vector<std::pair<cls::id_t, point_t>> lcc_points;
  lcc_points.clear();

  static auto &pstore = _pstore;

  for (cls::id_t &point_id : local_lcc_members) {
    // Go get the point from pstore
    comm.async(
        dist_pm_knng_t::get_owner(point_id, comm.size()),
        [](cls::id_t point_id) {
          auto point = pstore.find(point_id);
          if (point == pstore.end()) {
            std::cerr << "Cannot find point for ID: " << point_id << std::endl;
            std::abort();
          }

          point_t point_data(point->second.size());
          std::copy(point->second.begin(), point->second.end(),
                    point_data.begin());
          lcc_points.push_back({point_id, point_data});

          // Note, any of the below variants yields bad_alloc:
          // point_t point_data = point->second;
          // point_t point_data(point->second);
          // lcc_points.push_back({point_id, point->second})
        },
        point_id);
  }
  comm.barrier();

  return lcc_points;
}

// Randomly choose a single rep from each component that's not the LCC
// Do this instead of using the disjoint set reps so that it's actually
// random and go get the actual point data from pstore
std::vector<std::pair<cls::id_t, point_t>> get_small_component_random_reps(
    ygm::comm &comm, pstore_t &_pstore,
    std::vector<std::vector<cls::id_t>> &local_component_members,
    std::vector<uint64_t>               &component_sizes) {
  static auto &pstore = _pstore;

  static std::vector<std::pair<cls::id_t, point_t>> reps;
  reps.clear();
  reps.resize(local_component_members.size());

  uint64_t seed;
  if (comm.rank0()) {
    seed = std::random_device{}();
  }

  std::vector<uint64_t> local_sizes;
  std::vector<uint64_t> prefix_sums;
  local_sizes.reserve(local_component_members.size());
  prefix_sums.reserve(local_component_members.size());

  for (cls::id_t i = 0; i < local_component_members.size(); ++i) {
    local_sizes.push_back(local_component_members[i].size());
    prefix_sums.push_back(local_component_members[i].size());
  }

  // Broadcast the same random seed and get the prefix sums for each
  // component
  MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, comm.get_mpi_comm());
  MPI_Exscan(MPI_IN_PLACE, prefix_sums.data(), local_component_members.size(),
             MPI_UINT64_T, MPI_SUM, comm.get_mpi_comm());
  if (comm.rank() == 0) {
    std::fill(prefix_sums.begin(), prefix_sums.end(), 0);
  }

  // Generate the random representative for each connected component
  std::default_random_engine random(seed);

  for (cls::id_t i = 0; i < local_component_members.size(); i++) {
    cls::id_t chosen_idx = std::uniform_int_distribution<ssize_t>{
        0, static_cast<long>(component_sizes[i] - 1)}(random);

    // If the chosen point has point id stored on this rank
    if ((prefix_sums[i] <= chosen_idx) &
        (chosen_idx < prefix_sums[i] + local_sizes[i])) {
      uint64_t local_idx = chosen_idx - prefix_sums[i];

      // Go get the point from pstore
      cls::id_t point_id = local_component_members[i].at(local_idx);
      comm.async(
          dist_pm_knng_t::get_owner(point_id, comm.size()),
          [](cls::id_t point_id, cls::id_t component_id) {
            auto point = pstore.find(point_id);
            if (point == pstore.end()) {
              std::cerr << "Cannot find point for ID: " << point_id
                        << std::endl;
              std::abort();
            }

            // Set the rep on this rank
            point_t point_data(point->second.size());
            std::copy(point->second.begin(), point->second.end(),
                      point_data.begin());
            std::pair<cls::id_t, point_t> rep =
                std::make_pair(point_id, point_data);
            reps[component_id] = rep;
          },
          point_id, i);
    }
  }
  comm.barrier();

  // Broadcast the rep points to all ranks
  std::pair<cls::id_t, point_t> default_value;
  for (cls::id_t i = 0; i < local_component_members.size(); i++) {
    if (reps[i] != default_value) {
      comm.async_bcast(
          [](auto rep, cls::id_t component_id) { reps[component_id] = rep; },
          reps[i], i);
    }
  }
  comm.barrier();

  return reps;
}

// Get the best bridge edge from each small component rep to the (sampled)
// LCC
template <typename distance_function_t>
void bridge_edges_lcc_to_small(
    ygm::container::map<std::pair<cls::id_t, cls::id_t>,
                        std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
                                               &bridge_edge_map,
    std::vector<std::pair<cls::id_t, point_t>> &reps,
    std::vector<std::pair<cls::id_t, point_t>> &lcc_points,
    const distance_function_t &distance_function, cls::id_t lcc_component_id) {
  ygm::comm &comm = bridge_edge_map.comm();

  for (cls::id_t i = 0; i < reps.size() - 1; i++) {
    // Find the best bridge edge between each small component rep
    // and the (sampled) LCC
    cls::distance_t min_dist = std::numeric_limits<cls::distance_t>::max();

    cls::id_t best_point_id_lcc = 0;
    cls::id_t point_id_i        = reps[i].first;
    point_t   point_i           = reps[i].second;

    for (auto &lcc_point : lcc_points) {
      auto d = distance_function(point_i, lcc_point.second);
      if (d < min_dist) {
        min_dist          = d;
        best_point_id_lcc = lcc_point.first;
      }
    }

    // See if the local best bridge edge is better than the global one
    // stored for this pair of small component and lcc. Always store
    // component pairs as (smaller id, larger id)
    std::pair<cls::id_t, cls::id_t> component_pair{i, lcc_component_id};
    std::tuple<cls::id_t, cls::id_t, cls::distance_t> local_best_bridge_edge{
        point_id_i, best_point_id_lcc, min_dist};
    bridge_edge_map.async_visit(
        component_pair,
        [](const std::pair<cls::id_t, cls::id_t>             &pair_name,
           std::tuple<cls::id_t, cls::id_t, cls::distance_t> &best_bridge_edge,
           const std::tuple<cls::id_t, cls::id_t, cls::distance_t>
               &new_bridge_edge) {
          if (std::get<2>(new_bridge_edge) < std::get<2>(best_bridge_edge)) {
            best_bridge_edge = new_bridge_edge;
          }
        },
        local_best_bridge_edge);
  }
  comm.barrier();
}

// Get the potential bridge edges between the small components
template <typename distance_function_t>
void bridge_edges_small_to_small(
    ygm::container::map<std::pair<cls::id_t, cls::id_t>,
                        std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
                                               &bridge_edge_map,
    std::vector<std::pair<cls::id_t, point_t>> &reps,
    const distance_function_t                  &distance_function) {
  ygm::comm &comm = bridge_edge_map.comm();

  int num_small_components = reps.size();

  // Make a local copy of weight of the best bridge edges to the LCC
  std::vector<cls::distance_t> dist_to_lcc_vector(num_small_components);
  for (const auto &[component_pair, bridge_edge] : bridge_edge_map) {
    cls::id_t small_component_id              = component_pair.first;
    dist_to_lcc_vector.at(small_component_id) = std::get<2>(bridge_edge);
  }
  comm.barrier();
  MPI_Allreduce(MPI_IN_PLACE, dist_to_lcc_vector.data(), num_small_components,
                mpi_distance_type(), MPI_MAX, comm.get_mpi_comm());

  comm.cout0() << "Created local copies of LCC bridge edge weights."
               << std::endl;  // debug

  // Loop over every ordered pair of small components and check the rep
  // bridge edge - only store these best bridge edges if they have
  // distance
  // <= the better of the two small component bridge edges to the LCC
  for (cls::id_t i = 0; i < num_small_components; i++) {
    for (cls::id_t j = 0; j < num_small_components; j++) {
      if (i == j) {
        continue;
      }

      cls::id_t point_id_i = reps[i].first;
      point_t   point_i    = reps[i].second;
      cls::id_t point_id_j = reps[j].first;
      point_t   point_j    = reps[j].second;

      cls::distance_t rep_dist = distance_function(point_i, point_j);

      // See if the local best bridge edge is better than the global one
      // stored for this component pair - Only consider this bridge edge
      // if has distance smaller than at least one of the LCC bridge
      // edges for this component pair
      if ((rep_dist <= dist_to_lcc_vector[i]) |
          (rep_dist <= dist_to_lcc_vector[j])) {
        std::pair<cls::id_t, cls::id_t> component_pair{std::min(i, j),
                                                       std::max(i, j)};
        std::tuple<cls::id_t, cls::id_t, cls::distance_t> bridge_edge{
            point_id_i, point_id_j, rep_dist};
        bridge_edge_map.async_insert(component_pair, bridge_edge);
      }
    }
  }
  comm.barrier();
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
  ygm::utility::timer step_timer;

  // Read the input kNNG data

  step_timer.reset();
  dist_pm_knng_t pm_knng(comm.get_mpi_comm());
  // cls::dist_pm_knng_t pm_knng(comm.get_mpi_comm());
  pm_knng.open_read_only(opt.pm_knng_datastore_path);

  static auto     &knng   = pm_knng.get_knng();
  static pstore_t &pstore = pm_knng.get_point_store();

  comm.cout0() << "Opened datastore" << std::endl;

  // Get connected components
  cls::id_t num_components = 0;  // Number of CCs
  std::vector<uint64_t>
      component_sizes;  // Total size of each CC (on all ranks)
  std::vector<std::vector<cls::id_t>> local_component_members;  // Local CC size
  std::vector<cls::id_t> local_lcc_members;  // Local point ids for each CC
  uint64_t               lcc_size = 0;       // Largest connected component size
  {
    ygm::utility::timer cc_timer;

    ygm::container::disjoint_set<cls::id_t> disjoint_set(comm);
    for (const auto &[source, neighbors] : knng) {
      for (const auto &neighbor : neighbors) {
        disjoint_set.async_union(source, neighbor.id);
      }
    }
    comm.barrier();
    comm.cout0() << "  Time to read points and add to disjoint set (s): "
                 << cc_timer.elapsed() << std::endl;  // debug

    // Get connected components
    get_connected_components(disjoint_set, component_sizes,
                             local_component_members, local_lcc_members);
    num_components = component_sizes.size();
    lcc_size       = ygm::sum(local_lcc_members.size(), comm);

    // Write component sizes to file
    if (!opt.component_size_dump_file.empty()) {
      step_timer.reset();

      // Print component sizes
      if (comm.rank() == 0) {
        std::cout << "  Write component sizes to "
                  << opt.component_size_dump_file << std::endl;
        std::filesystem::path parent_dir =
            opt.component_size_dump_file.parent_path();
        std::filesystem::create_directories(parent_dir);
        std::ofstream ofs(opt.component_size_dump_file);
        if (!ofs.is_open()) {
          comm.cerr0() << "Cannot open file: " << opt.component_size_dump_file
                       << std::endl;
        } else {
          for (const auto &size : component_sizes) {
            ofs << size << "\n";
          }
          ofs.flush();
          ofs.close();
        }
      }

      comm.barrier();
      comm.cout0() << "  Writing CC sizes to file took (s): "
                   << step_timer.elapsed() << std::endl;
    }

    comm.cout0() << "CC took (s): " << cc_timer.elapsed() << std::endl;
    comm.cout0() << "#of CCs: " << num_components << std::endl;
  }

  /* Generate bridge edges */

  // YGM map of (component id i, component id j) -> best bridge edge (point
  // id, point id, distance)
  // Default value for initialization has max distance
  ygm::container::map<std::pair<cls::id_t, cls::id_t>,
                      std::tuple<cls::id_t, cls::id_t, cls::distance_t>>
      bridge_edge_map(
          comm,
          std::make_tuple(0, 0, std::numeric_limits<cls::distance_t>::max()));
  {
    comm.cout0() << "Finding bridge edges" << std::endl;
    ygm::utility::timer connect_timer;

    // Sample from the LCC if needed
    if ((0 <= opt.lcc_sample_fraction) & (opt.lcc_sample_fraction < 1.0) &
        (lcc_size > comm.size())) {
      ygm::utility::timer sample_timer;
      comm.cout0() << "Sampling from the LCC. Sample fraction: "
                   << opt.lcc_sample_fraction << std::endl;

      uint64_t num_samples =
          static_cast<uint64_t>(std::ceil(opt.lcc_sample_fraction * lcc_size));
      if (num_samples <= 0) {
        num_samples = 1;
      }
      comm.cout0() << "Number of samples: " << num_samples << std::endl;

      sample_largest_connected_component(comm, local_lcc_members, num_samples);

      comm.cout0() << "Time to sample LCC (s): " << sample_timer.elapsed()
                   << std::endl;
    }

    // Get the points in the (possibly sampled) LCC from pstore
    step_timer.reset();
    std::vector<std::pair<cls::id_t, point_t>> lcc_points =
        get_lcc_point_data(comm, pstore, local_lcc_members);
    comm.cout0() << "Got the LCC points from pointstore. Time (s): "
                 << step_timer.elapsed() << std::endl;  // debug
    // {
    //   lcc_points.clear();

    //   step_timer.reset();
    //   for (cls::id_t &point_id : local_lcc_members) {
    //     // Go get the point from pstore
    //     comm.async(
    //         dist_pm_knng_t::get_owner(point_id, comm.size()),
    //         [](cls::id_t point_id) {
    //           auto point = pstore.find(point_id);
    //           if (point == pstore.end()) {
    //             std::cerr << "Cannot find point for ID: " << point_id
    //                       << std::endl;
    //             std::abort();
    //           }

    //           point_t point_data(point->second.size());
    //           std::copy(point->second.begin(), point->second.end(),
    //                     point_data.begin());
    //           lcc_points.push_back({point_id, point_data});

    //           // Note, any of the below variants yields bad_alloc:
    //           // point_t point_data = point->second;
    //           // point_t point_data(point->second);
    //           // lcc_points.push_back({point_id, point->second})
    //         },
    //         point_id);
    //   }
    //   comm.barrier();
    //   local_lcc_members.clear();

    //   comm.cout0() << "Got the LCC points from pointstore. Time (s): "
    //                << step_timer.elapsed() << std::endl;  // debug
    // }

    // Randomly choose a single rep from each component that's not the LCC
    // Do this instead of using the disjoint set reps so that it's actually
    // random and go get the actual point data from pstore
    step_timer.reset();
    std::vector<std::pair<cls::id_t, point_t>> reps =
        get_small_component_random_reps(comm, pstore, local_component_members,
                                        component_sizes);
    comm.cout0() << "Got random rep for each component. Time (s): "
                 << step_timer.elapsed() << std::endl;  // debug

    // {
    //   ygm::utility::timer rep_timer;

    //   comm.cout0() << "Got random rep for each component. Time (s): "
    //                << step_timer.elapsed() << std::endl;  // debug

    //   reps.clear();
    //   reps.resize(num_components - 1);

    //   uint64_t seed;
    //   if (comm.rank0()) {
    //     seed = std::random_device{}();
    //   }

    //   std::vector<uint64_t> local_sizes;
    //   std::vector<uint64_t> prefix_sums;
    //   local_sizes.reserve(num_components - 1);
    //   prefix_sums.reserve(num_components - 1);

    //   for (cls::id_t i = 0; i < num_components - 1; ++i) {
    //     local_sizes.push_back(local_component_members[i].size());
    //     prefix_sums.push_back(local_component_members[i].size());
    //   }

    //   // Broadcast the same random seed and get the prefix sums for each
    //   // component
    //   MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, comm.get_mpi_comm());
    //   MPI_Exscan(MPI_IN_PLACE, prefix_sums.data(), num_components - 1,
    //              MPI_UINT64_T, MPI_SUM, comm.get_mpi_comm());
    //   if (comm.rank() == 0) {
    //     std::fill(prefix_sums.begin(), prefix_sums.end(), 0);
    //   }

    //   // Generate the random representative for each connected component
    //   std::default_random_engine random(seed);

    //   for (cls::id_t i = 0; i < num_components - 1; i++) {
    //     cls::id_t chosen_idx = std::uniform_int_distribution<ssize_t>{
    //         0, static_cast<long>(component_sizes[i] - 1)}(random);

    //     // If the chosen point has point id stored on this rank
    //     if ((prefix_sums[i] <= chosen_idx) &
    //         (chosen_idx < prefix_sums[i] + local_sizes[i])) {
    //       uint64_t local_idx = chosen_idx - prefix_sums[i];

    //       // Go get the point from pstore
    //       cls::id_t point_id = local_component_members[i].at(local_idx);
    //       comm.async(
    //           dist_pm_knng_t::get_owner(point_id, comm.size()),
    //           [](cls::id_t point_id, cls::id_t component_id) {
    //             auto point = pstore.find(point_id);
    //             if (point == pstore.end()) {
    //               std::cerr << "Cannot find point for ID: " << point_id
    //                         << std::endl;
    //               std::abort();
    //             }

    //             // Set the rep on this rank
    //             point_t point_data(point->second.size());
    //             std::copy(point->second.begin(), point->second.end(),
    //                       point_data.begin());
    //             std::pair<cls::id_t, point_t> rep =
    //                 std::make_pair(point_id, point_data);
    //             reps[component_id] = rep;
    //           },
    //           point_id, i);
    //     }
    //   }
    //   comm.barrier();

    //   // Broadcast the rep points to all ranks
    //   std::pair<cls::id_t, point_t> default_value;
    //   for (cls::id_t i = 0; i < num_components - 1; i++) {
    //     if (reps[i] != default_value) {
    //       comm.async_bcast(
    //           [](auto rep, cls::id_t component_id) {
    //             reps[component_id] = rep;
    //           },
    //           reps[i], i);
    //     }
    //   }
    //   comm.barrier();

    //   comm.cout0() << "Got random rep for each component. Time (s): "
    //                << step_timer.elapsed() << std::endl;  // debug
    // }

    auto dist_function = [](const auto &seq1, const auto &seq2) {
      return anarci_substitution_distance(seq1, seq2);
    };

    // Get the best bridge edge from each small component rep to the
    // (sampled) LCC
    step_timer.reset();
    cls::id_t lcc_component_id =
        num_components - 1;  // Give the LLC the last component id
    bridge_edges_lcc_to_small(bridge_edge_map, reps, lcc_points, dist_function,
                              lcc_component_id);
    comm.cout0() << "Done getting bridge edges between small components and "
                    "LCC. Time (s): "
                 << step_timer.elapsed() << std::endl;
    // {
    //   cls::id_t lcc_component_id =
    //       num_components - 1;  // Give the LLC the last component id
    //   for (cls::id_t i = 0; i < reps.size() - 1; i++) {
    //     // Find the best bridge edge between each small component rep
    //     // and the (sampled) LCC
    //     cls::distance_t min_dist =
    //     std::numeric_limits<cls::distance_t>::max();

    //     cls::id_t best_point_id_lcc = 0;
    //     cls::id_t point_id_i        = reps[i].first;
    //     point_t   point_i           = reps[i].second;

    //     for (auto &lcc_point : lcc_points) {
    //       auto d = anarci_substitution_distance(point_i, lcc_point.second);
    //       if (d < min_dist) {
    //         min_dist          = d;
    //         best_point_id_lcc = lcc_point.first;
    //       }
    //     }

    //     // See if the local best bridge edge is better than the global one
    //     // stored for this pair of small component and lcc. Always store
    //     // component pairs as (smaller id, larger id)
    //     std::pair<cls::id_t, cls::id_t> component_pair{i, lcc_component_id};
    //     std::tuple<cls::id_t, cls::id_t, cls::distance_t>
    //         local_best_bridge_edge{point_id_i, best_point_id_lcc, min_dist};
    //     bridge_edge_map.async_visit(
    //         component_pair,
    //         [](const std::pair<cls::id_t, cls::id_t> &pair_name,
    //            std::tuple<cls::id_t, cls::id_t, cls::distance_t>
    //                &best_bridge_edge,
    //            const std::tuple<cls::id_t, cls::id_t, cls::distance_t>
    //                &new_bridge_edge) {
    //           if (std::get<2>(new_bridge_edge) <
    //               std::get<2>(best_bridge_edge)) {
    //             best_bridge_edge = new_bridge_edge;
    //           }
    //         },
    //         local_best_bridge_edge);
    //   }
    //   comm.barrier();

    //   comm.cout0() << "Done getting bridge edges between small components and
    //   "
    //                   "LCC. Time (s): "
    //                << step_timer.elapsed() << std::endl;
    // }
    // step_timer.reset();

    if (!opt.skip_small_component_bridges) {
      step_timer.reset();
      bridge_edges_small_to_small(bridge_edge_map, reps, dist_function);

      // // Make a local copy of weight of the best bridge edges to the LCC
      // std::vector<cls::distance_t> dist_to_lcc_vector(num_components - 1);
      // for (const auto &[component_pair, bridge_edge] : bridge_edge_map) {
      //   cls::id_t small_component_id              = component_pair.first;
      //   dist_to_lcc_vector.at(small_component_id) = std::get<2>(bridge_edge);
      // }
      // comm.barrier();
      // MPI_Allreduce(MPI_IN_PLACE, dist_to_lcc_vector.data(), num_components,
      //               mpi_distance_type(), MPI_MAX, comm.get_mpi_comm());

      // comm.cout0()
      //     << "Created local copies of LCC bridge edge weights. Time (s): "
      //     << step_timer.elapsed() << std::endl;
      // // step_timer.reset();  // debug

      // // Loop over every ordered pair of small components and check the rep
      // // bridge edge - only store these best bridge edges if they have
      // // distance
      // // <= the better of the two small component bridge edges to the LCC
      // for (cls::id_t i = 0; i < num_components - 1; i++) {
      //   for (cls::id_t j = 0; j < num_components - 1; j++) {
      //     if (i == j) {
      //       continue;
      //     }

      //     cls::id_t point_id_i = reps[i].first;
      //     point_t   point_i    = reps[i].second;
      //     cls::id_t point_id_j = reps[j].first;
      //     point_t   point_j    = reps[j].second;

      //     cls::distance_t rep_dist =
      //         anarci_substitution_distance(point_i, point_j);

      //     // See if the local best bridge edge is better than the global one
      //     // stored for this component pair - Only consider this bridge edge
      //     // if has distance smaller than at least one of the LCC bridge
      //     // edges for this component pair
      //     if ((rep_dist <= dist_to_lcc_vector[i]) |
      //         (rep_dist <= dist_to_lcc_vector[j])) {
      //       std::pair<cls::id_t, cls::id_t> component_pair{std::min(i, j),
      //                                                      std::max(i, j)};
      //       std::tuple<cls::id_t, cls::id_t, cls::distance_t> bridge_edge{
      //           point_id_i, point_id_j, rep_dist};
      //       bridge_edge_map.async_insert(component_pair, bridge_edge);
      //     }
      //   }
      // }
      // comm.barrier();

      comm.cout0() << "Done getting bridge edges between all small "
                      "components. Time (s): "
                   << step_timer.elapsed() << std::endl;
    }

    comm.cout0() << "Number of bridge edges: " << bridge_edge_map.size()
                 << std::endl;
    comm.cout0() << "Finding bridge edges took (s): " << connect_timer.elapsed()
                 << std::endl;
  }

  // // Write outputs
  // step_timer.reset();
  // // Add the bridge edges found to the knng
  // for (auto &[component_pair, bridge_edge] : bridge_edge_map) {
  //   comm.async(
  //       dist_pm_knng_t::get_owner(std::get<0>(bridge_edge),
  //       comm.size()),
  //       [](cls::id_t a, cls::id_t b, cls::distance_t dist) {
  //         knng[a].emplace_back(b, dist);
  //       },
  //       std::get<0>(bridge_edge), std::get<1>(bridge_edge),
  //       std::get<2>(bridge_edge));
  // }
  // comm.barrier();
  // comm.cout0() << "Time to write output (s): " << step_timer.elapsed()
  //              << std::endl;

  comm.cout0() << "Entire algorithm took (s): " << root_timer.elapsed()
               << std::endl;

  return EXIT_SUCCESS;
}
