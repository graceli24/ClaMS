#define METALL_DISABLE_CONCURRENCY
#define METALL_DISABLE_OBJECT_CACHE

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cereal/types/array.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/set.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/io/multi_output.hpp>
#include <ygm/utility/timer.hpp>

#include "../common.hpp"
#include "distributed_clustering_types.hpp"

using namespace clams::clustering;

/**
 * @brief Sort and process MST edges.
 *
 * @param mst_edge_vector Vector of MST edges (distance, (endpoint1,
 * endpoint2)). Each rank has its own vector of edges and edges should not be
 * duplicated between ranks.
 * @param edge_endpoints_array A YGM array of edge id -> (edge_endpoint1,
 * edge_endpoint2). This array is not filled with values yet, but needs to have
 * size equal to the number of edges when intialized.
 * @param edge_contraction_array An YGM array of edge id ->
 * edge_contraction_info. This array is not filled with values yet, but needs to
 * have size equal to the number of edges when intialized.
 */
void sort_and_process_mst_edges_into_arrays(
    std::vector<std::pair<distance_t, std::pair<id_t, id_t>>> &mst_edge_vector,
    ygm::container::array<std::pair<id_t, id_t>> &edge_endpoints_array,
    ygm::container::array<edge_contraction_info> &edge_contraction_array) {
  ygm::comm &comm = edge_endpoints_array.comm();

  ygm::utility::timer timer;  // debug

  // Create a YGM array with the MST edges
  ygm::container::array<std::pair<distance_t, std::pair<id_t, id_t>>>
      edge_array(comm, mst_edge_vector);
  comm.barrier();

  // Sort the edges
  edge_array.sort();
  comm.barrier();

  comm.cout0() << "Time to put edges into YGM array and sort (s): "
               << timer.elapsed() << std::endl;
  timer.reset();

  // Fill in the edge endpoints and contraction map
  auto visit_sorted_array_lambda =
      [&edge_endpoints_array, &edge_contraction_array](
          const std::size_t                                  &index,
          const std::pair<distance_t, std::pair<id_t, id_t>> &value) {
        // Add this edge to the edge map with its sorted position as its id
        distance_t            distance       = value.first;
        std::pair<id_t, id_t> edge_endpoints = value.second;
        edge_endpoints_array.local_insert(index, edge_endpoints);
        edge_contraction_array.local_insert(
            index,
            edge_contraction_info{.endpoint_supernode_reps = edge_endpoints,
                                  .distance                = distance});
      };
  edge_array.for_all(visit_sorted_array_lambda);
  comm.barrier();

  comm.cout0() << "Time to fill YGM edge arrays (s): " << timer.elapsed()
               << std::endl;
}

/**
 * @brief Fill the initial min_incident_edge_map from the
 edge_endpoints_array
 *
 * @param edge_endpoints_array YGM array of edge id -> (edge_endpoint1,
 * edge_endpoint2)
 * @param min_incident_edge_map An empty YGM map of supernode representative
 ->
 * min incident edge id. Initially, each supernode representative is an
 * original node in the graph.
 */
void fill_init_min_incident_edge_map(
    ygm::container::map<id_t, id_t>              &min_incident_edge_map,
    ygm::container::array<std::pair<id_t, id_t>> &edge_endpoints_array) {
  ygm::comm &comm = edge_endpoints_array.comm();

  static uint32_t num_visits_received;
  num_visits_received = 0;

  static auto check_incident_edge_lambda =
      []([[maybe_unused]] const id_t &supernode_rep, id_t &min_incident_edge_id,
         const id_t &edge_id) {
        ++num_visits_received;
        if (edge_id < min_incident_edge_id) {
          min_incident_edge_id = edge_id;
        }
      };

  ygm::utility::timer timer;  // debug

  // // Fill the initial incidence array from the original MST edges
  // std::unordered_set<id_t> nodes_visited_on_same_rank;
  // uint32_t                 num_visits_on_same_rank = 0;
  // std::unordered_set<id_t> nodes_visited_on_different_rank;
  // uint32_t                 num_visits_on_different_rank = 0;
  // uint32_t                 num_edges                    = 0;
  // for (const auto &[edge_id, edge_endpoints] : edge_endpoints_array) {
  //   ++num_edges;
  //   for (const id_t &endpoint : {edge_endpoints.first,
  //   edge_endpoints.second}) {
  //     int owner = min_incident_edge_map.partitioner.owner(endpoint);
  //     if (owner == comm.rank()) {
  //       min_incident_edge_map.local_visit(endpoint,
  //       check_incident_edge_lambda,
  //                                         edge_id);
  //       ++num_visits_on_same_rank;
  //       nodes_visited_on_same_rank.insert(endpoint);
  //     } else {
  //       min_incident_edge_map.async_visit(endpoint,
  //       check_incident_edge_lambda,
  //                                         edge_id);
  //       ++num_visits_on_different_rank;
  //       nodes_visited_on_different_rank.insert(endpoint);
  //     }
  //   }
  // }
  // comm.barrier();
  // uint32_t num_nodes_visited_on_same_rank =
  // nodes_visited_on_same_rank.size(); uint32_t
  // num_nodes_visited_on_different_rank =
  //     nodes_visited_on_different_rank.size();

  static uint32_t num_edges;
  num_edges = 0;
  auto fill_init_min_incident_edge_map =
      [&min_incident_edge_map](const id_t            &edge_id,
                               std::pair<id_t, id_t> &edge_endpoints) {
        ++num_edges;
        min_incident_edge_map.async_visit(edge_endpoints.first,
                                          check_incident_edge_lambda, edge_id);
        min_incident_edge_map.async_visit(edge_endpoints.second,
                                          check_incident_edge_lambda, edge_id);
      };
  edge_endpoints_array.for_all(fill_init_min_incident_edge_map);
  comm.barrier();

  comm.cout0() << "Number of edges: sum = " << ygm::sum(num_edges, comm)
               << ", max = " << ygm::max(num_edges, comm)
               << ", min = " << ygm::min(num_edges, comm) << std::endl;

  // comm.cout0() << "Num visits on the same rank: sum = "
  //              << ygm::sum(num_visits_on_same_rank, comm)
  //              << ", max = " << ygm::max(num_visits_on_same_rank, comm)
  //              << ", min = " << ygm::min(num_visits_on_same_rank, comm)
  //              << std::endl;

  // comm.cout0() << "Unique nodes visited on the same rank: sum = "
  //              << ygm::sum(num_nodes_visited_on_same_rank, comm)
  //              << ", max = " << ygm::max(num_nodes_visited_on_same_rank,
  //              comm)
  //              << ", min = " << ygm::min(num_nodes_visited_on_same_rank,
  //              comm)
  //              << std::endl;

  // comm.cout0() << "Num visits to a different rank: sum = "
  //              << ygm::sum(num_visits_on_different_rank, comm)
  //              << ", max = " << ygm::max(num_visits_on_different_rank, comm)
  //              << ", min = " << ygm::min(num_visits_on_different_rank, comm)
  //              << std::endl;

  // comm.cout0() << "Unique nodes visited on a different rank: sum = "
  //              << ygm::sum(num_nodes_visited_on_different_rank, comm)
  //              << ", max = "
  //              << ygm::max(num_nodes_visited_on_different_rank, comm)
  //              << ", min = "
  //              << ygm::min(num_nodes_visited_on_different_rank, comm)
  //              << std::endl;

  comm.cout0() << "Number of messages received: sum = "
               << ygm::sum(num_visits_received, comm)
               << ", max = " << ygm::max(num_visits_received, comm)
               << ", min = " << ygm::min(num_visits_received, comm)
               << std::endl;

  comm.cout0() << "Time to fill initial min incident map: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();
}

int main(int argc, char *argv[]) {
  ygm::comm world(&argc, &argv);

  std::filesystem::path mst_edges_path;
  bool                  metall_mst                 = false;
  bool                  use_sequential_dummy_edges = false;

  int opt_char;
  while ((opt_char = getopt(argc, argv, "i:Ms")) != -1) {
    switch (opt_char) {
      case 'i':
        mst_edges_path = std::filesystem::path(optarg);
        break;
      case 'M':
        metall_mst = true;
        break;
      case 's':
        use_sequential_dummy_edges = true;
        break;
    }
  }

  int num_ranks = world.size();
  world.cout0() << "Number of ranks: " << num_ranks << std::endl;

  // Timers
  ygm::utility::timer timer;
  timer.reset();

  // Read MST edges

  // Vector to store mst edges read by this rank
  // Edges stored in the form (distance, endpoint pair (node 1, node2))
  static std::vector<std::pair<distance_t, std::pair<id_t, id_t>>>
      mst_edge_vector;
  mst_edge_vector.clear();
  static id_t local_num_edges_read;
  local_num_edges_read = 0;
  static id_t local_max_point_id;
  local_max_point_id = 0;

  if (!mst_edges_path.empty()) {
    if (metall_mst) {
      world.cout0() << "Attaching MST in Metall datastore" << std::endl;

      metall::manager metall_manager(metall::open_read_only, mst_edges_path);
      auto           *input_mst_edges =
          metall_manager
              .find<clams::weighted_edge_list_t>(metall::unique_instance)
              .first;
      if (!input_mst_edges) {
        world.cout0() << "Failed to find MST edges in Metall datastore at "
                      << mst_edges_path.string() << std::endl;
        std::abort();
      }
      world.cout0() << "Copying MST edges from Metall datastore" << std::endl;

      uint64_t num_edges  = (*input_mst_edges).size();
      uint64_t chunk_size = num_edges / world.size() + 1;
      uint64_t start_idx  = chunk_size * world.rank();
      uint64_t end_idx =
          std::min(chunk_size * (world.rank() + 1) - 1, num_edges - 1);

      mst_edge_vector.clear();
      for (uint64_t i = start_idx; i <= end_idx; ++i) {
        std::pair<distance_t, std::pair<id_t, id_t>> array_item =
            std::make_pair((*input_mst_edges)[i].distance,
                           std::make_pair((*input_mst_edges)[i].ids[0],
                                          (*input_mst_edges)[i].ids[1]));
        mst_edge_vector.push_back(array_item);

        if ((*input_mst_edges)[i].ids[0] > local_max_point_id) {
          local_max_point_id = (*input_mst_edges)[i].ids[0];
        }
        if ((*input_mst_edges)[i].ids[1] > local_max_point_id) {
          local_max_point_id = (*input_mst_edges)[i].ids[1];
        }
      }
      local_num_edges_read = mst_edge_vector.size();

      world.cout0() << "  Time to read edges (s): " << timer.elapsed()
                    << std::endl;
    } else {
      world.cout0() << "Reading MST edges from file and sorting to assign IDs"
                    << std::endl;
      world.cout0() << "Reading edges from: " << mst_edges_path.string()
                    << std::endl;

      std::vector<std::string> input_paths{mst_edges_path.c_str()};
      ygm::io::line_parser     mst_line_parser(world, input_paths);

      mst_edge_vector.clear();

      auto line_parser_lambda = [](const std::string &line) {
        if (std::isdigit(line[0])) {
          try {
            std::stringstream ss(line);
            id_t              node1, node2;
            distance_t        dist;
            ss >> node1 >> node2 >> dist;
            std::pair<distance_t, std::pair<id_t, id_t>> array_item =
                std::make_pair(dist, std::make_pair(node1, node2));
            mst_edge_vector.push_back(array_item);
            if (node1 > local_max_point_id) {
              local_max_point_id = node1;
            }
            if (node2 > local_max_point_id) {
              local_max_point_id = node2;
            }
          } catch (...) {
            std::cout << "Error reading mst line: " << line << std::endl;
          }
        } else {
          std::cout << "Read comment line in mst file: " << line << std::endl;
        }
      };
      mst_line_parser.for_all(line_parser_lambda);
      world.barrier();

      local_num_edges_read = mst_edge_vector.size();

      world.cout0() << "  Time to read edges (s): " << timer.elapsed()
                    << std::endl;
    }
  }

  id_t total_num_edges_read = ygm::sum(local_num_edges_read, world);
  id_t max_point_id         = ygm::max(local_max_point_id, world);

  // If we didn't read any edges, fill the MST with dummy edges
  if (total_num_edges_read == 0) {
    timer.reset();
    world.cout0() << "Filling mst with dummy edges" << std::endl;

    uint32_t num_points      = 1000010000;
    uint32_t points_per_rank = num_points / world.size();

    uint32_t start_id = world.rank() * points_per_rank;
    uint32_t end_id   = (world.rank() + 1) * points_per_rank;
    if (world.rank() == world.size() - 1) {
      end_id = num_points;
    }

    for (uint32_t point_id = 0; point_id < num_points; ++point_id) {
      distance_t dist = static_cast<distance_t>(point_id) /
                        static_cast<distance_t>(num_points);

      if (use_sequential_dummy_edges) {
        if (start_id <= point_id & point_id < end_id) {
          mst_edge_vector.push_back(
              std::make_pair(dist, std::make_pair(point_id, point_id + 1)));
        }
      } else {
        if (point_id % world.size() == world.rank()) {
          mst_edge_vector.push_back(
              std::make_pair(dist, std::make_pair(point_id, point_id + 1)));
        }
      }
    }
    world.barrier();

    total_num_edges_read = ygm::sum(mst_edge_vector.size(), world);
    max_point_id         = ygm::max(end_id, world);

    world.cout0() << "Time to make up dummy edges (s): " << timer.elapsed()
                  << std::endl;
  }

  world.cout0() << "Max point id: " << max_point_id << std::endl;

  // Sort and ingest MST edges
  timer.reset();

  // Initialize edge array of basic endpoints
  // and edge contraction array with edge_contraction_info
  ygm::container::array<std::pair<id_t, id_t>> edge_endpoints_array(
      world, total_num_edges_read);
  ygm::container::array<edge_contraction_info> edge_contraction_array(
      world, total_num_edges_read);

  // Sort the MST edges and add them to our edge arrays
  sort_and_process_mst_edges_into_arrays(mst_edge_vector, edge_endpoints_array,
                                         edge_contraction_array);

  world.cout0() << "Number of MST edges: " << edge_endpoints_array.size()
                << std::endl;
  world.cout0() << "Total time to sort edges and fill YGM edge arrays (s): "
                << timer.elapsed() << std::endl;

  if (edge_endpoints_array.size() == 0) {
    world.cout0() << "Error: 0 MST edges" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Incidence map of supernode rep (we know the round, so we drop it) ->
  // vector of all incident edge ids
  ygm::container::map<id_t, id_t> min_incident_edge_map(
      world, std::numeric_limits<id_t>::max());

  timer.reset();

  fill_init_min_incident_edge_map(min_incident_edge_map, edge_endpoints_array);

  world.cout0()
      << "Time to fill min incident edge map by visiting edge endpoints (s): "
      << timer.elapsed() << std::endl;

  return EXIT_SUCCESS;
}