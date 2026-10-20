#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include <cereal/types/array.hpp>

#include <ygm/comm.hpp>
#include <ygm/container/array.hpp>
#include <ygm/container/disjoint_set.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/set.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/utility/timer.hpp>

#include "distributed_clustering_types.hpp"

namespace clams::clustering {

uint32_t LOCAL_CACHE_SIZE_LIMIT              = 100000;
uint32_t LOCAL_COPY_NUM_SUPERNODES_THRESHOLD = 25000;

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
void sort_and_process_mst_edges_into_maps(
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

  comm.cout0() << "Time to fill YGM edge maps (s): " << timer.elapsed()
               << std::endl;
}

// /**
//  * @brief Fill the initial min_incident_edge_map from the
//  edge_endpoints_array
//  *
//  * @param edge_endpoints_array YGM array of edge id -> (edge_endpoint1,
//  * edge_endpoint2)
//  * @param min_incident_edge_map An empty YGM map of supernode representative
//  ->
//  * min incident edge id. Initially, each supernode representative is an
//  * original node in the graph.
//  */
// void fill_init_min_incident_edge_map(
//     ygm::container::map<id_t, id_t>              &min_incident_edge_map,
//     ygm::container::array<std::pair<id_t, id_t>> &edge_endpoints_array,
//     id_t                                          max_point_id) {
//   ygm::comm &comm = edge_endpoints_array.comm();

//   static auto check_incident_edge_lambda =
//       []([[maybe_unused]] const id_t &supernode_rep, id_t
//       &min_incident_edge_id,
//          const id_t &edge_id) {
//         if (edge_id < min_incident_edge_id) {
//           min_incident_edge_id = edge_id;
//         }
//       };

//   ygm::utility::timer timer;  // debug

//   // Fill the initial incidence array from the original MST edges
//   // To avoid bottleneck, do a round robin sending (point_id, edge_id)'s to
//   the
//   // correct rank (the rank of owner of point_id) for visiting
//   // Similar to neo_dnnd priv_partition_points function to add_points

//   num_points_batch_size =
//       SALTATLAS_NEO_DNND_FV_SEND_BATCH_SIZE_BYTE / sizeof(id_t);

//   std::vector<std::vector<id_t>> send_point_ids(comm.size());
//   std::vector<std::vector<id_t>> send_edge_ids(comm.size());
//   std::size_t                    n_sends = 0;
//   int                            owner;

//   for (const &[ edge_id, edge_endpoints ] : edge_endpoints_array) {
//     owner = min_incident_edge_map.partitioner.owner(edge_endpoints.first);
//     send_point_ids[owner].emplace_back(edge_endpoints.first);
//     send_edge_ids[owner].emplace_back(edge_id);

//     owner = min_incident_edge_map.partitioner.owner(edge_endpoints.second);
//     send_point_ids[owner].emplace_back(edge_endpoints.second);
//     send_edge_ids[owner].emplace_back(edge_id);
//     n_sends = n_sends + 2;

//     // Send ids if we've collected too many
//     if (n_sends > num_points_batch_size) {
//       mpi::pair_wise_all_to_all(
//           m_comm.size(), m_comm.rank(), [&](const int pair_rank) {
//             // std::cout << "Loop " << n_loops << ": my_rank = " << my_rank
//             //           << ", pair_rank = " << pair_rank
//             //           << " send_ids.size = " << send_ids[pair_rank].size()
//             //           << ", send_fvs.size = " <<
//             send_fvs[pair_rank].size()
//             //           << ", div 200 = " << send_fvs[pair_rank].size() /
//             200
//             //           << std::endl;  // debug

//             std::vector<id_type> recv_ids;
//             m_comm.sendrecv_arb_size(pair_rank, send_ids[pair_rank],
//             recv_ids);

//             // std::cout << "Loop " << n_loops << ": pair_rank = " <<
//             pair_rank
//             //           << " recv_ids.size = " << recv_ids.size()
//             //           << std::endl;  // debug

//             ids.insert(ids.end(), recv_ids.begin(), recv_ids.end());
//             std::vector<fe_type> recv_fvs;
//             m_comm.sendrecv_arb_size(pair_rank, send_fvs[pair_rank],
//             recv_fvs);

//             // std::cout << "Loop " << n_loops << ": pair_rank = " <<
//             pair_rank
//             //           << " recv_fvs.size = " << recv_fvs.size()
//             //           << std::endl;  // debug

//             for (std::size_t i = 0; i < recv_ids.size(); ++i) {
//               std::vector<fe_type> fv(num_dims());
//               std::memcpy(fv.data(), recv_fvs.data() + i * num_dims(),
//                           num_dims() * sizeof(fe_type));
//               fvs.emplace_back(std::move(fv));
//             }
//           });
//     }
//   }

//   for (; n_sends < m_fv_send_batch_size && iitr != ids_end; ++iitr,
//   ++n_sends) {
//     const auto id = *iitr;
//     send_ids[priv_owner(id)].emplace_back(id);
//   }

//   auto fill_init_min_incident_edge_map =
//       [&min_incident_edge_map](const id_t            &edge_id,
//                                std::pair<id_t, id_t> &edge_endpoints) {
//         min_incident_edge_map.async_visit(edge_endpoints.first,
//                                           check_incident_edge_lambda,
//                                           edge_id);
//         min_incident_edge_map.async_visit(edge_endpoints.second,
//                                           check_incident_edge_lambda,
//                                           edge_id);
//       };
//   edge_endpoints_array.for_all(fill_init_min_incident_edge_map);
//   comm.barrier();

//   comm.cout0() << "Time to fill initial min incident map: " <<
//   timer.elapsed()
//                << std::endl;  // debug
//   timer.reset();

//   // Remove any extra nodes
//   id_t num_points_erased = 0;
//   for (auto &[point_id, edge_id] : min_incident_edge_map) {
//     if (edge_id == std::numeric_limits<id_t>::max()) {
//       min_incident_edge_map.local_erase(point_id);
//       ++num_points_erased;
//     }
//   }
//   comm.barrier();

//   comm.cout0() << "Time to check and delete any extraneous points: "
//                << timer.elapsed() << std::endl;  // debug
//   comm.cout0() << "Number of extraneous points deleted: "
//                << ygm::sum(num_points_erased, comm) << std::endl;  // debug
//   comm.cout0() << "min_incident_edge_map.size = "
//                << min_incident_edge_map.size() << std::endl;
//   timer.reset();
// }

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
void fill_init_min_incident_edge_map_v2(
    ygm::container::map<id_t, id_t>              &min_incident_edge_map,
    ygm::container::array<std::pair<id_t, id_t>> &edge_endpoints_array,
    id_t                                          max_point_id) {
  ygm::comm &comm = edge_endpoints_array.comm();

  static auto check_incident_edge_lambda =
      []([[maybe_unused]] const id_t &supernode_rep, id_t &min_incident_edge_id,
         const id_t &edge_id) {
        if (edge_id < min_incident_edge_id) {
          min_incident_edge_id = edge_id;
        }
      };

  ygm::utility::timer timer;  // debug

  for (id_t id = 0; id <= max_point_id; ++id) {
    int owner = min_incident_edge_map.partitioner.owner(id);
    if (owner == comm.rank()) {
      min_incident_edge_map.local_insert(id, std::numeric_limits<id_t>::max());
    }
  }
  comm.barrier();

  comm.cout0() << "Time to put nodes in min incident map: " << timer.elapsed()
               << std::endl;  // debug
  comm.cout0() << "max point id = " << max_point_id << std::endl;
  comm.cout0() << "min_incident_edge_map.size() = "
               << min_incident_edge_map.size() << std::endl;
  timer.reset();

  // Fill the initial incidence array from the original MST edges
  auto fill_init_min_incident_edge_map =
      [&min_incident_edge_map](const id_t            &edge_id,
                               std::pair<id_t, id_t> &edge_endpoints) {
        min_incident_edge_map.async_visit(edge_endpoints.first,
                                          check_incident_edge_lambda, edge_id);
        min_incident_edge_map.async_visit(edge_endpoints.second,
                                          check_incident_edge_lambda, edge_id);
      };
  edge_endpoints_array.for_all(fill_init_min_incident_edge_map);
  comm.barrier();

  comm.cout0() << "Time to fill initial min incident map: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();

  // Remove any extra nodes
  id_t num_points_erased;
  num_points_erased = 0;
  std::vector<id_t> points_to_erase;
  for (const auto &[point_id, edge_id] : min_incident_edge_map) {
    if (edge_id == std::numeric_limits<id_t>::max()) {
      points_to_erase.push_back(point_id);
      std::cout << comm.rank() << " - " << point_id << std::endl;
      ++num_points_erased;
    }
  }
  comm.barrier();
  for (id_t point_id : points_to_erase) {
    min_incident_edge_map.local_erase(point_id);
  }
  comm.barrier();

  comm.cout0() << "Time to check and delete any extraneous points: "
               << timer.elapsed() << std::endl;  // debug
  comm.cout0() << "Number of extraneous points deleted: "
               << ygm::sum(num_points_erased, comm) << std::endl;  // debug
  // comm.cout0() << "Number of extraneous points deleted: "
  //              << ygm::sum(points_to_erase.size(), comm) << std::endl;  //
  //              debug
  comm.cout0() << "min_incident_edge_map.size = "
               << min_incident_edge_map.size() << std::endl;
  timer.reset();
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

  // std::cout << "Rank " << comm.rank()
  //           << ": num_visits_on_same_rank = " << num_visits_on_same_rank
  //           << " to " << nodes_visited_on_same_rank.size()
  //           << " nodes. num_visits_on_different_rank = "
  //           << num_visits_on_different_rank << " to "
  //           << nodes_visited_on_different_rank.size() << std::endl;

  // auto fill_init_min_incident_edge_map =
  //     [&min_incident_edge_map](const id_t            &edge_id,
  //                              std::pair<id_t, id_t> &edge_endpoints) {
  //       min_incident_edge_map.async_visit(edge_endpoints.first,
  //                                         check_incident_edge_lambda,
  //                                         edge_id);
  //       min_incident_edge_map.async_visit(edge_endpoints.second,
  //                                         check_incident_edge_lambda,
  //                                         edge_id);
  //     };
  // edge_endpoints_array.for_all(fill_init_min_incident_edge_map);
  // comm.barrier();

  comm.cout0() << "Time to fill initial min incident map: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();
}

/**
 * @brief Fill the init_min_incident_edge_array from the edge_endpoints_array
 *
 * @param edge_endpoints_array YGM array of edge id -> (edge_endpoint1,
 * edge_endpoint2)
 * @param min_incident_edge_map An empty YGM map of supernode representative ->
 * min incident edge id. Initially, each supernode representative is an
 * original node in the graph.
 */
void fill_init_min_incident_edge_array(
    ygm::container::array<id_t>                  &min_incident_edge_array,
    ygm::container::array<std::pair<id_t, id_t>> &edge_endpoints_array) {
  ygm::comm &comm = edge_endpoints_array.comm();

  static auto check_incident_edge_lambda =
      []([[maybe_unused]] const id_t &supernode_rep, id_t &min_incident_edge_id,
         const id_t &edge_id) {
        if (edge_id < min_incident_edge_id) {
          min_incident_edge_id = edge_id;
        }
      };

  ygm::utility::timer timer;  // debug

  // Fill the initial incidence array from the original MST edges
  auto fill_init_min_incident_edge_array =
      [&min_incident_edge_array](const id_t            &edge_id,
                                 std::pair<id_t, id_t> &edge_endpoints) {
        min_incident_edge_array.async_visit(
            edge_endpoints.first, check_incident_edge_lambda, edge_id);
        min_incident_edge_array.async_visit(
            edge_endpoints.second, check_incident_edge_lambda, edge_id);
      };
  edge_endpoints_array.for_all(fill_init_min_incident_edge_array);
  comm.barrier();

  comm.cout0() << "Time to fill initial min incident array: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();
}

/**
 * @brief Find all MST edges to contract this round and add to disjoint set
 *
 * @param _round The round of MST contraction we're on.
 * @param min_incident_edge_map YGM map of supernode representative ->
 * min incident edge id.
 * @param edge_contraction_array YGM array of edge id -> edge_contraction_info
 * @param alpha_edge_map_ptr Pointer to YGM map of alpha edge id ->
 * alpha_edge_info. Alpha edges are ones that are not contracted in the first
 * round. As we contract alpha edges, we update their child supernodes in the
 * dendrogram.
 * @param tree_components_djset YGM disjoint_set for getting connected
 * components after edge contraction.
 */
template <typename ygm_container_type>
void contract_edges(
    uint32_t _round, ygm_container_type &min_incident_edge_map,
    ygm::container::array<edge_contraction_info> &edge_contraction_array,
    ygm::ygm_ptr<ygm::container::map<id_t, alpha_edge_info>> alpha_edge_map_ptr,
    ygm::container::disjoint_set<id_t> &tree_components_djset) {
  ygm::comm &comm                      = edge_contraction_array.comm();
  auto       tree_components_djset_ptr = tree_components_djset.get_ygm_ptr();

  static uint32_t round;
  round = _round;

  ygm::utility::timer timer;  // debug

  [[maybe_unused]] static auto add_dendrogram_child_lambda =
      []([[maybe_unused]] const id_t &edge_id, alpha_edge_info &info,
         supernode_t new_child) {
        if (info.dendrogram_children[0] == BLANK_SUPERNODE) {
          info.dendrogram_children[0] = new_child;
        } else {
          info.dendrogram_children[1] = new_child;
        }
      };

  auto contract_edges_lambda =
      [&edge_contraction_array, alpha_edge_map_ptr, tree_components_djset_ptr](
          const id_t &supernode_rep, const id_t &min_incident_edge_id) {
        // The shortest edge incident to each supernode is contracted

        // Add the contracted edge to the disjoint set and set as
        // contracted For an edge between supernodes (rep1, round-1) and
        // (rep2, round-1), we add the rep1-rep2 edge to the disjoint set
        auto add_edge_to_djset_lambda = []([[maybe_unused]] const id_t &edge_id,
                                           edge_contraction_info &edge_info,
                                           auto tree_components_djset_ptr) {
          edge_info.contraction_round = round;
          tree_components_djset_ptr->async_union(
              edge_info.endpoint_supernode_reps.first,
              edge_info.endpoint_supernode_reps.second);
        };
        if (min_incident_edge_id != std::numeric_limits<id_t>::max()) {
          edge_contraction_array.async_visit(min_incident_edge_id,
                                             add_edge_to_djset_lambda,
                                             tree_components_djset_ptr);
        }
        // Set this supernode as the dendrogram child of the min incident edge
        if (round >= 2) {
          alpha_edge_map_ptr->async_visit(
              min_incident_edge_id, add_dendrogram_child_lambda,
              std::make_pair(supernode_rep, round - 1));
        }
      };
  min_incident_edge_map.for_all(contract_edges_lambda);
  comm.barrier();

  comm.cout0() << "Time to contract edges: " << timer.elapsed()
               << std::endl;  // debug

  return;
}

/**
 * @brief Find all MST edges to contract this round and add to disjoint set
 *
 * @param _round The round of MST contraction we're on.
 * @param min_incident_edge_map YGM map of supernode representative ->
 * min incident edge id.
 * @param edge_contraction_array YGM array of edge id -> edge_contraction_info
 * @param alpha_edge_map_ptr Pointer to YGM map of alpha edge id ->
 * alpha_edge_info. Alpha edges are ones that are not contracted in the first
 * round. As we contract alpha edges, we update their child supernodes in the
 * dendrogram.
 * @param tree_components_djset YGM disjoint_set for getting connected
 * components after edge contraction.
 */
void contract_edges_first_round(
    ygm::container::array<id_t>                  &min_incident_edge_array,
    ygm::container::array<edge_contraction_info> &edge_contraction_array,
    ygm::ygm_ptr<ygm::container::map<id_t, alpha_edge_info>> alpha_edge_map_ptr,
    ygm::container::disjoint_set<id_t> &tree_components_djset) {
  ygm::comm &comm                      = edge_contraction_array.comm();
  auto       tree_components_djset_ptr = tree_components_djset.get_ygm_ptr();

  ygm::utility::timer timer;  // debug

  [[maybe_unused]] static auto add_dendrogram_child_lambda =
      []([[maybe_unused]] const id_t &edge_id, alpha_edge_info &info,
         supernode_t new_child) {
        if (info.dendrogram_children[0] == BLANK_SUPERNODE) {
          info.dendrogram_children[0] = new_child;
        } else {
          info.dendrogram_children[1] = new_child;
        }
      };

  auto contract_edges_lambda = [&edge_contraction_array, alpha_edge_map_ptr,
                                tree_components_djset_ptr](
                                   [[maybe_unused]] const id_t &supernode_rep,
                                   const id_t &min_incident_edge_id) {
    // The shortest edge incident to each supernode is contracted

    // Add the contracted edge to the disjoint set and set as
    // contracted For an edge between supernodes (rep1, round-1) and
    // (rep2, round-1), we add the rep1-rep2 edge to the disjoint set
    auto add_edge_to_djset_lambda = []([[maybe_unused]] const id_t &edge_id,
                                       edge_contraction_info       &edge_info,
                                       auto tree_components_djset_ptr) {
      edge_info.contraction_round = 1;
      tree_components_djset_ptr->async_union(
          edge_info.endpoint_supernode_reps.first,
          edge_info.endpoint_supernode_reps.second);
    };
    if (min_incident_edge_id != std::numeric_limits<id_t>::max()) {
      edge_contraction_array.async_visit(min_incident_edge_id,
                                         add_edge_to_djset_lambda,
                                         tree_components_djset_ptr);
    }
  };
  min_incident_edge_array.for_all(contract_edges_lambda);
  comm.barrier();

  comm.cout0() << "Time to contract edges: " << timer.elapsed()
               << std::endl;  // debug

  return;
}

/**
 * @brief Updates the edge contraction map by updating chain parent edge id of
 * contracted edges e by looking up the current supernode they are in and seeing
 * if the supernode's min incident edge has larger id than e. This version
 * directly visits the YGM min_incident_edge_map.
 *
 * @param round The round of MST contraction we're on.
 * @param edge_contraction_array YGM array of edge id -> edge_contraction_info
 * @param min_incident_edge_map YGM map of supernode representative ->
 * min incident edge id.
 */
void update_edge_chain_parent_edge_id(
    uint32_t                                      round,
    ygm::container::array<edge_contraction_info> &edge_contraction_array,
    ygm::container::map<id_t, id_t>              &min_incident_edge_map) {
  ygm::comm &comm                       = edge_contraction_array.comm();
  auto       edge_contraction_array_ptr = edge_contraction_array.get_ygm_ptr();

  ygm::utility::timer timer;  // debug

  static std::unordered_map<id_t, id_t> local_min_incident_edge_map;
  local_min_incident_edge_map.clear();

  static uint32_t number_of_async_visits;
  static uint32_t number_of_visits_saved;
  static uint32_t number_of_update_visits_saved;
  number_of_async_visits        = 0;
  number_of_visits_saved        = 0;  // debug
  number_of_update_visits_saved = 0;

  static std::unordered_set<id_t> supernode_reps_with_visit_sent;
  supernode_reps_with_visit_sent.clear();
  static std::unordered_map<id_t, std::vector<id_t>>
      map_of_edges_awaiting_visit_return;
  map_of_edges_awaiting_visit_return.clear();

  auto update_chain_parents_lambda = [&min_incident_edge_map, round,
                                      edge_contraction_array_ptr](
                                         const id_t            &edge_id,
                                         edge_contraction_info &edge_info) {
    // Only see if we can update an edge's chain parent edge id if not
    // assigned yet (if its 0) and if the edge is already contracted
    if (edge_info.chain_parent_edge_id == 0 &&
        0 < edge_info.contraction_round &&
        edge_info.contraction_round < round) {
      // If we already have the min incident edge id stored locally, then
      // directly try to update the chain parent
      if (local_min_incident_edge_map.find(edge_info.chain_supernode.first) !=
          local_min_incident_edge_map.end()) {
        id_t min_incident_edge_id =
            local_min_incident_edge_map.at(edge_info.chain_supernode.first);
        if (min_incident_edge_id > edge_id) {
          edge_info.chain_parent_edge_id = min_incident_edge_id;
          ++number_of_update_visits_saved;
        }
        ++number_of_visits_saved;
      } else {
        // If we already sent out an async visit call for this chain
        // supernode, then add this edge to the local map and await the
        // returned info to update it later
        if (supernode_reps_with_visit_sent.find(
                edge_info.chain_supernode.first) !=
            supernode_reps_with_visit_sent.end()) {
          map_of_edges_awaiting_visit_return[edge_info.chain_supernode.first]
              .push_back(edge_id);
        }
        // Do an async visit to min incident edge map the first time we
        // encounter this chain supernode locally
        else {
          ++number_of_async_visits;
          supernode_reps_with_visit_sent.insert(
              edge_info.chain_supernode.first);
          min_incident_edge_map.async_visit(
              edge_info.chain_supernode.first,
              []([[maybe_unused]] const id_t &supernode_rep,
                 const id_t &min_incident_edge_id, const id_t &edge_id,
                 auto edge_contraction_array_ptr) {
                // If the supernode's min incident edge id is larger than
                // edge id, we found the chain's parent edge id in the
                // dendrogram
                edge_contraction_array_ptr->async_visit(
                    edge_id,
                    []([[maybe_unused]] const id_t &edge_id,
                       edge_contraction_info       &edge_info,
                       id_t min_incident_edge_id, id_t supernode_rep) {
                      if (min_incident_edge_id > edge_id) {
                        edge_info.chain_parent_edge_id = min_incident_edge_id;
                      }

                      if (local_min_incident_edge_map.size() <
                          LOCAL_CACHE_SIZE_LIMIT) {
                        local_min_incident_edge_map[supernode_rep] =
                            min_incident_edge_id;
                      }
                    },
                    min_incident_edge_id, supernode_rep);
              },
              edge_id, edge_contraction_array_ptr);
        }
      }
    }
  };
  edge_contraction_array.for_all(update_chain_parents_lambda);
  comm.barrier();

  comm.cout0() << "Time to update chain parents part 1: " << timer.elapsed()
               << std::endl;  // debug
  comm.cout0() << "Number of async visits made in part 1: "
               << ygm::sum(number_of_async_visits, comm) << std::endl;  // debug
  comm.cout0() << "Max map_of_edges_awaiting_visit_return.size = "
               << ygm::max(map_of_edges_awaiting_visit_return.size(), comm)
               << std::endl;

  // Go through the edges awaiting return messages and update them. If we
  // didn't get a return message and stored value, then do the required
  // async visits
  for (auto &[supernode_rep, edge_id_vector] :
       map_of_edges_awaiting_visit_return) {
    // If we already have the min incident edge id stored locally, then
    // directly try to update the chain parents
    if (local_min_incident_edge_map.find(supernode_rep) !=
        local_min_incident_edge_map.end()) {
      id_t min_incident_edge_id = local_min_incident_edge_map.at(supernode_rep);
      for (id_t edge_id : edge_id_vector) {
        if (min_incident_edge_id > edge_id) {
          edge_contraction_array.local_visit(
              edge_id,
              []([[maybe_unused]] const id_t &edge_id,
                 edge_contraction_info &edge_info, id_t chain_parent_edge_id) {
                edge_info.chain_parent_edge_id = chain_parent_edge_id;
              },
              min_incident_edge_id);
          ++number_of_update_visits_saved;
        }
        ++number_of_visits_saved;
      }
    } else {
      for (id_t &edge_id : edge_id_vector) {
        // Do an async visit to min incident edge map
        ++number_of_async_visits;
        min_incident_edge_map.async_visit(
            supernode_rep,
            []([[maybe_unused]] const id_t &supernode_rep,
               const id_t &min_incident_edge_id, const id_t &edge_id,
               auto edge_contraction_array_ptr) {
              // If the supernode's min incident edge id is larger than
              // edge id, we found the chain's parent edge id in the
              // dendrogram
              if (min_incident_edge_id > edge_id) {
                edge_contraction_array_ptr->async_visit(
                    edge_id,
                    []([[maybe_unused]] const id_t &edge_id,
                       edge_contraction_info       &edge_info,
                       id_t chain_parent_edge_id, id_t supernode_rep) {
                      edge_info.chain_parent_edge_id = chain_parent_edge_id;

                      if (local_min_incident_edge_map.size() <
                          LOCAL_CACHE_SIZE_LIMIT) {
                        local_min_incident_edge_map[supernode_rep] =
                            chain_parent_edge_id;
                      }
                    },
                    min_incident_edge_id, supernode_rep);
              }
            },
            edge_id, edge_contraction_array_ptr);
      }
    }
  }
  comm.barrier();

  comm.cout0() << "Total time to update chain parents: " << timer.elapsed()
               << std::endl;  // debug
  comm.cout0() << "Number of async visits made: "
               << ygm::sum(number_of_async_visits, comm) << std::endl;  // debug
  comm.cout0() << "Number of saved async visits that resulted in an update: "
               << ygm::sum(number_of_update_visits_saved, comm)
               << std::endl;  // debug
  comm.cout0() << "Number of saved async visits: "
               << ygm::sum(number_of_visits_saved, comm) << std::endl;  // debug
  timer.reset();

  return;
}

/**
 * @brief Updates the edge contraction map by updating the chain parent edge
 * id of contracted edges e by looking up the current supernode they are in
 * and seeing if the supernode's min incident edge has larger id than e. Makes
 * a local copy of the YGM min_incident_edge_map on each rank to avoid
 * bottlenecks. This version should be used instead of
 * update_edge_chain_parent_edge_id when the min_incident_edge_map is small.
 *
 * @param round The round of MST contraction we're on.
 * @param edge_contraction_array YGM array of edge id -> edge_contraction_info
 * @param min_incident_edge_map YGM map of supernode representative ->
 * min incident edge id.
 */
void update_edge_chain_parent_edge_id_from_local_map(
    uint32_t                                      round,
    ygm::container::array<edge_contraction_info> &edge_contraction_array,
    ygm::container::map<id_t, id_t>              &min_incident_edge_map) {
  ygm::comm &comm = edge_contraction_array.comm();

  ygm::utility::timer timer;  // debug

  // Give all ranks a full copy of min_incident_edge_map
  std::unordered_map<id_t, id_t> local_min_incident_edge_map;
  min_incident_edge_map.for_all(
      [&local_min_incident_edge_map](const id_t &supernode_rep,
                                     const id_t &min_incident_edge_id) {
        local_min_incident_edge_map[supernode_rep] = min_incident_edge_id;
      });
  comm.barrier();

  comm.cout0() << "Time to get local entries of min incident edge map: "
               << timer.elapsed() << std::endl;  // debug
  timer.reset();

  static std::unordered_map<id_t, id_t> min_incident_edge_std_map;
  min_incident_edge_std_map.clear();
  comm.async_bcast(
      [](const std::unordered_map<id_t, id_t> &local_map) {
        min_incident_edge_std_map.insert(local_map.begin(), local_map.end());
      },
      local_min_incident_edge_map);
  comm.barrier();
  local_min_incident_edge_map.clear();

  comm.cout0() << "Time to give each rank full copy of min incident edge map: "
               << timer.elapsed() << std::endl;  // debug
  timer.reset();

  // Update the chain parents of edges using local lookup
  auto update_chain_parents_lambda = [&round](
                                         [[maybe_unused]] const id_t &edge_id,
                                         edge_contraction_info &edge_info) {
    // Only see if we can update an edge's chain parent edge id if not
    // assign yet (if its 0) and if the edge is already contracted
    if (edge_info.chain_parent_edge_id == 0 &&
        0 < edge_info.contraction_round &&
        edge_info.contraction_round < round) {
      if (min_incident_edge_std_map.at(edge_info.chain_supernode.first) >
          edge_id) {
        edge_info.chain_parent_edge_id =
            min_incident_edge_std_map.at(edge_info.chain_supernode.first);
      }
    }
  };
  edge_contraction_array.for_all(update_chain_parents_lambda);
  comm.barrier();

  comm.cout0() << "Time to update chain parents: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();

  return;
}

/**
 * @brief Updates edge_contraction_array by updating the supernode endpoints
 * of uncontracted edges and updates the current chain supernode of contracted
 * edges if the chain parent edge id is not yet found. This version directly
 * visits the tree_conmponents_djset.
 *
 * @param _round The round of MST contraction we're on.
 * @param edge_contraction_array YGM array of edge id -> edge_contraction_info
 * @param tree_components_djset YGM disjoint_set of connected components of
 * the contracted tree. We use this to get the new supernode reps.
 */
void update_edge_endpoints_and_chain_supernode(
    uint32_t                                      _round,
    ygm::container::array<edge_contraction_info> &edge_contraction_array,
    ygm::container::disjoint_set<id_t>           &tree_components_djset) {
  static uint32_t round;
  round = _round;

  ygm::comm &comm                       = edge_contraction_array.comm();
  auto       edge_contraction_array_ptr = edge_contraction_array.get_ygm_ptr();

  ygm::utility::timer timer;  // debug

  auto update_edge_endpoints_and_chain_supernode_lambda =
      [&tree_components_djset, edge_contraction_array_ptr](
          const id_t &edge_id, edge_contraction_info &edge_info) {
        // If the edge is not contracted yet, update its supernode endpoints
        if (edge_info.contraction_round == 0) {
          // Update the first supernode endpoint of this edge
          tree_components_djset.async_visit(
              edge_info.endpoint_supernode_reps.first,
              []([[maybe_unused]] const auto &djset_item_data_pair,
                 const id_t &edge_id, auto edge_contraction_array_ptr) {
                edge_contraction_array_ptr->async_visit(
                    edge_id,
                    []([[maybe_unused]] const id_t &edge_id,
                       edge_contraction_info       &edge_info,
                       id_t new_first_endpoint_supernode_rep) {
                      edge_info.endpoint_supernode_reps.first =
                          new_first_endpoint_supernode_rep;
                    },
                    djset_item_data_pair.second.get_parent());
              },
              edge_id, edge_contraction_array_ptr);

          // Update the second supernode endpoint of this edge
          tree_components_djset.async_visit(
              edge_info.endpoint_supernode_reps.second,
              []([[maybe_unused]] const auto &djset_item_data_pair,
                 const id_t &edge_id, auto edge_contraction_array_ptr) {
                edge_contraction_array_ptr->async_visit(
                    edge_id,
                    []([[maybe_unused]] const id_t &edge_id,
                       edge_contraction_info       &edge_info,
                       id_t new_second_endpoint_supernode_rep) {
                      edge_info.endpoint_supernode_reps.second =
                          new_second_endpoint_supernode_rep;
                    },
                    djset_item_data_pair.second.get_parent());
              },
              edge_id, edge_contraction_array_ptr);
        }

        // If an edge is already contracted, see if it already has a known
        // chain parent edge id, or if we should update the chain supernode to
        // find the chain this edge belongs to
        else {
          // If we haven't found a chain_parent_edge_id > edge_id in the
          // dendrogram yet, then update the chain supernode
          if (edge_info.chain_parent_edge_id == 0) {
            // If the edge was contracted this round, then get its first
            // supernode from one of its endpoints
            id_t supernode_rep_to_visit =
                edge_info.endpoint_supernode_reps.first;
            // Otherwise, get the parent supernode of its last supernode
            if (edge_info.contraction_round < round) {
              supernode_rep_to_visit = edge_info.chain_supernode.first;
            }

            // Update the edge's chain supernode
            tree_components_djset.async_visit(
                supernode_rep_to_visit,
                []([[maybe_unused]] const auto &djset_item_data_pair,
                   const id_t edge_id, auto edge_contraction_array_ptr) {
                  edge_contraction_array_ptr->async_visit(
                      edge_id,
                      []([[maybe_unused]] const id_t &edge_id,
                         edge_contraction_info       &edge_info,
                         const supernode_t            new_supernode) {
                        edge_info.chain_supernode = new_supernode;
                      },
                      std::make_pair(djset_item_data_pair.second.get_parent(),
                                     round));
                },
                edge_id, edge_contraction_array_ptr);
          }
        }
      };
  edge_contraction_array.for_all(
      update_edge_endpoints_and_chain_supernode_lambda);
  comm.barrier();

  comm.cout0() << "Time to update edge endpoints and chain supernodes: "
               << timer.elapsed() << std::endl;  // debug
  timer.reset();

  return;
}

/**
 * @brief Updates edge_contraction_array by updating the supernode endpoints
 * of uncontracted edges and updates the current chain supernode of contracted
 * edges if the chain parent edge id is not yet found. Makes a
 * local copy of the tree_components_djset representatives on each rank to
 * avoid bottlenecks. This version should be used instead of
 * update_edge_endpoints_and_chain_supernode when the number of tree
 * components is small.
 *
 * @param _round The round of MST contraction we're on.
 * @param edge_contraction_array YGM array of edge id -> edge_contraction_info
 * @param tree_components_djset YGM disjoint_set of connected components of
 * the contracted tree. We use this to get the new supernode reps.
 */
void update_edge_endpoints_and_chain_supernode_from_local_map(
    uint32_t                                      round,
    ygm::container::array<edge_contraction_info> &edge_contraction_array,
    ygm::container::disjoint_set<id_t>           &tree_components_djset) {
  ygm::comm &comm = edge_contraction_array.comm();

  static std::unordered_map<id_t, id_t> old_to_new_supernode_rep_map;
  static std::unordered_map<id_t, id_t> local_old_to_new_supernode_rep_map;
  old_to_new_supernode_rep_map.clear();
  local_old_to_new_supernode_rep_map.clear();

  ygm::utility::timer timer;  // debug

  // Create a local copy of the full tree_components djset on every rank
  tree_components_djset.for_all([](const id_t &old_supernode_rep,
                                   const id_t &new_supernode_rep) {
    local_old_to_new_supernode_rep_map[old_supernode_rep] = new_supernode_rep;
  });
  comm.barrier();

  comm.cout0() << "Time to get local tree components: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();

  comm.async_bcast(
      [](const std::unordered_map<id_t, id_t> &local_map) {
        old_to_new_supernode_rep_map.insert(local_map.begin(), local_map.end());
      },
      local_old_to_new_supernode_rep_map);
  comm.barrier();

  comm.cout0() << "Time to bcast tree components to all ranks: "
               << timer.elapsed() << std::endl;  // debug
  timer.reset();

  // Update edge info by referencing the contents of the local map
  auto update_edge_endpoints_and_chain_supernode_lambda =
      [round]([[maybe_unused]] const id_t &edge_id,
              edge_contraction_info       &edge_info) {
        // If the edge is not contracted yet, update its supernode
        // endpoints
        if (edge_info.contraction_round == 0) {
          // Update the first supernode endpoint of this edge
          edge_info.endpoint_supernode_reps.first =
              old_to_new_supernode_rep_map.at(
                  edge_info.endpoint_supernode_reps.first);

          // Update the second supernode endpoint of this edge
          edge_info.endpoint_supernode_reps.second =
              old_to_new_supernode_rep_map.at(
                  edge_info.endpoint_supernode_reps.second);
        }
        // If an edge is already contracted, see if it already has a known
        // chain parent edge id, or if we should update the chain supernode to
        // find the chain this edge belongs to
        else {
          // If we haven't found a chain_parent_edge_id > edge_id in the
          // dendrogram yet, then update the chain supernode
          if (edge_info.chain_parent_edge_id == 0) {
            // If the edge was contracted this round, then get its first
            // supernode from one of its endpoints
            id_t old_supernode_rep = edge_info.endpoint_supernode_reps.first;
            // Otherwise, get the parent supernode of its last supernode
            if (edge_info.contraction_round < round) {
              old_supernode_rep = edge_info.chain_supernode.first;
            }

            // Update the edge's chain supernode
            supernode_t new_supernode = std::make_pair(
                old_to_new_supernode_rep_map.at(old_supernode_rep), round);
            edge_info.chain_supernode = new_supernode;
          }
        }
      };
  edge_contraction_array.for_all(
      update_edge_endpoints_and_chain_supernode_lambda);
  comm.barrier();

  comm.cout0() << "Time to update edge endpoints and chain supernodes: "
               << timer.elapsed() << std::endl;  // debug
  timer.reset();

  return;
}

/**
 * @brief Clear the min_incident_edge_map and fill with new supernode reps as
 * keys.
 *
 * @param min_incident_edge_map YGM map of supernode representative ->
 * min incident edge id. We will clear and refill this map.
 * @param tree_components_djset YGM disjoint_set of connected components of
 * the contracted tree. We use this to get the new supernode reps.
 */
void reinit_min_incident_edge_map(
    ygm::container::map<id_t, id_t>    &min_incident_edge_map,
    ygm::container::disjoint_set<id_t> &tree_components_djset) {
  ygm::comm &comm = min_incident_edge_map.comm();

  ygm::utility::timer timer;  // debug

  min_incident_edge_map.clear();
  comm.barrier();

  comm.cout0() << "Time to clear min incident edge map: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();

  auto reinit_min_incident_edge_map_lambda =
      [&min_incident_edge_map]([[maybe_unused]] const id_t &old_supernode_rep,
                               const id_t                  &new_supernode_rep) {
        min_incident_edge_map.async_insert(new_supernode_rep,
                                           std::numeric_limits<id_t>::max());
      };
  tree_components_djset.for_all(reinit_min_incident_edge_map_lambda);
  comm.barrier();

  comm.cout0() << "Time to reinit min incident edge map: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();

  return;
}

/**
 * @brief Fills out the values of the min_incident_edge_map. Assumes that the
 * min_incident_edge_map already has the correct supernode reps as keys, e.g.,
 * by running reinit_min_incident_edge_map first.
 *
 * @param edge_contraction_array YGM map of edge id -> edge contraction info.
 * Assumes that edges already have their supernode endpoints updated.
 * @param min_incident_edge_map YGM map of supernode representative ->
 * min incident edge id. Assumes that the map already has the correct
 * supernode reps as keys.
 */
void update_min_incident_edge_map(
    ygm::container::array<edge_contraction_info> &edge_contraction_array,
    ygm::container::map<id_t, id_t>              &min_incident_edge_map) {
  ygm::comm &comm = edge_contraction_array.comm();

  ygm::utility::timer timer;  // debug

  static auto check_incident_edge_lambda =
      []([[maybe_unused]] const id_t &supernode_rep, id_t &min_incident_edge_id,
         const id_t &edge_id) {
        if (edge_id < min_incident_edge_id) {
          min_incident_edge_id = edge_id;
        }
      };

  auto update_min_incident_edge_map_lambda =
      [&min_incident_edge_map](const id_t            &edge_id,
                               edge_contraction_info &edge_info) {
        // If the edge is not contracted yet
        if (edge_info.contraction_round == 0) {
          min_incident_edge_map.async_visit(
              edge_info.endpoint_supernode_reps.first,
              check_incident_edge_lambda, edge_id);
          min_incident_edge_map.async_visit(
              edge_info.endpoint_supernode_reps.second,
              check_incident_edge_lambda, edge_id);
        }
      };
  edge_contraction_array.for_all(update_min_incident_edge_map_lambda);
  comm.barrier();

  comm.cout0() << "Time to update min incident edge map: " << timer.elapsed()
               << std::endl;  // debug
  timer.reset();
}

}  // namespace clams::clustering