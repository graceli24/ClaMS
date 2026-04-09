// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <metall/utility/metall_mpi_adaptor.hpp>
#include <ygm/comm.hpp>

#include "../common.hpp"

class dist_cc {
 private:
  using self_t = dist_cc;

 public:
  using cc_table_t = boost::unordered_flat_map<clams::id_t, clams::id_t>;

  dist_cc(ygm::comm &world, const clams::pm_knng_t &knng)
      : m_world(world), m_knng(knng) {}

  const cc_table_t &run_cc() {
    m_world.cout0() << "Running weakly connected components algorithm."
                    << std::endl;
    {
      const auto uknng = priv_make_undirected_graph();
      m_world.cout0() << "Converted to undirected knng with "
                      << ygm::sum(uknng.num_points(), m_world) << " points and "
                      << ygm::sum(uknng.count_all_neighbors(), m_world)
                      << " edges." << std::endl;
      priv_run_cc(uknng);
    }
    m_world.cout0() << "Connected components algorithm finished." << std::endl;

    return get_cc_table();
  }

  // Each rank has its own CC table
  const cc_table_t &get_cc_table() const { return m_cc_table; }

  // Only rank 0 has result
  std::unordered_map<clams::id_t, size_t> count_cc_size() {
    return priv_count_cc_size();
  }

 private:
  struct cc_visitor {
    template <typename comm_t>
    void operator()(comm_t *comm, auto this_ptr, const clams::id_t &pid,
                    const clams::distance_t &cc_id) {
      auto &cc_table = this_ptr->m_cc_table;
      if (cc_table.count(pid) == 0) {
        comm->cerr() << "CC table does not contain point ID: " << pid
                     << std::endl;
        MPI_Abort(comm->get_mpi_comm(), EXIT_FAILURE);
      }
      if (cc_table.at(pid) <= cc_id) {
        return;
      }
      cc_table.at(pid) = cc_id;
      assert(this_ptr->m_ref_knng.has_value());
      auto &knng = this_ptr->m_ref_knng.value();
      for (auto nitr = knng.neighbors_begin(pid),
                nend = knng.neighbors_end(pid);
           nitr != nend; ++nitr) {
        comm->async(clams::dnnd_t::get_owner(nitr->id, comm->size()),
                    cc_visitor{}, this_ptr->m_self, nitr->id, cc_id);
      }
    }
  };

  clams::pm_knng_t priv_make_undirected_graph() {
    static clams::pm_knng_t uknng;
    uknng.reset();
    uknng.reserve(m_knng.num_points());
    m_world.cf_barrier();

    // Make knng undirected
    for (auto pitr = m_knng.points_begin(), pend = m_knng.points_end();
         pitr != pend; ++pitr) {
      const auto &source = pitr->first;
      for (auto nitr = m_knng.neighbors_begin(source),
                nend = m_knng.neighbors_end(source);
           nitr != nend; ++nitr) {
        const auto &neighbor = *nitr;
        m_world.async(
            clams::dnnd_t::get_owner(neighbor.id, m_world.size()),
            [](clams::id_t src, clams::neighbor_t ngbr) {
              std::swap(src, ngbr.id);
              uknng.insert(src, ngbr);
            },
            source, neighbor);
      }
    }
    m_world.barrier();

    uknng.merge(const_cast<clams::pm_knng_t &>(m_knng));
    for (auto pitr = uknng.points_begin(), pend = uknng.points_end();
         pitr != pend; ++pitr) {
      const auto &source = pitr->first;
      uknng.sort_and_remove_duplicate_neighbors(source);
    }

    return std::move(uknng);
  }

  void priv_run_cc(const clams::pm_knng_t &knng) {
    m_ref_knng = knng;

    m_world.cout0() << "Init CC table" << std::endl;
    // Init CC table
    m_cc_table.clear();
    m_cc_table.reserve(knng.num_points());
    for (auto pitr = knng.points_begin(), pend = knng.points_end();
         pitr != pend; ++pitr) {
      const auto &source = pitr->first;
      m_cc_table[source] = source;  // Initialize CC table
    }
    m_world.cf_barrier();

    m_world.cout0() << "Start CC algorithm" << std::endl;

    // Launch CC algorithm
    for (auto pitr = knng.points_begin(), pend = knng.points_end();
         pitr != pend; ++pitr) {
      const auto &source = pitr->first;
      const auto  cc_id  = m_cc_table.at(source);
      if (cc_id != source) {
        // Already processed
        continue;
      }
      for (auto nitr = knng.neighbors_begin(source),
                nend = knng.neighbors_end(source);
           nitr != nend; ++nitr) {
        const auto &neighbor = *nitr;
        m_world.async(clams::dnnd_t::get_owner(neighbor.id, m_world.size()),
                      cc_visitor{}, m_self, neighbor.id, cc_id);
      }
    }
    m_world.barrier();
    m_world.cout0() << "CC algorithm finished" << std::endl;
  }

  std::unordered_map<clams::id_t, size_t> priv_count_cc_size() {
    std::unordered_map<clams::id_t, size_t> l_cc_size_table;
    for (const auto &cc : m_cc_table) {
      const auto cc_id = cc.second;
      if (l_cc_size_table.count(cc_id) == 0) {
        l_cc_size_table[cc_id] = 0;
      }
      ++l_cc_size_table[cc_id];
    }

    static std::unordered_map<clams::id_t, size_t> g_cc_size_table;
    g_cc_size_table.clear();
    m_world.cf_barrier();

    m_world.async(
        0,
        [](const auto &table) {
          for (const auto &entry : table) {
            const auto cc_id = entry.first;
            const auto size  = entry.second;
            if (g_cc_size_table.count(cc_id) == 0) {
              g_cc_size_table[cc_id] = 0;
            }
            g_cc_size_table[cc_id] += size;
          }
        },
        l_cc_size_table);
    m_world.barrier();

    return std::move(g_cc_size_table);
  }

  ygm::comm              &m_world;
  const clams::pm_knng_t &m_knng;  // Original KNNG
  cc_table_t              m_cc_table{};
  ygm::ygm_ptr<self_t>    m_self{this};
  // KNNG used for CC, could be undirected or directed
  std::optional<clams::pm_knng_t> m_ref_knng;
};
