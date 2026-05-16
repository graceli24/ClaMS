// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#define CLAMS_USE_SALTATLAS
#define METALL_DISABLE_CONCURRENCY
#define METALL_DISABLE_OBJECT_CACHE

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <saltatlas/neo_dnnd/mpi.hpp>
#include <saltatlas/neo_dnnd/neo_dnnd.hpp>
#include <ygm/comm.hpp>

#include "build_knng.hpp"

using id_t       = clams::id_t;
using fe_t       = clams::fe_t;
using dist_t     = clams::distance_t;
using neo_dnnd_t = saltatlas::neo_dnnd<id_t, fe_t, dist_t>;

auto build_knng(const clams::option_t                    &opt,
                const std::vector<std::filesystem::path> &paths,
                saltatlas::mpi::communicator             &comm) {
  neo_dnnd_t dnnd(
      saltatlas::distance::distance_function<typename neo_dnnd_t::point_type,
                                             dist_t>(opt.distance_name),
      comm, opt.verbose);
  comm.barrier();

  comm.cout0() << "\n<<Read Points>>" << std::endl;
  dnnd.load_points(paths.begin(), paths.end(), opt.point_file_format);

  comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
  auto knng = dnnd.build(opt.index_k, opt.r, opt.delta, 0.2, opt.batch_size);
  return knng;
}

int main(int argc, char **argv) {
  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    saltatlas::mpi::communicator comm;
    const int                    mpi_rank = comm.rank();
    const int                    mpi_size = comm.size();

    if (provided < MPI_THREAD_FUNNELED) {
      comm.cerr0()
          << "The threading support level is lesser than that demanded."
          << std::endl;
      comm.abort();
    }

    clams::option_t opt;
    bool            help{false};
    if (!parse_options(argc, argv, opt, help)) {
      comm.cerr0() << "Invalid option" << std::endl;
      clams::usage(argv[0], comm.cerr0());
      return 0;
    }
    if (help) {
      clams::usage(argv[0], comm.cout0());
      return 0;
    }
    show_options(opt, comm.cout0());
    comm.barrier();

    const auto point_files =
        saltatlas::utility::find_file_paths(opt.point_file_names);

    ygm::utility::timer neo_dnnd_const_timer;
    // Build kNNG using neo_dnnd
    auto neo_knng = build_knng(opt, point_files, comm);
    comm.barrier();
    comm.cout0() << "\nNEO-DNND kNNG construction took (s)\t"
                 << neo_dnnd_const_timer.elapsed() << std::endl;

    // Build DNND index
    {
      comm.cout0() << "\n<<Construct DNND PM Datastore>>" << std::endl;
      ygm::comm ygm_comm(comm.comm());

      ygm::utility::timer                  dnnd_knng_const_timer;
      static clams::dnnd_t::knn_index_type dnnd_init_knng;
      {
        dnnd_init_knng.reserve(neo_knng.size());
        for (const auto &pair : neo_knng) {
          const auto src = pair.first;
          ygm_comm.async(
              clams::dnnd_t::get_owner(src, ygm_comm.size()),
              [](auto, const id_t sid, const auto &neighbors) {
                dnnd_init_knng.reserve_neighbors(sid, neighbors.size());
                for (const auto nb : neighbors) {
                  dnnd_init_knng.insert(sid, nb);
                }
              },
              src, pair.second);
        }
        ygm_comm.barrier();
        neo_knng.clear();
      }
      comm.cout0() << "Preparing init kNNG for DNND took (s)\t"
                   << dnnd_knng_const_timer.elapsed() << std::endl;

      ygm::utility::timer dnnd_data_const_timer;
      {
        clams::dnnd_t g(saltatlas::create_only, opt.scratchpath, ygm_comm,
                        std::random_device{}(), opt.verbose);
        g.load_points(point_files.begin(), point_files.end(),
                      opt.point_file_format);
        g.build(saltatlas::distance::convert_to_distance_id(opt.distance_name),
                opt.index_k, dnnd_init_knng, opt.r, opt.delta, false, 0.1);
      }
      comm.cout0() << "\nConstructing DNND PM datastore took (s)\t"
                   << dnnd_data_const_timer.elapsed() << std::endl;
      comm.barrier();

      if (opt.scratchpath != opt.datastorepath && !opt.datastorepath.empty()) {
        comm.cout0() << "Copying DNND PM datastore to " << opt.datastorepath
                     << std::endl;
        ygm::utility::timer copy_timer;
        clams::dnnd_t::copy(opt.scratchpath, opt.datastorepath, ygm_comm);
        comm.cout0() << "Copy took (s): " << copy_timer.elapsed() << std::endl;
      }
    }
  }
  ::MPI_Finalize();

  return 0;
}
