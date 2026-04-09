// ClaMS clustering pipeline
// Copyright (C) 2023–2025 Lawrence Livermore National Security (LLNS), LLC

#define CLAMS_USE_SALTATLAS
#define METALL_DISABLE_CONCURRENCY

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "build_knng.hpp"

using id_t   = clams::id_t;
using fe_t   = clams::fe_t;
using dist_t = clams::distance_t;

int main(int argc, char **argv) {
  ygm::comm comm(&argc, &argv);
  clams::show_config<id_t, fe_t, dist_t>(comm);
  {
    clams::option_t opt;
    bool            help{false};
    if (!clams::parse_options(argc, argv, opt, help)) {
      comm.cerr0() << "Invalid option" << std::endl;
      clams::usage(argv[0], comm.cerr0());
      return 0;
    }
    if (help) {
      clams::usage(argv[0], comm.cout0());
      return 0;
    }
    clams::show_options(opt, comm.cout0());

    {
      clams::dnnd_t g(saltatlas::create_only, opt.scratchpath, comm,
                      std::random_device{}(), opt.verbose);
      {
        comm.cout0() << "\n<<Read Points>>" << std::endl;
        const auto paths =
            saltatlas::utility::find_file_paths(opt.point_file_names);
        ygm::utility::timer point_read_timer;
        g.load_points(paths.begin(), paths.end(), opt.point_file_format);
        comm.cout0() << "\nReading points took (s)\t"
                     << point_read_timer.elapsed() << std::endl;
        comm.cout0() << "#of points\t" << g.num_points() << std::endl;
      }

      size_t index_id{};
      {
        comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
        ygm::utility::timer const_timer;
        index_id = g.build(
            saltatlas::distance::convert_to_distance_id(opt.distance_name),
            opt.index_k, opt.r, opt.delta);
        comm.cout0() << "\nkNNG construction took (s)\t"
                     << const_timer.elapsed() << std::endl;
      }
    }
    comm.cf_barrier();
    if (opt.scratchpath != opt.datastorepath && !opt.datastorepath.empty()) {
      comm.cout0() << "Copying DNND PM datastore to " << opt.datastorepath
                   << std::endl;
      ygm::utility::timer copy_timer;
      clams::dnnd_t::copy(opt.scratchpath, opt.datastorepath, comm);
      comm.cout0() << "Copy took (s): " << copy_timer.elapsed() << std::endl;
    }
  }

  return 0;
}
