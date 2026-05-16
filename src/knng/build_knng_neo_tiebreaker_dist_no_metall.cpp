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
#include <ygm/io/line_parser.hpp>

#include "build_knng.hpp"

using id_t       = clams::id_t;
using fe_t       = clams::fe_t;
using dist_t     = clams::distance_t;
using neo_dnnd_t = saltatlas::neo_dnnd<id_t, fe_t, dist_t>;

template <typename read_feature_t, typename point_t>
std::pair<std::vector<id_t>, std::vector<point_t>> read_points(
    saltatlas::mpi::communicator             &comm,
    const std::vector<std::filesystem::path> &paths) {
  std::vector<std::string> str_paths;
  for (const std::filesystem::path &path : paths) {
    str_paths.push_back(path.string());
    // comm.cout0() << "Input file: " << path.string() << std::endl;  //
    // debug
  }

  ygm::comm            ygm_comm(comm.comm());
  ygm::io::line_parser input_line_parser(ygm_comm, str_paths);

  std::vector<id_t>    id_vector;
  std::vector<point_t> point_vector;

  auto parse_line_lambda = [&id_vector,
                            &point_vector](const std::string &line) {
    bool           first = true;
    id_t           id;
    read_feature_t feature;
    std::string    buf;
    for (std::stringstream ss(line); ss >> buf;) {
      if (first) {
        id    = saltatlas::detail::str_cast<id_t>(buf);
        first = false;
      }
      feature.push_back(saltatlas::detail::str_cast<fe_t>(buf));
    }

    id_vector.push_back(id);
    // point_t converted_feature = feature;
    point_vector.push_back(feature);
  };
  input_line_parser.for_all(parse_line_lambda);
  ygm_comm.barrier();

  return std::make_pair(id_vector, point_vector);
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

    const saltatlas::distance::distance_function_type<
        typename neo_dnnd_t::point_type, dist_t>
        base_dist_function = saltatlas::distance::distance_function<
            typename neo_dnnd_t::point_type, dist_t>(opt.distance_name);
    static const auto tiebreaker_distance_lambda =
        [&base_dist_function](const std::span<fe_t> &p0,
                              const std::span<fe_t> &p1) {
          assert(p0.size() == p1.size());
          uint32_t size = p0.size();

          // Calculate the distance
          dist_t d = 0;

          // Regular distance function on the pair of points with the first
          // entry (which is the point id) removed
          d += base_dist_function(p0.last(size - 1), p0.last(size - 1));

          // Add some small perturbation from hash function on the point ids to
          // as a tiebreaker to the distance
          std::size_t hash_value = 0;
          boost::hash_combine(hash_value, p0.front());
          boost::hash_combine(hash_value, p1.front());
          d += static_cast<dist_t>(hash_value) /
               static_cast<dist_t>(std::numeric_limits<size_t>::max());

          return d;
        };

    neo_dnnd_t dnnd(tiebreaker_distance_lambda, comm, opt.verbose);
    comm.barrier();

    comm.cout0() << "\n<<Read Points>>" << std::endl;
    {
      std::pair<std::vector<id_t>, std::vector<neo_dnnd_t::point_type>>
          read_data = read_points<std::vector<fe_t>, neo_dnnd_t::point_type>(
              comm, point_files);
      dnnd.add_points(read_data.first.begin(), read_data.first.end(),
                      read_data.second.begin(), read_data.second.end());
      comm.barrier();
    }
    // dnnd.load_points(paths.begin(), paths.end(), opt.point_file_format);

    comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
    auto knng = dnnd.build(opt.index_k, opt.r, opt.delta, 0.2, opt.batch_size);
    comm.barrier();

    comm.cout0() << "\nNEO-DNND kNNG construction took (s)\t"
                 << neo_dnnd_const_timer.elapsed() << std::endl;

    ygm::utility::timer dump_data_timer;

    if (!opt.datastorepath.empty()) {
      comm.cout0() << "\nDump KNNG to " << opt.datastorepath << std::endl;
      std::error_code ec;
      std::filesystem::create_directories(opt.datastorepath, ec);
      comm.barrier();
      // dump distance = true since we need it for the mst
      dnnd.dump_graph(knng, opt.datastorepath.string() + "/knng", true);
      comm.barrier();
    }

    comm.cout0() << "\nDumping data took (s)\t" << dump_data_timer.elapsed()
                 << std::endl;
  }
  ::MPI_Finalize();

  return 0;
}
