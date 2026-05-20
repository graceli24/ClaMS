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

#include <boost/range/adaptor/sliced.hpp>
#include "build_knng.hpp"

using id_t       = clams::id_t;
using fe_t       = clams::fe_t;
using dist_t     = clams::distance_t;
using neo_dnnd_t = saltatlas::neo_dnnd<id_t, fe_t, dist_t>;

std::pair<std::vector<id_t>, std::vector<neo_dnnd_t::point_type>>
read_points_neo_dnnd(saltatlas::mpi::communicator             &comm,
                     const std::vector<std::filesystem::path> &paths) {
  std::vector<std::string> str_paths;
  for (const std::filesystem::path &path : paths) {
    str_paths.push_back(path.string());
    // comm.cout0() << "Input file: " << path.string() << std::endl;  //
    // debug
  }

  ygm::comm            ygm_comm(comm.comm());
  ygm::io::line_parser input_line_parser(ygm_comm, str_paths);

  std::vector<id_t>                   id_vector;
  std::vector<neo_dnnd_t::point_type> point_vector;

  auto parse_line_lambda = [&id_vector,
                            &point_vector](const std::string &line) {
    bool              first = true;
    id_t              id;
    std::vector<fe_t> feature;
    std::string       buf;
    for (std::stringstream ss(line); ss >> buf;) {
      if (first) {
        id    = saltatlas::detail::str_cast<id_t>(buf);
        first = false;
      }
      feature.push_back(saltatlas::detail::str_cast<fe_t>(buf));
    }

    id_vector.push_back(id);
    point_vector.push_back(feature);
  };
  input_line_parser.for_all(parse_line_lambda);
  ygm_comm.barrier();

  return std::make_pair(id_vector, point_vector);
}

auto build_knng(const clams::option_t                    &opt,
                const std::vector<std::filesystem::path> &paths,
                saltatlas::mpi::communicator             &comm) {
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

        // Regular distance function on the pair of points with the first entry
        // (which is the point id) removed
        d += base_dist_function(p0.last(size - 1), p0.last(size - 1));

        // Add some small perturbation from hash function on the point ids to
        // as a tiebreaker to the distance
        std::size_t hash_value = 0;
        if (p0.front() < p1.front()) {
          boost::hash_combine(hash_value, p0.front());
          boost::hash_combine(hash_value, p1.front());
        } else {
          boost::hash_combine(hash_value, p1.front());
          boost::hash_combine(hash_value, p0.front());
        }
        d += static_cast<dist_t>(hash_value) /
             static_cast<dist_t>(std::numeric_limits<size_t>::max());

        return d;
      };

  // neo_dnnd_t dnnd(
  //     saltatlas::distance::distance_function<typename neo_dnnd_t::point_type,
  //                                            dist_t>(opt.distance_name),
  //     comm, opt.verbose);
  neo_dnnd_t dnnd(tiebreaker_distance_lambda, comm, opt.verbose);
  comm.barrier();

  comm.cout0() << "\n<<Read Points>>" << std::endl;
  {
    std::pair<std::vector<id_t>, std::vector<neo_dnnd_t::point_type>>
        read_data = read_points_neo_dnnd(comm, paths);
    dnnd.add_points(read_data.first.begin(), read_data.first.end(),
                    read_data.second.begin(), read_data.second.end());
    comm.barrier();
  }
  // dnnd.load_points(paths.begin(), paths.end(), opt.point_file_format);

  comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
  // 0.2 is fraction of feature vectors to duplicate
  // auto knng = dnnd.build(opt.index_k, opt.r, opt.delta, 0.2, opt.batch_size);
  auto knng = dnnd.build(opt.index_k, opt.r, opt.delta, 0, opt.batch_size);
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
        {
          auto line_parser = [](const std::string &line) {
            id_t                               id;
            saltatlas::pm_feature_vector<fe_t> feature;
            std::string                        buf;
            bool                               first = true;
            for (std::stringstream ss(line); ss >> buf;) {
              if (first) {
                id    = saltatlas::detail::str_cast<id_t>(buf);
                first = false;
              }
              feature.push_back(saltatlas::detail::str_cast<fe_t>(buf));
            }

            return std::make_pair(id, feature);
          };
          g.load_points(point_files.begin(), point_files.end(), line_parser);
          // g.load_points(point_files.begin(), point_files.end(),
          //               opt.point_file_format);
          comm.barrier();
        }

        {
          const saltatlas::distance::distance_function_type<
              std::span<const fe_t>, dist_t>
              base_dist_function =
                  saltatlas::distance::distance_function<std::span<const fe_t>,
                                                         dist_t>(
                      opt.distance_name);
          static const auto dnnd_tiebreaker_distance_lambda =
              [&base_dist_function](
                  const saltatlas::pm_feature_vector<fe_t> &p0,
                  const saltatlas::pm_feature_vector<fe_t> &p1) {
                assert(p0.size() == p1.size());

                // Calculate the distance
                dist_t d = 0;

                // Regular distance function on the pair of points with the
                // first entry (which is the point id) removed
                std::span<const fe_t> p0_span = std::span(p0).subspan(1);
                std::span<const fe_t> p1_span = std::span(p1).subspan(1);
                // auto p0_feature = p0 | boost::adaptors::sliced(1,
                // p0.size()); std::span<fe_t> p0_slice_view(p0.begin() + 1,
                // p0.end()); std::span<fe_t> p1_slice_view(p1.begin() + 1,
                // p1.end()); d += base_dist_function(p0_slice_view,
                // p1_slice_view);
                d += base_dist_function(p0_span, p1_span);

                // Add some small perturbation from hash function on the
                // point ids to as a tiebreaker to the distance
                std::size_t hash_value = 0;
                if (p0.at(0) < p1.at(0)) {
                  boost::hash_combine(hash_value, p0.at(0));
                  boost::hash_combine(hash_value, p1.at(1));
                } else {
                  boost::hash_combine(hash_value, p1.at(1));
                  boost::hash_combine(hash_value, p0.at(0));
                }
                d += static_cast<dist_t>(hash_value) /
                     static_cast<dist_t>(std::numeric_limits<size_t>::max());

                return d;
              };

          g.build(dnnd_tiebreaker_distance_lambda, opt.index_k, dnnd_init_knng,
                  opt.r, opt.delta, false, 0.1);
        }
        // g.build(saltatlas::distance::convert_to_distance_id(opt.distance_name),
        //         opt.index_k, dnnd_init_knng, opt.r, opt.delta, false, 0.1);
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
