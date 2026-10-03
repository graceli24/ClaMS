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

#include "anarci_common.hpp"

namespace cls = clams;

using id_t       = cls::id_t;
using fe_t       = amino_acid_t;
using dist_t     = cls::distance_t;
using neo_dnnd_t = saltatlas::neo_dnnd<id_t, fe_t, dist_t>;

struct option_t {
  int                                index_k        = 20;
  double                             r              = 0.5;
  double                             replicate_rate = 0.0;
  double                             delta          = 0.0001;
  std::vector<std::filesystem::path> point_file_names{};
  std::filesystem::path              scratchpath = "/dev/shm/dnnd_datastore";
  std::filesystem::path              datastorepath{};
  std::filesystem::path              backup_datastorepath{};
  std::size_t                        batch_size   = 1ULL << 23;
  int                                num_threads  = -1;
  bool                               verbose      = false;
  std::vector<std::string>           region_names = {"full"};
};

inline bool parse_options(int argc, char **argv, option_t &opt, bool &help) {
  opt.point_file_names.clear();
  std::string provided_region_option = "";
  help                               = false;

  int n;
  while ((n = ::getopt(argc, argv, "k:r:R:d:s:o:b:T:p:vh")) != -1) {
    switch (n) {
      case 'k':
        opt.index_k = std::stoi(optarg);
        break;

      case 'r':
        opt.r = std::stod(optarg);
        break;

      case 'R':
        opt.replicate_rate = std::stod(optarg);
        break;

      case 'd':
        opt.delta = std::stod(optarg);
        break;

      case 's':
        opt.scratchpath = std::filesystem::path(optarg);
        break;

      case 'o':
        opt.datastorepath = std::filesystem::path(optarg);
        break;

      case 'b':
        opt.batch_size = std::stoul(optarg);
        break;

      case 'T':
        opt.num_threads = std::stoi(optarg);
        break;

      case 'p':
        provided_region_option = optarg;
        break;

      case 'v':
        opt.verbose = true;
        break;

      case 'h':
        help = true;
        return true;

      default:
        return false;
    }
  }

  for (int index = optind; index < argc; index++) {
    opt.point_file_names.emplace_back(argv[index]);
  }

  if (opt.index_k <= 0) {
    return false;
  }

  if (opt.replicate_rate < 0.0 || opt.replicate_rate > 1.0) {
    return false;
  }

  if (opt.point_file_names.empty()) {
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

template <typename cout_type>
inline void usage(std::string_view exe_name, cout_type &cout) {
  cout << "Usage: " << exe_name << " [options] point_file1 [point_file2 ...]"
       << std::endl;
  cout << "Options:" << std::endl;
  cout << "  -k <int>    kNNG k parameter (required)" << std::endl;
  cout << "  -r <float>  NN-Descent r parameter (default: 0.5)" << std::endl;
  cout << "  -R <float>  Replication rate in [0, 1] (default: 0.0)"
       << std::endl;
  cout << "  -d <float>  NN-Descent delta parameter (default: 0.001)"
       << std::endl;
  cout << "  -o <path>   If specified, copy the PM datastore from the "
          "scratchpad to to this path."
       << std::endl;
  cout << "  -s <path>   Path to datastore scratchpad. Hold a PM datastore "
          "during computation (default: /dev/shm/dnnd_datastore). Recommended "
          "to be on a fast storage, e.g., tmpfs (/dev/shm)."
       << std::endl;
  cout << "  -b <int>    Batch size (default: 1^23)" << std::endl;
  cout << "  -T <int>    Number of threads for NEO-DNND (optional)"
       << std::endl;
  cout << "  -p <str>    Region names. Regions of the antibody ANARCI sequence "
          "to use. Options are 'full', 'cdrs', 'cdr{i}' where i is 1 2 or 3, "
          "or a comma-separated list of positions (default: full)"
       << std::endl;
  cout << "  -v          Verbose mode" << std::endl;
  cout << "  -h          Show this message" << std::endl;
}

template <typename cout_type>
inline void show_options(const option_t &opt, cout_type &cout) {
  cout << "Options:" << std::endl;
  cout << "  k: " << opt.index_k << std::endl;
  cout << "  r: " << opt.r << std::endl;
  cout << "  replicate rate: " << opt.replicate_rate << std::endl;
  cout << "  delta: " << opt.delta << std::endl;
  cout << "  datastore scratch path: " << opt.scratchpath << std::endl;
  cout << "  datastore path: " << opt.datastorepath << std::endl;
  cout << "  batch size: " << opt.batch_size << std::endl;
  cout << "  verbose: " << opt.verbose << std::endl;
  std::stringstream ss;
  ss << "  region names: ";
  for (auto &region : opt.region_names) {
    ss << region << " ";
  }
  cout << ss.str() << std::endl;
}

template <typename id_type, typename feature_element_type,
          typename distance_type>
inline void show_config(ygm::comm &comm) {
  comm.welcome();
}

int main(int argc, char **argv) {
  int provided;
  ::MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  {
    saltatlas::mpi::communicator comm;

    if (provided < MPI_THREAD_FUNNELED) {
      comm.cerr0()
          << "The threading support level is lesser than that demanded."
          << std::endl;
      comm.abort();
    }

    option_t opt;
    bool     help{false};
    if (!parse_options(argc, argv, opt, help)) {
      comm.cerr0() << "Invalid option" << std::endl;
      usage(argv[0], comm.cerr0());
      return 0;
    }
    if (help) {
      usage(argv[0], comm.cout0());
      return 0;
    }
    show_options(opt, comm.cout0());

    // Set the number of threads per rank
    if (opt.num_threads > 0) {
      comm.cout0() << "Setting #of OpenMP threads to " << opt.num_threads
                   << std::endl;
      saltatlas::utility::omp::set_num_threads(opt.num_threads);
    } else if (std::getenv("OMP_NUM_THREADS")) {
      comm.cout0() << "Using OMP_NUM_THREADS=" << std::getenv("OMP_NUM_THREADS")
                   << std::endl;
    } else {
      comm.cout0() << "Setting #of OpenMP threads to " << 1 << std::endl;
      saltatlas::utility::omp::set_num_threads(1);
    }
    comm.barrier();

    const auto point_files =
        saltatlas::utility::find_file_paths(opt.point_file_names);
    if (point_files.empty()) {
      comm.cerr0() << "No point files found" << std::endl;
      return 1;
    }

    ygm::utility::timer read_data_timer;

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

    comm.cout0() << "Positions used in distance calculation: "
                 << positions_in_calc_ss.str() << std::endl;

    // Build kNNG using neo_dnnd
    // std::function<dist_t(const neo_dnnd_sequence_t &,
    //                      const neo_dnnd_sequence_t &)>
    //     anarci_distance = anarci_substitution_distance<neo_dnnd_sequence_t>;
    static auto anarci_distance = [](const auto &seq1, const auto &seq2) {
      return anarci_substitution_distance(seq1, seq2);
    };
    neo_dnnd_t neo_dnnd(anarci_distance, comm, opt.verbose);

    comm.cout0() << "\n<<Read Points>>" << std::endl;
    {
      // Get all input paths as strings for YGM line parser
      const auto paths =
          saltatlas::utility::find_file_paths(opt.point_file_names);
      std::vector<std::string> str_paths;
      for (const std::filesystem::path &path : paths) {
        str_paths.push_back(path.string());
      }

      std::vector<id_t>       id_vector;
      std::vector<sequence_t> point_vector;

      {
        ygm::comm            ygm_comm(comm.comm());
        ygm::io::line_parser input_line_parser(ygm_comm, str_paths);

        auto parse_id_sequence_line_lambda = [&id_vector, &point_vector](
                                                 const std::string &line) {
          size_t pos = line.find_first_of(" \t");

          // Extract and parse the integer for sequence id
          id_t id = std::stoull(line.substr(0, pos));

          // Extract the JSON string (skip leading whitespace)
          size_t      json_start  = line.find_first_not_of(" \t", pos);
          std::string anarci_json = line.substr(json_start);

          sequence_t anarci_sequence = parse_anarci_json_to_saltatlas_sequence(
              anarci_json, idxs_for_distance_calculation, id, false,
              blosum62_char_to_int_converter);
          // use missing end character = false

          id_vector.push_back(id);
          point_vector.push_back(anarci_sequence);
        };
        input_line_parser.for_all(parse_id_sequence_line_lambda);
        ygm_comm.barrier();
      }

      neo_dnnd.add_points(id_vector.begin(), id_vector.end(),
                          point_vector.begin(), point_vector.end());
    }
    comm.barrier();
    comm.cout0() << "\nIngesting points took (s)\t" << read_data_timer.elapsed()
                 << std::endl;

    ygm::utility::timer neo_dnnd_const_timer;
    comm.cout0() << "\n<<kNNG Construction>>" << std::endl;
    auto knng = neo_dnnd.build(opt.index_k, opt.r, opt.delta,
                               opt.replicate_rate, opt.batch_size);
    comm.barrier();
    comm.cout0() << "\nNEO-DNND kNNG construction took (s)\t"
                 << neo_dnnd_const_timer.elapsed() << std::endl;

    ygm::utility::timer pm_data_const_timer;
    {
      comm.cout0() << "\n<<Construct distributed PM kNNG Datastore>>"
                   << std::endl;

      cls::dist_pm_knng_t pm_knng(comm.comm());
      pm_knng.create(opt.scratchpath);
      ygm::comm ygm_comm(comm.comm());

      static auto &ref_knng = pm_knng.get_knng();
      for (const auto &pair : knng) {
        const auto src = pair.first;
        ygm_comm.async(
            cls::dist_pm_knng_t::get_owner(src, ygm_comm.size()),
            [](const id_t sid, const auto &neighbors) {
              auto &dst = ref_knng[sid];
              dst.reserve(neighbors.size());
              for (const auto &neighbor : neighbors) {
                dst.emplace_back(neighbor);
              }
            },
            src, pair.second);
      }
      knng.clear();
      ygm_comm.barrier();

      static auto &ref_pstore = pm_knng.get_point_store();
      for (const auto &point : neo_dnnd.local_points()) {
        const auto       &id = point.first;
        const auto       &fv = point.second;
        std::vector<fe_t> fv_vec(fv.begin(), fv.end());
        ygm_comm.async(
            cls::dist_pm_knng_t::get_owner(id, comm.size()),
            [](const auto &pid, const auto &fv) {
              const auto [it, inserted] =
                  ref_pstore.try_emplace(pid, fv.begin(), fv.end());
              if (!inserted) {
                std::cerr << "Warning: Duplicate point with id " << pid
                          << " found. Ignoring the duplicate." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
            },
            id, fv_vec);
      }
      ygm_comm.barrier();
    }
    comm.barrier();
    comm.cout0() << "\nConstructing distributed PM kNNG datastore took (s)\t"
                 << pm_data_const_timer.elapsed() << std::endl;

    if (opt.scratchpath != opt.datastorepath && !opt.datastorepath.empty()) {
      comm.cout0() << "Copying PM datastore to " << opt.datastorepath
                   << std::endl;
      metall::logger::set_log_level(metall::logger::level_filter::critical);
      ygm::utility::timer copy_timer;
      const auto          ret = cls::dist_pm_knng_t::copy(
          opt.scratchpath, opt.datastorepath, comm.comm());
      if (!ret) {
        comm.cerr0() << "Failed to copy PM datastore" << std::endl;
        comm.abort();
      }
      comm.cout0() << "Copy took (s): " << copy_timer.elapsed() << std::endl;

      if (opt.datastorepath != opt.backup_datastorepath &&
          !opt.backup_datastorepath.empty()) {
        comm.cout0() << "Making a backup of DNND PM datastore at "
                     << opt.backup_datastorepath << std::endl;
        metall::logger::set_log_level(metall::logger::level_filter::critical);
        ygm::utility::timer backup_timer;
        const auto          backup_ret = clams::dist_pm_knng_t::copy(
            opt.scratchpath, opt.backup_datastorepath, comm.comm());
        if (!backup_ret) {
          comm.cerr0() << "Failed to copy PM datastore" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        comm.cout0() << "Backup copy took (s): " << backup_timer.elapsed()
                     << std::endl;
      }
    }
    comm.barrier();
  }
  ::MPI_Finalize();

  return 0;
}
