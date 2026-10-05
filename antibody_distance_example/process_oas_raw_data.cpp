
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/map.hpp>
#include <ygm/io/csv_parser.hpp>
#include <ygm/io/multi_output.hpp>
#include <ygm/utility/timer.hpp>

#include <boost/json/src.hpp>

#include "anarci_common.hpp"

using anarci_vector_t = std::vector<amino_acid_char_t>;

/* Show usage */
void show_help() {
  std::cout << "<<Usage>>\n"
               "Required arguments:\n"
               "  -i <path> Path to directory of OAS data files.\n"
               "  -o <path> Path to dump output files.\n"
               "Optional arguments:\n"
               "  -p Output problem sequence to their own file."
               "  -h Show help.\n"
            << std::endl;
}

/* Function to convert tuple of V,D,J gene names into one germline name*/
// A germline is a string of the form "v-gene-name.d-gene-name.j-gene-name"
// where a "." separates each gene name, and if no gene is found "<NA>" is
// used
std::string germline_genes_to_single_string(
    const std::vector<std::string> &germline_genes) {
  std::string germline_name;
  for (int i = 0; i < 3; ++i) {
    std::string gene_name = germline_genes[i];
    if (gene_name == "" || gene_name == " " || gene_name == "NA") {
      gene_name = "<NA>";
    }
    germline_name = germline_name + gene_name;
    if (i <= 1) {
      germline_name = germline_name + ".";
    }
  }
  return germline_name;
}

// The Richardson 2022 study contains all the kymouse sequences in the OAS data
// set These files (from 7 kymouse) have ambiguous V,D,J genes where the v_call,
// d_call, or j_call columns can have entries which list multiple gene alleles
// like IGHD1-1*01,IGHD1-20*01,IGHD1-7*01
// We will ignore these files for now
static std::vector<std::string> files_to_ignore = {
    "Mouse-1_Richardson_2022_1_Heavy_IGHM.csv",
    "Mouse-2_Richardson_2022_1_Heavy_IGHM.csv",
    "Mouse-3_Richardson_2022_1_Heavy_IGHM.csv",
    "Mouse-4_Richardson_2022_1_Heavy_IGHM.csv",
    "Mouse-5_Richardson_2022_1_Heavy_IGHM.csv",
    "Mouse-6_Richardson_2022_1_Heavy_IGHM.csv",
    "Mouse-7_Richardson_2022_1_Heavy_IGHM.csv"};

std::vector<std::string> sequence_region_names = {
    "fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa",
    "cdr1_aa", "cdr2_aa", "cdr3_aa"};

// Function to determine whether or not we filter out this sequence
// Returns a pair (true, "") if the sequence is valid,
// and a pair (false, invalid reason) if the sequence is not valid
std::pair<bool, std::string> is_sequence_valid(
    std::string                        &sequence,
    std::map<std::string, std::string> &sequence_metadata_map) {
  bool              is_valid = true;
  std::stringstream reason;

  // Don't consider sequence that has stop codon *
  if (sequence.find("*") != std::string::npos) {
    is_valid = false;
    reason << "stop codon;";
  }

  // Don't consider sequences that are cdr3 region
  if (sequence_metadata_map["cdr3_aa"] == "") {
    is_valid = false;
    reason << "missing cdr3_aa;";
  }

  // // Don't consider sequences that are missing any of the key regions
  // // CDR 1 to 3 and FWR 1 to 4
  // for (const std::string &region_name : sequence_region_names) {
  //   if (sequence_metadata_map[region_name] == "") {
  //     is_valid = false;
  //     reason << "missing " << region_name << ";";
  //   }
  // }

  // // Don't consider sequences with CDR3 region > 37 residues
  // if (sequence_metadata_map["cdr3_aa"].length() > 37) {
  //   bool is_valid = false;
  //   reason << "CDR3 longer than 37;";
  // }

  return std::make_pair(is_valid, reason.str());
}

// Function to parse the metadata line at the start of each OAS file
boost::json::object parse_metadata_line(const std::string &metadata_line) {
  // Metadata line is not a valid json line and needs to be fixed first
  std::string fixed;
  fixed.reserve(metadata_line.size());

  // Replace "" with " everywhere
  // Also ignore the first and last characters, which are extra "
  for (size_t i = 1; i < metadata_line.size() - 1; ++i) {
    // If we see two consecutive double quotes, replace with one
    if (metadata_line[i] == '"' && i + 1 < metadata_line.size() &&
        metadata_line[i + 1] == '"') {
      fixed += '"';
      ++i;  // Skip the next quote
    } else {
      fixed += metadata_line[i];
    }
  }

  // Check if there are any "NaN" values and repalce with null to make it
  // parsable
  fixed = std::regex_replace(fixed, std::regex(":\\s*NaN"), ": null");

  // Parse the JSON string using boost and access the oject
  try {
    return boost::json::parse(fixed).as_object();
  } catch (...) {
    std::cerr << "Error reading json metadata line: " << fixed << std::endl;
    return boost::json::parse("{}").as_object();
  }
}

// Takes the json object of the metadata for an OAS file and
// determines if it matches a healthy human
bool is_healthy_human(boost::json::object &metadata_obj) {
  if (metadata_obj["Species"].as_string() != "human") {
    return false;
  }
  if (metadata_obj["Disease"].as_string() != "None") {
    return false;
  }
  if (metadata_obj["Vaccine"].as_string() != "None") {
    return false;
  }

  return true;
}

// Get the clade from the V gene string
// Each V gene string starts with IG*V#,
// where * can be H, L, K for the chain type (H for heavy, L for lambda light
// chain, K for kappa light chain)
// and # is some integer
std::string clade_from_v_gene(const std::string &v_gene) {
  if (v_gene.size() < 4) {
    std::cerr << "Error! When determining clade, fewer than 4 characters in "
                 "the provided V gene: "
              << v_gene << std::endl;
    return v_gene;
  }

  std::string clade = v_gene.substr(0, 4);
  uint16_t    i     = 4;
  // Append all consecutive digits after the first 4 characters
  while (i < v_gene.size() &&
         std::isdigit(static_cast<unsigned char>(v_gene[i]))) {
    clade += v_gene[i];
    ++i;
  }
  return clade;
}

// Convert a value in a boost json object to a string for printing
std::string boost_value_to_str(boost::json::value &boost_value) {
  std::string converted_value;

  // Convert value to string based on its type
  if (boost_value.is_string()) {
    converted_value = boost_value.as_string();
  } else if (boost_value.is_int64()) {
    converted_value = std::to_string(boost_value.as_int64());
  } else if (boost_value.is_double()) {
    converted_value = std::to_string(boost_value.as_double());
  } else if (boost_value.is_bool()) {
    converted_value = boost_value.as_bool() ? "true" : "false";
  } else if (boost_value.is_null()) {
    converted_value = "null";
  } else {
    // For arrays or objects, you could serialize them if needed
    converted_value = boost::json::serialize(boost_value);
  }

  return converted_value;
}

int main(int argc, char **argv) {
  // ygm::comm world(&argc, &argv, 524288);
  ygm::comm world(&argc, &argv);

  // Input and output directories
  std::filesystem::path input_dir;
  std::filesystem::path output_dir;
  bool                  write_problem_sequences_to_file = false;

  // Read the arguments if available
  int opt_char;
  while ((opt_char = getopt(argc, argv, "hpi:o:")) != -1) {
    switch (opt_char) {
      case 'i':
        input_dir = std::filesystem::path(optarg);
        break;
      case 'o':
        output_dir = std::filesystem::path(optarg);
        break;
      case 'p':
        write_problem_sequences_to_file = true;
        break;
      case 'h':
        if (world.rank() == 0) {
          show_help();
        }
        return 1;
      default:
        if (world.rank() == 0) {
          std::cerr << "Unrecognized option: " << static_cast<char>(optopt)
                    << ", ignore." << std::endl;
          show_help();
        }
    }
  }

  // Print out the options
  world.cout0("Number of ranks: ", world.size());
  world.cout0("Input directory with OAS files: ", input_dir);
  world.cout0("Output directory: ", output_dir);

  // Time the script
  ygm::utility::timer total_timer{};
  ygm::utility::timer step_timer{};

  // Full anarci position index vector from 0 to 199
  static std::vector<anarci_idx_t> anarci_full_idx_vector;
  for (anarci_idx_t i = 0; i < 200; ++i) {
    anarci_full_idx_vector.push_back(i);
  }

  /*
  Prepare to read and write files
  */

  // YGM bag of input files to read
  ygm::container::bag<std::string> input_file_bag(world);

  {
    // Put all the OAS files in the input directories into a YGM bag
    if (world.rank() == 0) {
      for (auto &p : std::filesystem::directory_iterator(input_dir)) {
        // std::cout << p.path() << '\n';
        std::string filename = p.path().filename().string();
        if (filename.find("csv") != std::string::npos) {
          if (std::find(files_to_ignore.begin(), files_to_ignore.end(),
                        filename) == files_to_ignore.end()) {
            input_file_bag.async_insert(p.path());
          } else {
            std::cout << "Ignoring file: " << filename << std::endl;
          }
        }
      }
    }
    world.barrier();
  }

  /*
  Read through the OAS files and fill YGM map of sequence -> output string
  By populating a YGM map, duplicate sequences will only be populated the
  first time and subsequent instances will not be output
  */

  // YGM map of sequences to print sequence -> output string
  // The sequence is the actual amino acid string or the anarci string
  ygm::container::map<std::string, std::string> sequence_map(world);

  // YGM map of problem sequences in a germline of interest
  // problem sequence --> vector[file name, germline, problem reason]
  ygm::container::map<std::string, std::vector<std::string>>
                  problem_sequence_map(world);
  static uint32_t num_problem_sequences;

  // Track the total number of valid sequences (not unique)
  static uint64_t num_valid_sequences;
  num_valid_sequences = 0;

  {
    // Read the OAS data files and collect the sequences
    // Lambda to read the target column in each file and add to ygm map
    auto process_file_lambda = [&problem_sequence_map,
                                &sequence_map](const std::string &in_filename) {
      std::filesystem::path p(in_filename);
      std::string           short_in_filename = p.filename().string();

      std::ifstream ifs(in_filename);
      std::string   line;

      // Skip the first line (has the metadata)
      if (std::getline(ifs, line)) {
        // Read the csv header (second line) into a vector

        // Column indices for "v_call", "d_call", and "j_call"
        // which are the V,D,J gene names in the OAS dataset
        // and for "sequence_alignment_aa" for the amino acid sequence
        uint32_t v_col_idx = 8;
        uint32_t d_col_idx = 9;
        uint32_t j_col_idx = 10;

        // Column index for the amino acid sequence (aligned to germline)
        uint32_t sequence_col_idx = 13;

        // Col indicies for anarci numbering and status
        uint32_t anarci_numbering_col_idx = 94;
        uint32_t anarci_status_col_idx    = 95;

        // col indices for sequence region names (e.g., cdr1, fwr1)
        // indicies are in the same order as sequence_region_names
        std::vector<uint32_t> region_col_idx(sequence_region_names.size(), 0);

        uint32_t                 idx = 0;
        std::vector<std::string> csv_header;
        std::getline(ifs, line);
        std::stringstream ssline(line);
        while (ssline >> std::ws) {
          std::string germline;
          std::string csv_field;

          if (ssline.peek() == '"') {
            ssline >> std::quoted(csv_field);
            std::string discard;
            std::getline(ssline, discard, ',');
          } else {
            std::getline(ssline, csv_field, ',');
          }

          if (csv_field == "v_call") {
            v_col_idx = idx;
          } else if (csv_field == "d_call") {
            d_col_idx = idx;
          } else if (csv_field == "j_call") {
            j_col_idx = idx;
          } else if (csv_field == "sequence_alignment_aa") {
            sequence_col_idx = idx;
          } else if (csv_field == "ANARCI_numbering") {
            anarci_numbering_col_idx = idx;
          } else if (csv_field == "ANARCI_status") {
            anarci_status_col_idx = idx;
          } else {
            auto it = std::find(sequence_region_names.begin(),
                                sequence_region_names.end(), csv_field);
            if (it != sequence_region_names.end()) {
              uint32_t position =
                  std::distance(sequence_region_names.begin(), it);
              region_col_idx[position] = idx;
            }
          }
          ++idx;
          csv_header.push_back(csv_field);
        }

        uint32_t max_region_col_idx =
            *std::max_element(region_col_idx.begin(), region_col_idx.end());
        uint32_t max_col_idx =
            std::max({v_col_idx, d_col_idx, j_col_idx, sequence_col_idx,
                      anarci_numbering_col_idx, anarci_status_col_idx,
                      max_region_col_idx});

        if (max_col_idx <= csv_header.size()) {
          // As a sanity check, check the names of the V,D,J gene columns
          // They should be v_call, d_call, j_call
          if ((csv_header[v_col_idx] != "v_call") ||
              (csv_header[d_col_idx] != "d_call") ||
              (csv_header[j_col_idx] != "j_call") ||
              (csv_header[sequence_col_idx] != "sequence_alignment_aa") ||
              (csv_header[anarci_numbering_col_idx] != "ANARCI_numbering") ||
              (csv_header[anarci_status_col_idx] != "ANARCI_status")) {
            std::cout << "File " << in_filename << std::endl;
            std::cout
                << "Reading V, D, J genes, sequence, and anarci info from "
                   "incorrect columns: "
                << csv_header[v_col_idx] << ", " << csv_header[d_col_idx]
                << ", " << csv_header[j_col_idx] << ", "
                << csv_header[sequence_col_idx] << ", "
                << csv_header[anarci_numbering_col_idx] << ", "
                << csv_header[anarci_status_col_idx] << std::endl;
          }

          // Read each line of the csv file
          while (std::getline(ifs, line)) {
            std::stringstream ssline(line);

            // Vector for the V,D,J gene names
            std::vector<std::string> germline_genes = {"", "", ""};

            std::string sequence;  // amino acid sequence

            // Map of sequence regions: region_name --> amino acid subsequence
            std::map<std::string, std::string> sequence_region_map;

            // Strings to store anarci_numbering (json object) and anarci
            // status
            std::string anarci_numbering;
            std::string anarci_status;

            uint32_t idx =
                0;  // track which column of the csv file we're reading
            while (ssline >> std::ws) {
              std::string csv_field;
              if (ssline.peek() == '"') {
                ssline >> std::quoted(csv_field);
                std::string discard;
                std::getline(ssline, discard, ',');
              } else {
                std::getline(ssline, csv_field, ',');
              }

              // Populate the V,D,J names in the tuple as we read them
              if (idx == v_col_idx) {
                germline_genes[0] = csv_field;
              } else if (idx == d_col_idx) {
                germline_genes[1] = csv_field;
              } else if (idx == j_col_idx) {
                germline_genes[2] = csv_field;
              } else if (idx == sequence_col_idx) {
                sequence = csv_field;
              } else if (idx == anarci_numbering_col_idx) {
                anarci_numbering = csv_field;
              } else if (idx == anarci_status_col_idx) {
                anarci_status = csv_field;
              } else {
                auto it = std::find(region_col_idx.begin(),
                                    region_col_idx.end(), idx);
                if (it != region_col_idx.end()) {
                  uint32_t position = std::distance(region_col_idx.begin(), it);
                  sequence_region_map[sequence_region_names[position]] =
                      csv_field;
                }
              }
              ++idx;

              // No need to read the rest of the columns after we get V,D,J
              // names and the aa sequence
              if (idx > max_col_idx) {
                break;
              }
            }

            // Convert germline tuple to a single string of the form
            // v_gene.d_gene.j_gene
            std ::string germline =
                germline_genes_to_single_string(germline_genes);

            // Only consider valid sequences (see process_file_common.hpp)
            std::pair<bool, std::string> is_sequence_valid_output =
                is_sequence_valid(sequence, sequence_region_map);

            if (is_sequence_valid_output.first) {
              ++num_valid_sequences;
              // Convert the anarci numbering from json to readable string
              std::replace(anarci_numbering.begin(), anarci_numbering.end(),
                           '\'', '"');
              anarci_vector_t anarci_vector =
                  parse_anarci_json_to_anarci_vector(anarci_numbering,
                                                     anarci_full_idx_vector, 0);
              std::string identifier_str =
                  anarci_vector_to_string(anarci_vector);
              std::replace(anarci_numbering.begin(), anarci_numbering.end(),
                           '"',
                           '\'');  // put back single quotes for printing

              // Ouput line is: "sequence, filename, germline_id,
              // anarci_numbering, anarci_status"
              std::stringstream ss;
              ss << sequence << "," << short_in_filename << ","
                 << germline_to_id_map[germline] << ","
                 << "\"" << anarci_numbering << "\"" << ","
                 << "\"" << anarci_status << "\"";

              sequence_map.async_insert(identifier_str, ss.str());
            }
            // Stick the sequences that are invalid in a map to print out
            else {
              std::vector<std::string> sequence_info{
                  in_filename, germline, is_sequence_valid_output.second};
              problem_sequence_map.async_insert(sequence, sequence_info);
              ++num_problem_sequences;
            }
          }
        } else {
          std::cout << "At least one of v_call, d_call, j_call, and "
                       "sequence_alignment_aa "
                       "columns not found in file: "
                    << in_filename << std::endl;
        }
      } else {
        std::cout << "The file is empty or an error occurred "
                     "while reading the first line of: "
                  << in_filename << std::endl;
      }
    };

    input_file_bag.for_all(process_file_lambda);
    world.barrier();

    world.cout0("Number of OAS files found: ", input_file_bag.size());
    world.cout0(
        "Number of sequences read (unique or not) that belong to the "
        "selected germlines: ",
        ygm::sum(num_valid_sequences, world));
    world.cout0("Time to go through OAS data and extract sequences: ",
                step_timer.elapsed(), " seconds\n");
    step_timer.reset();
  }

  /* Write deduplicated sequences to file(s) */
  // Track the total number of sequences we output
  uint64_t num_sequences_output = 0;
  {
    world.cout0("Writing sequences to file");
    step_timer.reset();

    // Get output file name and open it
    std::string out_filename = "sequences_and_anarci_with_germline_labels/" +
                               std::to_string(world.rank()) + ".csv";
    std::filesystem::path output_sequence_dir =
        output_dir / "sequences_and_anarci_with_germline_labels/";
    if (world.rank() == 0) {
      std::cout << "Writing files containing cluster labels to: "
                << output_sequence_dir.string() << std::endl;
      std::filesystem::create_directories(output_sequence_dir);
    }
    world.barrier();

    std::filesystem::path out_path = output_dir / out_filename;
    std::ofstream         ofs(out_path);
    if (!ofs.is_open()) {
      std::cout << "Error! Unable to open output file: " << out_path.string()
                << std::endl;
    }

    ofs << "sequence,filename,germline_id,anarci_numbering,anarci_status\n";
    for (const auto &[identifier_str, output_str] : sequence_map) {
      ofs << output_str << "\n";
      ++num_sequences_output;
    }
    ofs.flush();
    ofs.close();
    world.barrier();

    world.cout0("Number of unique sequences output: ",
                ygm::sum(num_sequences_output, world));
    world.cout0("Time to write all files (s): ", step_timer.elapsed(), "\n");
  }

  /*
  Print remaining files
  */

  // Print a file with number of sequences per file. This will be used to assign
  // sequence ids later.
  {
    // Gather all the numbers of sequences outputted on rank 0
    int                   num_ranks = world.size();
    std::vector<uint64_t> num_sequences_per_rank;
    if (world.rank() == 0) {
      num_sequences_per_rank.resize(num_ranks, 0);
    }
    MPI_Gather(&num_sequences_output, 1, MPI_UINT64_T,
               num_sequences_per_rank.data(), 1, MPI_UINT64_T, 0,
               world.get_mpi_comm());

    if (world.rank() == 0) {
      std::filesystem::path out_path =
          output_dir / "num_sequences_per_file.tsv";
      std::ofstream ofs(out_path);
      if (!ofs.is_open()) {
        std::cout << "Error! Unable to open output file: " << out_path.string()
                  << std::endl;
      } else {
        ofs << "file_name\tnum_sequences\n";
        for (int i = 0; i < num_sequences_per_rank.size(); ++i) {
          ofs << "sequences_and_anarci_with_germline_labels/" << i << ".csv"
              << "\t" << num_sequences_per_rank.at(i) << "\n";
        }
        ofs.flush();
        ofs.close();
        std::cout << "Wrote file: " << out_path << std::endl;
      }
    }
    world.barrier();
  }

  if (write_problem_sequences_to_file) {
    // YGM multi output for writing to file, append = false
    ygm::io::multi_output mo(world, output_dir.string(), 1024, false);

    // File where each line is
    // "filename \t germline \t problem_sequence \t reason for problem"
    if (problem_sequence_map.size() > 0) {
      if (world.rank() == 0) {
        mo.async_write_line("problem_sequence_info.tsv",
                            "file\tgermline\tsequence\tproblem_reason");
      }
      world.barrier();
      // sequence_info = [filename, germline, anarci_status]
      auto print_problem_sequence_lambda =
          [&mo](const std::string              &sequence,
                const std::vector<std::string> &sequence_info) {
            std::stringstream ss;
            ss << sequence_info[0] << "\t" << sequence_info[1] << "\t"
               << sequence << "\t" << "\"" << sequence_info[2] << "\"";
            mo.async_write_line("problem_sequence_info.tsv", ss.str());
          };
      problem_sequence_map.for_all(print_problem_sequence_lambda);
    }
    world.barrier();

    world.cout0("Number of problem sequences: ", problem_sequence_map.size());
  }
  world.cout0("Time to write all files (s): ", step_timer.elapsed(), "\n");

  world.barrier();
  world.cout0("Total time (s): ", total_timer.elapsed());

  return 0;
}
