
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "anarci_common.hpp"

/* Show usage */
void show_help() {
  std::cout
      << "<<Usage>>\n"
         "assign_sequence_ids -n file_of_num_sequences_per_file input_dir\n"
         "  -h Show help.\n"
      << std::endl;
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  // Input directory
  std::filesystem::path input_dir;

  // Optional arguments
  std::string num_of_sequences_file = "";
  bool        single_file           = false;

  // Read the arguments if available
  int opt_char;
  while ((opt_char = getopt(argc, argv, "hn:i:")) != -1) {
    switch (opt_char) {
      case 'n':
        num_of_sequences_file = optarg;
        break;
      case 's':
        single_file = true;
        break;
      case 'h':
        show_help();
        return 1;
      default:
        std::cerr << "Unrecognized option: " << static_cast<char>(optopt)
                  << ", ignore." << std::endl;
        show_help();
    }
  }

  // Input directory is non-optional argument
  input_dir = argv[optind];
  if (!std::filesystem::exists(input_dir)) {
    world.cout0() << "Error: Input directory does not exist" << std::endl;
    world.cout0() << "Provided directory: " << input_dir << std::endl;
    return 1;
  }

  if (!std::filesystem::exists(num_of_sequences_file)) {
    world.cout0() << "Provided -n does not exist alone: "
                  << num_of_sequences_file << std::endl;
    num_of_sequences_file = input_dir / num_of_sequences_file;
    world.cout0() << "Concatenating with input directory" << std::endl;
    if (!std::filesystem::exists(num_of_sequences_file)) {
      world.cout0() << "File does not exist: " << num_of_sequences_file
                    << std::endl;
      return 1;
    }
  }

  // Make input directory static string for convinience
  static std::string input_dir_str;
  input_dir_str = input_dir.c_str();

  // Time the script
  ygm::utility::timer step_timer{};

  // Print out the options
  world.cout0() << "Directory with files to assign point ids: " << input_dir
                << std::endl;
  world.cout0() << "File with number of sequences per file: "
                << num_of_sequences_file << std::endl;

  // Create required directories
  std::filesystem::create_directories(input_dir / "anarci_numberings");
  std::filesystem::create_directories(input_dir / "sequences");
  std::filesystem::create_directories(input_dir / "labels");
  std::filesystem::create_directories(input_dir / "other_info");

  // Vector of index number to start numbering sequences
  static std::vector<uint64_t> idx_start;

  // Read the number of sequences per file
  {
    world.cout0() << "Read in the number of sequences per file" << std::endl;

    // Do first pass to get number of lines
    std::ifstream info_ifs(num_of_sequences_file);
    std::string   line;
    uint32_t      line_count = 0;
    std::getline(info_ifs, line);  // skip the first line
    while (std::getline(info_ifs, line)) {
      ++line_count;
    }

    // Read the file again and populate the num sequences in each file
    std::vector<uint32_t> num_sequence_per_file(line_count, 0);

    info_ifs.clear();                  // Clear eofbit or failbit if set
    info_ifs.seekg(0, std::ios::beg);  // Seek to the beginning of the file
    std::string filename;
    uint32_t    num_sequences;
    std::getline(info_ifs, line);  // skip the first line
    while (std::getline(info_ifs, line)) {
      std::stringstream ss(line);
      ss >> filename;
      ss >> num_sequences;
      std::filesystem::path file_path    = input_dir / filename;
      filename                           = file_path.stem().c_str();
      uint32_t file_number               = std::stoi(filename);
      num_sequence_per_file[file_number] = num_sequences;
    }

    // Get cummulative sum of number of sequences in files before so we can
    // index the sequences
    idx_start.resize(num_sequence_per_file.size());

    uint64_t cumsum = 0;
    for (int i = 1; i < num_sequence_per_file.size(); ++i) {
      cumsum += num_sequence_per_file[i - 1];
      idx_start[i] = cumsum;
    }
  }

  // std::cout << "Start idxs" << std::endl;
  // for (int i = 0; i < idx_start.size(); ++i) {
  //   std::cout << i << ", " << idx_start[i] << std::endl;
  // }

  // Put all the files in the input directory into a YGM bag
  ygm::container::bag<std::string> input_file_bag(world);
  {
    world.cout0() << "Putting all input files in bag" << std::endl;
    if (world.rank() == 0) {
      std::filesystem::path dir_to_process =
          input_dir / "sequences_and_anarci_with_germline_labels";
      for (auto &p : std::filesystem::directory_iterator(dir_to_process)) {
        // std::cout << p.path() << '\n';
        std::string filename = p.path().filename().string();
        if (filename.find("csv") != std::string::npos) {
          input_file_bag.async_insert(p.path());
        }
      }
    }
    world.barrier();
    world.cout0() << "Time elapsed (s): " << step_timer.elapsed() << std::endl;
  }

  // Go through the files and assign numbers to sequences
  {
    step_timer.reset();

    static uint64_t num_sequences;
    num_sequences = 0;

    // Lambda to add sequence id by reading through the provided file
    auto process_file_lambda = [](const std::string &filename) {
      std::cout << filename << std::endl;

      // Get file number
      std::filesystem::path file_path   = filename;
      uint32_t              file_number = std::stoi(file_path.stem().c_str());

      // Add sequence ids by reading through the provided file
      std::string anarci_file = input_dir_str + "/anarci_numberings/" +
                                std::to_string(file_number) + ".txt";
      std::string sequence_file =
          input_dir_str + "/sequences/" + std::to_string(file_number) + ".txt";
      std::string label_file =
          input_dir_str + "/labels/" + std::to_string(file_number) + ".txt";
      std::string other_info_file =
          input_dir_str + "/other_info/" + std::to_string(file_number) + ".csv";

      uint64_t      sequence_id = idx_start[file_number];
      std::string   line;
      std::ifstream ifs(filename);
      std::ofstream anarci_ofs(anarci_file);
      std::ofstream sequence_ofs(sequence_file);
      std::ofstream label_ofs(label_file);
      std::ofstream other_info_ofs(other_info_file);

      // ignore first line
      std::getline(ifs, line);

      while (std::getline(ifs, line)) {
        uint32_t    germline_id      = -1;
        std::string sequence         = "";
        std::string filename         = "";
        std::string anarci_numbering = "";
        std::string anarci_status    = "";

        uint16_t          idx = 0;
        std::stringstream ssline(line);
        while (ssline >> std::ws) {
          std::string csv_field;

          if (ssline.peek() == '"') {
            ssline >> std::quoted(csv_field);
            std::string discard;
            std::getline(ssline, discard, ',');
          } else {
            std::getline(ssline, csv_field, ',');
          }

          if (idx == 0) {
            sequence = csv_field;
          } else if (idx == 1) {
            filename = csv_field;
          } else if (idx == 2) {
            germline_id = std::stoi(csv_field);
          } else if (idx == 3) {
            anarci_numbering = csv_field;
            std::replace(anarci_numbering.begin(), anarci_numbering.end(), '\'',
                         '"');
          } else if (idx == 4) {
            anarci_status = csv_field;
          }
          ++idx;
        }

        anarci_ofs << sequence_id << "\t" << anarci_numbering << std::endl;
        sequence_ofs << sequence_id << "\t" << sequence << std::endl;
        label_ofs << sequence_id << "\t" << germline_id << std::endl;
        other_info_ofs << sequence_id << "," << filename << ","
                       << "\"" << anarci_status << "\"" << std::endl;
        ++sequence_id;
        ++num_sequences;
      }
    };
    input_file_bag.for_all(process_file_lambda);
    world.barrier();

    world.cout0() << "Number of files: " << input_file_bag.size() << std::endl;
    world.cout0() << "Number of sequences: " << ygm::sum(num_sequences, world)
                  << std::endl;

    float seconds = step_timer.elapsed();
    world.cout0() << "Time to add sequence ids to files: " << seconds
                  << " sec = " << seconds / 60.0 << " min" << std::endl;
  }

  return 0;
}
