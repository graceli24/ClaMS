#include <span>

#include <ygm/comm.hpp>
#include <ygm/detail/collective.hpp>
#include <ygm/io/line_parser.hpp>
#include <ygm/utility/timer.hpp>

#include <boost/json/src.hpp>
#include <saltatlas/dnnd/feature_vector.hpp>
#include "../src/common.hpp"

namespace cls = clams;

// ANARCI-related types

// We'll read amino acids in as unsigned chars
using amino_acid_char_t = unsigned char;

// We'll represent amino acids as ints
using amino_acid_t = uint8_t;

// how we'll encode letters not in our alphabet when converting char to
// amino_acid_t
amino_acid_t MISSING_LETTER = std::numeric_limits<amino_acid_t>::max();

// Smith Waterman similarity is a non-negative integer
// We convert this to a float in [0,1] to get a distance
using sim_t = int32_t;

// 20 standard amino acids + "X" for unknown amino acid
// std::string alphabet = "ARNDCQEGHILKMFPSTWYVX";
amino_acid_char_t UNKNOWN_CHAR = 'X';

// Cost parameters for gap penalty for alignment
int GAP_OPEN_COST   = 11;
int GAP_EXTEND_COST = 1;

// // Type for an ANARCI vector of characters
// using anarci_vector_t = saltatlas::feature_vector<amino_acid_char_t>;

// Type for index positions in an ANARCI vector.
// An OAS ANARCI vector has length 200.
using anarci_idx_t = uint16_t;

// Represent a sequence as a saltatlas vector of amino acids
// using sequence_t = saltatlas::feature_vector<amino_acid_t>;
using sequence_t = std::vector<amino_acid_t>;

// Represent a sequence as a span of amino acids
using neo_dnnd_sequence_t = std::span<amino_acid_t>;

/**
 * @brief Vector of canonical position names in an ANARCI numbering from the OAS
 * data set
 *
 * @details Taken from
 * https://github.com/oxpig/kasearch/blob/main/kasearch/canonical_alignment.py
 *
 */
std::vector<std::string> anarci_canonical_numbering = {
    "1 ",   "2 ",   "3 ",   "3A",   "4 ",   "5 ",   "6 ",   "7 ",   "8 ",
    "9 ",   "10 ",  "11 ",  "12 ",  "13 ",  "14 ",  "15 ",  "16 ",  "17 ",
    "18 ",  "19 ",  "20 ",  "21 ",  "22 ",  "23 ",  "24 ",  "25 ",  "26 ",
    "27 ",  "28 ",  "29 ",  "30 ",  "31 ",  "32 ",  "32A",  "32B",  "33C",
    "33B",  "33A",  "33 ",  "34 ",  "35 ",  "36 ",  "37 ",  "38 ",  "39 ",
    "40 ",  "40A",  "41 ",  "42 ",  "43 ",  "44 ",  "44A",  "45 ",  "45A",
    "46 ",  "46A",  "47 ",  "47A",  "48 ",  "48A",  "48B",  "49 ",  "49A",
    "50 ",  "51 ",  "51A",  "52 ",  "53 ",  "54 ",  "55 ",  "56 ",  "57 ",
    "58 ",  "59 ",  "60 ",  "60A",  "60B",  "60C",  "60D",  "61E",  "61D",
    "61C",  "61B",  "61A",  "61 ",  "62 ",  "63 ",  "64 ",  "65 ",  "66 ",
    "67 ",  "67A",  "67B",  "68 ",  "68A",  "68B",  "69 ",  "69A",  "69B",
    "70 ",  "71 ",  "71A",  "71B",  "72 ",  "73 ",  "73A",  "73B",  "74 ",
    "75 ",  "76 ",  "77 ",  "78 ",  "79 ",  "80 ",  "80A",  "81 ",  "81A",
    "81B",  "81C",  "82 ",  "82A",  "83 ",  "83A",  "83B",  "84 ",  "85 ",
    "85A",  "85B",  "85C",  "85D",  "86 ",  "86A",  "87 ",  "88 ",  "89 ",
    "90 ",  "91 ",  "92 ",  "93 ",  "94 ",  "95 ",  "96 ",  "96A",  "97 ",
    "98 ",  "99 ",  "100 ", "101 ", "102 ", "103 ", "104 ", "105 ", "106 ",
    "107 ", "108 ", "109 ", "110 ", "111 ", "111A", "111B", "111C", "111D",
    "111E", "111F", "111G", "111H", "111I", "111J", "111K", "111L", "112L",
    "112K", "112J", "112I", "112H", "112G", "112F", "112E", "112D", "112C",
    "112B", "112A", "112 ", "113 ", "114 ", "115 ", "116 ", "117 ", "118 ",
    "119 ", "119A", "120 ", "121 ", "122 ", "123 ", "124 ", "125 ", "126 ",
    "127 ", "128 "};

/**
 * @brief Map specifying which position numbers correspond to which antibody
 * regions.
 */
std::map<std::string, std::pair<anarci_idx_t, anarci_idx_t>>
    region_numbering_bounds = {
        {"fwr1", std::make_pair(1, 26)},   {"cdr1", std::make_pair(27, 38)},
        {"fwr2", std::make_pair(39, 55)},  {"cdr2", std::make_pair(56, 65)},
        {"fwr3", std::make_pair(66, 104)}, {"cdr3", std::make_pair(105, 117)},
        {"fwr4", std::make_pair(118, 128)}};

/**
 * @brief Uses the canonical position numbering to get a map from position
 * name to an integer index from 0 to 199.
 */
std::map<std::string, anarci_idx_t> get_full_position_to_idx_map() {
  std::map<std::string, anarci_idx_t> position_to_idx_map;
  anarci_idx_t                        idx = 0;
  for (std::string position_str : anarci_canonical_numbering) {
    position_to_idx_map[position_str] = idx;
    ++idx;
  }
  return position_to_idx_map;
}

/**
 * @brief Map from position name in the canonical numbering to an integer index
 * from 0 to 199.
 */
std::map<std::string, anarci_idx_t> full_position_to_idx_map =
    get_full_position_to_idx_map();

/**
 * @brief Given a region name, returns a map from positions names in that region
 * to their integer index in full_position_to_idx_map.
 *
 * @param region_name Region name string. Valid region names are "fwr1", "fwr2",
 * "fwr3", "fwr4", "cdr1", "cdr2", "cdr3".
 *
 * @details The options for region names are "fwr1", "fwr2", "fwr3", "fwr4",
 * "cdr1", "cdr2", "cdr3". If a region name not equal to one of these valid
 * region names is provided, then returns full_position_to_idx_map. The default
 * behavior is to return full_position_to_idx_map.
 */
std::map<std::string, anarci_idx_t> get_position_to_idx_map_from_region_name(
    std::string region_name = "full") {
  std::map<std::string, anarci_idx_t> position_to_idx_map;
  if (region_numbering_bounds.find(region_name) !=
      region_numbering_bounds.end()) {
    for (std::string position_str : anarci_canonical_numbering) {
      int position_int =
          std::stoi(position_str.substr(0, position_str.length() - 1));
      if ((region_numbering_bounds[region_name].first <= position_int) &&
          (position_int <= region_numbering_bounds[region_name].second)) {
        position_to_idx_map[position_str] =
            full_position_to_idx_map[position_str];
      }
    }
  } else {
    position_to_idx_map = full_position_to_idx_map;
  }

  return position_to_idx_map;
}

/**
 * @brief Given a vector of region name(s), returns a map from positions names
 * in those regions to their integer index in full_position_to_idx_map.
 *
 * @param region_names Vector of region name strings. Valid region names are
 * "fwr1", "fwr2", "fwr3", "fwr4", "cdr1", "cdr2", "cdr3".
 *
 * @details The options for region names are "fwr1", "fwr2", "fwr3", "fwr4",
 * "cdr1", "cdr2", "cdr3". If no valid region names are provided, then it
 * returns full_position_to_idx_map. This is the default behavior.
 */
std::map<std::string, anarci_idx_t> get_position_to_idx_map_from_region_name(
    std::vector<std::string> region_names) {
  std::map<std::string, anarci_idx_t> position_to_idx_map;
  bool                                valid_region_name = false;

  for (const std::string &region_name : region_names) {
    if (region_numbering_bounds.find(region_name) !=
        region_numbering_bounds.end()) {
      valid_region_name = true;
      for (std::string position_str : anarci_canonical_numbering) {
        int position_int =
            std::stoi(position_str.substr(0, position_str.length() - 1));
        if ((region_numbering_bounds[region_name].first <= position_int) &&
            (position_int <= region_numbering_bounds[region_name].second)) {
          position_to_idx_map[position_str] =
              full_position_to_idx_map[position_str];
        }
      }
    }
  }

  if (!valid_region_name) {
    position_to_idx_map = full_position_to_idx_map;
  }

  return position_to_idx_map;
}

/* Read the substitution matrix file */
// Returns the substitution matrix
// and a vector to convert amino acid characters to ints
std::pair<std::vector<std::vector<sim_t>>, std::vector<amino_acid_t>>
read_substitution_matrix(const std::string &substitution_matrix_path) {
  // converts amino acid from char to int
  std::vector<amino_acid_t> char_to_int_converter(128, MISSING_LETTER);

  amino_acid_char_t amino_acid_char;
  amino_acid_t      amino_acid_int = 0;
  std::string       line;

  std::ifstream ifs(substitution_matrix_path);
  // Read the alphabet and fill map of amino acid chars to ints
  if (std::getline(ifs, line)) {
    std::stringstream ss(line);
    while (ss >> amino_acid_char) {
      char_to_int_converter[amino_acid_char] = amino_acid_int;
      ++amino_acid_int;
    }
  } else {
    std::cerr << "The substitution matrix file is empty or an error occurred"
                 "while reading the first line."
              << std::endl;
  }

  amino_acid_t alphabet_size = amino_acid_int;

  // Initialize substitution matrix
  std::vector<sim_t>              zero_row(alphabet_size, 0);
  std::vector<std::vector<sim_t>> substitution_matrix(alphabet_size, zero_row);
  sim_t                           max_substitution_matrix_entry = 0;

  // Read the substitution matrix entries
  amino_acid_t row_idx = 0;
  sim_t        matrix_entry;
  while (std::getline(ifs, line)) {
    try {
      std::stringstream ss(line);

      // First entry in row is the amino acid character
      ss >> amino_acid_char;
      row_idx =
          char_to_int_converter[amino_acid_char];  // just in case amino
                                                   // acid order on rows
                                                   // and cols is different
      std::vector<sim_t> substitution_matrix_row(alphabet_size, 0);

      amino_acid_t col_idx = 0;
      while (ss >> matrix_entry) {
        substitution_matrix_row[col_idx] = matrix_entry;
        ++col_idx;
        if (matrix_entry > max_substitution_matrix_entry) {
          max_substitution_matrix_entry = matrix_entry;
        }
      }
      substitution_matrix[row_idx] = substitution_matrix_row;
    } catch (...) {
      std::cerr << "Error reading line in substitution matrix: " << line
                << std::endl;
    }
  }
  ifs.close();
  ifs.clear();

  return std::make_pair(substitution_matrix, char_to_int_converter);
}

/**
 * @brief Calculate the identity distance between two ANARCI vectors. The
 * distance is calculated as 1 - identity, and identity is calculated with
 * missing ends included in the length.
 */
template <typename anarci_vect_type>
cls::distance_t anarci_identity_distance_with_ends(
    const anarci_vect_type &anarci1, const anarci_vect_type &anarci2) {
  anarci_idx_t num_positions_agree = 0;
  anarci_idx_t length              = 0;
  for (int i = 0; i < anarci1.size(); ++i) {
    // If at least one of the sequences have a residue in this position, include
    // it in the length
    if ((static_cast<amino_acid_char_t>(anarci1[i]) != '|' &&
         static_cast<amino_acid_char_t>(anarci1[i]) != '\0') ||
        (static_cast<amino_acid_char_t>(anarci2[i]) != '|' &&
         static_cast<amino_acid_char_t>(anarci2[i]) != '\0')) {
      ++length;

      // If both sequences agree with a character that's not a missing residue,
      // add to number of positions that agree
      if (anarci1[i] == anarci2[i]) {
        ++num_positions_agree;
      }
    }
  };

  cls::distance_t identity_fraction = 0.0;
  if (length > 0) {
    identity_fraction = static_cast<cls::distance_t>(num_positions_agree) /
                        static_cast<cls::distance_t>(length);
  }

  return 1.0 - identity_fraction;
}

/**
 * @brief Calculate the distance between two ANARCI vectors. The distance is
 * calculated as 1 - identity, and identity is calculated with missing ends
 * NOT included in the length.
 */
template <typename anarci_vect_type>
cls::distance_t anarci_identity_distance_without_ends(
    const anarci_vect_type &anarci1, const anarci_vect_type &anarci2) {
  anarci_idx_t num_positions_agree = 0;
  anarci_idx_t length              = 0;
  for (int i = 0; i < anarci1.size(); ++i) {
    // If in this position, neither sequence has a missing end and both
    // sequences don't have a missing residue, add to the length
    if ((static_cast<amino_acid_char_t>(anarci1[i]) != '|') &&
        (static_cast<amino_acid_char_t>(anarci2[i]) != '|') &&
        (static_cast<amino_acid_char_t>(anarci1[i]) != '\0' &&
         static_cast<amino_acid_char_t>(anarci2[i]) != '\0')) {
      ++length;

      // If both characters agree (and are not a missing residue),
      // then add to the number of positions that agree
      if (anarci1[i] == anarci2[i]) {
        ++num_positions_agree;
      }
    }
  }

  cls::distance_t identity_fraction = 0.0;
  if (length > 0) {
    identity_fraction = static_cast<cls::distance_t>(num_positions_agree) /
                        static_cast<cls::distance_t>(length);
  }

  return 1.0 - identity_fraction;
}

/**
 * @brief Calculate the (unnormalized) Hamming distance between two ANARCI
 * vectors. Ends and missing characters are all treated the same in this case
 * and are included in the distance calculation.
 */
template <typename anarci_vect_type>
cls::distance_t anarci_hamming_distance_with_ends(
    const anarci_vect_type &anarci1, const anarci_vect_type &anarci2) {
  anarci_idx_t num_positions_agree = 0;
  for (int i = 0; i < anarci1.size(); ++i) {
    // If both sequences agree with the same character or both sequences have a
    // missing residue, increase the number of positions that agree
    if ((static_cast<amino_acid_char_t>(anarci1[i]) ==
         static_cast<amino_acid_char_t>(anarci2[i])) ||
        (static_cast<amino_acid_char_t>(anarci1[i]) == '\0' &&
         static_cast<amino_acid_char_t>(anarci2[i]) == '|') ||
        (static_cast<amino_acid_char_t>(anarci1[i]) == '|' &&
         static_cast<amino_acid_char_t>(anarci2[i]) == '\0')) {
      ++num_positions_agree;
    }
  };

  cls::distance_t hamming_dist =
      static_cast<cls::distance_t>(anarci1.size()) -
      static_cast<cls::distance_t>(num_positions_agree);
  return hamming_dist;
}

/**
 * @brief Parses an anarci string in json format into a sequence_t vector
 *
 * @param anarci_json anarci string in json format.
 *
 * @param idx_vector A sorted std::vector<anarci_idx_t> of the indicies (0 to
 * 199) of anarci numbering positions to conider. We only consider consider
 * positions corresponding to those in in idx_vector. The function returns
 * an sequence_t vector of the same size as idx_vector.
 *
 * @details Returns a vector with entries of anarci_amino_acid_t with the same
 * size as idx_vector. Each position in the vector corresponds to a position
 * in the anarci numbering scheme provided in idx_vector. If there is no amino
 * acid in the position, a null \0 character is used. Missing amino acids at
 * the ends of the full anarci vector (of size 200) are filled with '|' (that
 * is we have
 * '|' until the first amino acid, and after the last amino acid)
 */
sequence_t parse_anarci_json_to_saltatlas_sequence(
    const std::string &anarci_json, const std::vector<anarci_idx_t> &idx_vector,
    const cls::id_t sequence_id, const bool use_missing_end_char = false,
    const std::vector<amino_acid_t> &char_to_int_converter = {}) {
  // Parse the JSON string using boost and access the oject
  boost::json::object anarci_obj = boost::json::parse(anarci_json).as_object();

  // Initialize a sequence_t vector of length 200 initialized with
  // MISSING_LETTER. Read the anarci string into it using
  // the full_position_to_idx_map and char_to_int_converter
  sequence_t full_anarci_sequence(200, MISSING_LETTER);

  for (const auto &region : anarci_obj) {
    auto region_obj = region.value().as_object();
    for (const auto &position : region_obj) {
      std::string amino_acid_str = position.value().as_string().c_str();
      if (amino_acid_str.size() == 1) {
        amino_acid_char_t amino_acid_char = amino_acid_str[0];
        if (char_to_int_converter.size() == 0) {
          full_anarci_sequence[full_position_to_idx_map[position.key()]] =
              static_cast<amino_acid_t>(amino_acid_char);
        } else {
          full_anarci_sequence[full_position_to_idx_map[position.key()]] =
              char_to_int_converter[amino_acid_char];
        }
      } else {
        std::cout << "In sequence id " << sequence_id
                  << ", read value that's not a single amino acid char: "
                  << amino_acid_str << std::endl;
      }
    }
  }

  // Set the missing residues at the ends of the sequence to '|'
  if (use_missing_end_char) {
    for (int i = 0; i < 200; ++i) {
      if (static_cast<amino_acid_char_t>(full_anarci_sequence[i]) == '\0') {
        full_anarci_sequence[i] = static_cast<amino_acid_t>('|');
      } else {
        break;
      }
    }
    for (int i = 199; i >= 0; --i) {
      if (static_cast<amino_acid_char_t>(full_anarci_sequence[i]) == '\0') {
        full_anarci_sequence[i] = static_cast<amino_acid_t>('|');
      } else {
        break;
      }
    }
  }

  // Initialize anarci sequence and only use the positions in idx_vector
  sequence_t anarci_sequence;
  for (anarci_idx_t i : idx_vector) {
    anarci_sequence.push_back(full_anarci_sequence[i]);
  }

  return anarci_sequence;
}

/**
 * @brief Parses an anarci vector of chars into a string for easy
 * printing/reading
 *
 * @param sequence_t Anarci vector of characters, e.g., the output of
 * parse_anarci_json_to_anarci_vector.
 */
template <typename anarci_vect_type>
std::string anarci_to_string(const anarci_vect_type &anarci_vector) {
  std::string formatted_anarci_str = "";
  for (const amino_acid_char_t &amino_acid : anarci_vector) {
    if (amino_acid == '\0' || amino_acid == '|') {
      formatted_anarci_str += "-";
    } else {
      formatted_anarci_str += amino_acid;
    }
  }
  return formatted_anarci_str;
}

const auto get_anarci_substitution_distance_lambda(
    const std::vector<std::vector<sim_t>> &substitution_matrix,
    const cls::distance_t                 &inverse_power) {
  auto anarci_substitution_distance_lambda =
      [&substitution_matrix, &inverse_power](const auto &seq1,
                                             const auto &seq2) {
        // Get sequence lengths
        const uint32_t m =
            seq1.size();  // m+1 rows in the alignment score matrix
        const uint32_t n =
            seq2.size();  // n+1 columns in the alignment score matrix

        // If either sequence is empty, their distance is the max value
        if (m == 0 || n == 0) {
          return std::numeric_limits<cls::distance_t>::max();
        }

        sim_t sim_score = 0;  // Similarity score

        for (int i = 0; i < seq1.size(); ++i) {
          // If both sequences have no residue, don't change score
          if (seq1[i] == MISSING_LETTER && seq2[i] == MISSING_LETTER) {
            // Do nothing
          }
          // If one sequence has no residue, but the other does, add the gap
          // extension penalty
          else if (seq1[i] == MISSING_LETTER || seq2[i] == MISSING_LETTER) {
            sim_score -= GAP_EXTEND_COST;
          }
          // If both sequences have residues, add the substitution matrix
          // score if possible. If for some reason this pair is not in the
          // substitution matrix, again, add the gap extension cost
          else {
            try {
              sim_score += substitution_matrix[seq1[i]][seq2[i]];
            } catch (...) {
              sim_score -= GAP_EXTEND_COST;
            }
          }
        }

        if (sim_score <= 0) {
          return std::numeric_limits<cls::distance_t>::max();
        } else {
          cls::distance_t distance =
              std::pow(static_cast<cls::distance_t>(sim_score), -inverse_power);
          return distance;
        }
      };

  return anarci_substitution_distance_lambda;
}

const auto get_anarci_identity_distance_lambda(
    const bool dont_include_ends, const bool use_hamming_distance = false) {
  auto anarci_identity_distance_lambda =
      [&dont_include_ends, &use_hamming_distance](const auto &anarci1,
                                                  const auto &anarci2) {
        anarci_idx_t num_positions_agree = 0;
        anarci_idx_t length              = 0;

        if (use_hamming_distance) {
          for (int i = 0; i < anarci1.size(); ++i) {
            // If both sequences agree with the same character or both sequences
            // have a missing residue, increase the number of positions that
            // agree
            if (anarci1[i] == anarci2[i] ||
                (static_cast<amino_acid_char_t>(anarci1[i]) == '\0' &&
                 static_cast<amino_acid_char_t>(anarci2[i]) == '|') ||
                (static_cast<amino_acid_char_t>(anarci1[i]) == '|' &&
                 static_cast<amino_acid_char_t>(anarci2[i]) == '\0')) {
              ++num_positions_agree;
            }
          };

          cls::distance_t hamming_dist =
              static_cast<cls::distance_t>(anarci1.size()) -
              static_cast<cls::distance_t>(num_positions_agree);
          return hamming_dist;
        }

        if (dont_include_ends) {
          for (int i = 0; i < anarci1.size(); ++i) {
            // If in this position, neither sequence has a missing end and
            // both sequences don't have a missing residue, add to the length
            if ((static_cast<amino_acid_char_t>(anarci1[i]) != '|') &&
                (static_cast<amino_acid_char_t>(anarci2[i]) != '|') &&
                (static_cast<amino_acid_char_t>(anarci1[i]) != '\0' &&
                 static_cast<amino_acid_char_t>(anarci2[i]) != '\0')) {
              ++length;

              // If both characters agree (and are not a missing residue),
              // then add to the number of positions that agree
              if (anarci1[i] == anarci2[i]) {
                ++num_positions_agree;
              }
            }
          }
        } else {
          for (int i = 0; i < anarci1.size(); ++i) {
            // If at least one of the sequences have a residue in this
            // position, include it in the length
            if ((static_cast<amino_acid_char_t>(anarci1[i]) != '|' &&
                 static_cast<amino_acid_char_t>(anarci1[i]) != '\0') ||
                (static_cast<amino_acid_char_t>(anarci2[i]) != '|' &&
                 static_cast<amino_acid_char_t>(anarci2[i]) != '\0')) {
              ++length;

              // If both sequences agree with a character that's not a missing
              // residue, add to number of positions that agree
              if (anarci1[i] == anarci2[i]) {
                ++num_positions_agree;
              }
            }
          }
        }

        cls::distance_t identity_fraction = 0.0;
        if (length > 0) {
          identity_fraction =
              static_cast<cls::distance_t>(num_positions_agree) /
              static_cast<cls::distance_t>(length);
        }

        return static_cast<cls::distance_t>(1.0) - identity_fraction;
      };

  return anarci_identity_distance_lambda;
}

// * Hard code BLOSUM62 substitution distance * //

// Hard code in the BLOSUM62 substitution matrix instead of reading at run time
std::vector<amino_acid_t> blosum62_char_to_int_converter{
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 23,  255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 0,   20,  4,   3,   6,   13,  7,   8,   9,   255,
    11,  10,  12,  2,   255, 14,  5,   1,   15,  16,  255, 19,  17,  22,  18,
    21,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255};

std::vector<std::vector<sim_t>> blosum62_substitution_matrix{
    {4,  -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1,
     -1, -2, -1, 1,  0, -3, -2, 0, -2, -1, 0,  -4},
    {-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,
     -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,  -1, -4},
    {-2, 0,  6,  1, -3, 0,  0,  0,  1, -3, -3, 0,
     -2, -3, -2, 1, 0,  -4, -2, -3, 3, 0,  -1, -4},
    {-2, -2, 1,  6, -3, 0,  2,  -1, -1, -3, -4, -1,
     -3, -3, -1, 0, -1, -4, -3, -3, 4,  1,  -1, -4},
    {0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
     -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1, 1,  0,  0, -3, 5,  2,  -2, 0, -3, -2, 1,
     0,  -3, -1, 0, -1, -2, -1, -2, 0, 3,  -1, -4},
    {-1, 0,  0,  2, -4, 2,  5,  -2, 0, -3, -3, 1,
     -2, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
    {0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
     -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4},
    {-2, 0,  1,  -1, -3, 0,  0, -2, 8, -3, -3, -1,
     -2, -1, -2, -1, -2, -2, 2, -3, 0, 0,  -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
     1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
     2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4},
    {-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,
     -1, -3, -1, 0,  -1, -3, -2, -2, 0,  1,  -1, -4},
    {-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1,
     5,  0,  -2, -1, -1, -1, -1, 1,  -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3,
     0,  6,  -4, -2, -2, 1,  3,  -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
     -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4},
    {1,  -1, 1,  0, -1, 0,  0,  0,  -1, -2, -2, 0,
     -1, -2, -1, 4, 1,  -3, -2, -2, 0,  0,  0,  -4},
    {0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1,
     -1, -2, -1, 1,  5,  -2, -2, 0,  -1, -1, 0,  -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
     -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
     -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4},
    {0, -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2,
     1, -1, -2, -2, 0,  -3, -1, 4,  -3, -2, -1, -4},
    {-2, -1, 3,  4, -3, 0,  1,  -1, 0, -3, -4, 0,
     -3, -3, -2, 0, -1, -4, -3, -3, 4, 1,  -1, -4},
    {-1, 0,  0,  1, -3, 3,  4,  -2, 0, -3, -3, 1,
     -1, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
    {0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
     -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

template <typename sequence_type>
cls::distance_t anarci_substitution_distance(const sequence_type &seq1,
                                             const sequence_type &seq2) {
  // Get sequence lengths
  const uint32_t m = seq1.size();  // m+1 rows in the alignment score matrix
  const uint32_t n = seq2.size();  // n+1 columns in the alignment score matrix

  // If either sequence is empty, their distance is the max value
  if (m == 0 || n == 0) {
    return std::numeric_limits<cls::distance_t>::max();
  }

  sim_t sim_score = 0;  // Similarity score

  for (int i = 0; i < seq1.size(); ++i) {
    // If both sequences have no residue, don't change score
    if (seq1[i] == MISSING_LETTER && seq2[i] == MISSING_LETTER) {
      // Do nothing
    }
    // If one sequence has no residue, but the other does, add the gap
    // extension penalty
    else if (seq1[i] == MISSING_LETTER || seq2[i] == MISSING_LETTER) {
      sim_score -= GAP_EXTEND_COST;
    }
    // If both sequences have residues, add the substitution matrix
    // score if possible. If for some reason this pair is not in the
    // substitution matrix, again, add the gap extension cost
    else {
      try {
        sim_score += blosum62_substitution_matrix[seq1[i]][seq2[i]];
      } catch (...) {
        sim_score -= GAP_EXTEND_COST;
      }
    }
  }

  if (sim_score <= 0) {
    return std::numeric_limits<cls::distance_t>::max();
  } else {
    // Distance = similarity ^ (-1)
    cls::distance_t distance =
        std::pow(static_cast<cls::distance_t>(sim_score), -1.0);
    return distance;
  }
}