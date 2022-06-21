/*  Copyright (C) 2018-2019 Andrew D. Smith
 *
 *  Authors: Andrew D. Smith
 *
 *  This file is part of ABISMAL.
 *
 *  ABISMAL is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  ABISMAL is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 */

#ifndef ABISMAL_INDEX_HPP
#define ABISMAL_INDEX_HPP

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <deque>
#include <bitset>
#include <cassert>
#include <climits>
#include <cmath>

#include "smithlab_utils.hpp"
#include "dna_four_bit.hpp"

typedef size_t element_t;
typedef std::vector<element_t> Genome;
typedef bool two_letter_t;
typedef uint8_t three_letter_t;

static inline char random_base() {return "ACGT"[rand() & 3];}

namespace seed {
  // number of positions in the hashed portion of the seed
  static const uint32_t key_weight = 25u;
  static const uint32_t key_weight_three = 16u;

  // window in which we select the best k-mer. The longer it is,
  // the longer the minimum read length that guarantees an exact
  // match will be mapped
  static const uint32_t window_size = 20u;

  // number of positions to sort within buckets
  static const uint32_t n_sorting_positions = 256u;

  static const size_t hash_mask = (1ull << seed::key_weight) - 1;
  static const size_t hash_mask_three = pow(3, key_weight_three);

  // the purpose of padding the left and right ends of the
  // concatenated genome is so that later we can avoid having to check
  // the (unlikely) case that a read maps partly off either end of the
  // genome.
  static const size_t padding_size = std::numeric_limits<int16_t>::max();

  void read(FILE* in);
  void write(FILE* out);
};

struct ChromLookup {
  std::vector<std::string> names;
  std::vector<uint32_t> starts;

  void
  get_chrom_idx_and_offset(const uint32_t pos,
                           uint32_t &chrom_idx,
                           uint32_t &offset) const;
  bool
  get_chrom_idx_and_offset(const uint32_t pos,
                           const uint32_t readlen,
                           uint32_t &chrom_idx,
                           uint32_t &offset) const;

  uint32_t
  get_pos(const std::string &chrom, const uint32_t offset) const;
  uint32_t
  get_genome_size() const {return starts.back();}

  FILE * read(FILE *in);
  std::istream & read(std::istream &in);
  void read(const std::string &infile);

  FILE * write(FILE *out) const;
  std::ostream & write(std::ostream &out) const;
  void write(const std::string &outfile) const;

  std::string tostring() const;
};

template <class G>
void
load_genome(const std::string &genome_file, G &genome, ChromLookup &cl) {
  size_t num_ns = 0;
  std::ifstream in(genome_file);
  if (!in)
    throw std::runtime_error("bad genome file: " + genome_file);

  const auto begin_pos = in.tellg();
  in.seekg(0, std::ios_base::end);
  const size_t file_size = in.tellg() - begin_pos;
  in.seekg(0, std::ios_base::beg);

  genome.clear();
  // pad on at start; the space for padding at the end will be
  // available because of the newlines and chromosome names
  genome.reserve(file_size + seed::padding_size);

  // pad the start of the concatenated sequence
  cl.names.push_back("pad_start");
  for (size_t i = 0; i < seed::padding_size; ++i)
    genome.push_back('Z');
  cl.starts.push_back(genome.size());

  std::string line;
  while (getline(in, line))
    if (line[0] != '>') {
      for (auto it(begin(line)); it != end(line); ++it) {
        if (base2int(*it) == 4) { // non-acgts become random bases 
          ++num_ns;
          *it = random_base();
        }
      }
      copy(std::begin(line), std::end(line), std::back_inserter(genome));
    }
    else {
      cl.names.push_back(line.substr(1, line.find_first_of(" \t") - 1));
      cl.starts.push_back(genome.size());
    }

  // now pad the end of the concatenated sequence
  cl.names.push_back("pad_end");
  cl.starts.push_back(genome.size());
  for (size_t i = 0; i < seed::padding_size; ++i)
    genome.push_back('Z');
  cl.starts.push_back(genome.size());

}

std::ostream &
operator<<(std::ostream &out, const ChromLookup &cl);

enum three_conv_type { c_to_t, g_to_a};
struct AbismalIndex {

  static bool VERBOSE;

  uint32_t max_candidates;

  size_t counter_size; // number of kmers indexed
  size_t counter_size_three; // number of kmers indexed

  size_t index_size; // number of genome positions indexed
  size_t index_size_three; // number of genome positions indexed

  std::vector<uint32_t> index; // genome positions for each k-mer
  std::vector<uint32_t> index_t; // genome positions for each k-mer
  std::vector<uint32_t> index_a; // genome positions for each k-mer

  std::vector<uint32_t> counter; // offset of each k-mer in "index"
  std::vector<uint32_t> counter_t; // offset of each k-mer in "index"
  std::vector<uint32_t> counter_a; // offset of each k-mer in "index"

  // a vector indicating whether each position goes into two-
  // or three-letter encoding
  std::vector<bool> is_two_letter;
  std::vector<bool> keep;

  Genome genome; // the genome
  ChromLookup cl; // the starting position of each chromosome

  void create_index(const std::string &genome_file);

  // count how many positions must be stored for each hash value
  template<const bool use_mask>
  void get_bucket_sizes();

  template<const three_conv_type the_conv, const bool use_mask>
  void get_bucket_sizes_three();

  // selects which positions go into two-letter
  void select_two_letter_positions();

  // selects which positions to keep based on k-mer frequencies
  void compress_dp();

  // put genome positions in the appropriate buckets
  void hash_genome();

  // Sort each bucket, if the seed length is more than
  // seed::key_weight, then use binary search for the rest part of the seed
  void sort_buckets();

  // convert the genome to 4-bit encoding
  void encode_genome(const std::vector<uint8_t> &input_genome);

  // write index to disk
  void write(const std::string &index_file) const;

  // read index from disk
  void read(const std::string &index_file);

  static std::string internal_identifier;
  AbismalIndex() {}
};

// A/T nucleotide to 1-bit value (0100 | 0001 = 5) is for A or G.
inline two_letter_t
get_bit(const uint8_t nt) {return (nt & 5) == 0;}

template<const three_conv_type the_conv>
inline three_letter_t
get_three_letter_num(const uint8_t &nt) {
  return ((the_conv == c_to_t) ?
      ((((nt & 4) != 0)<<1)  | ((nt & 1) != 0)) : // C=T=0, A=1, G=2
      ((((nt & 8) != 0)<<1)  | ((nt & 2) != 0))); // A=G=0,C=1,T=2
}

inline void
shift_hash_key(const uint8_t c, uint32_t &hash_key) {
  hash_key = (((hash_key << 1) | get_bit(c)) & seed::hash_mask);
}

template<const three_conv_type the_conv>
inline void
shift_three_key(const uint8_t c, uint32_t &hash_key) {
  hash_key = (
    hash_key*3 + get_three_letter_num<the_conv>(c))
  %seed::hash_mask_three;
}

// get the hash value for a k-mer (specified as some iterator/pointer)
// and the encoding for the function above
template <class T>
void
get_1bit_hash(T r, uint32_t &k) {
  const auto lim = r + seed::key_weight;
  k = 0;
  while (r != lim) {
    k = ((k << 1) | get_bit(*r));
    ++r;
  }
}

template <const three_conv_type the_conv, class T>
void
get_base_3_hash(T r, uint32_t &k) {
  const auto lim = r + seed::key_weight_three;
  k = 0;
  while (r != lim) {
    shift_three_key<the_conv>(*r, k);
    ++r;
  }
}

#endif
