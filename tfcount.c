#include <bcutils/Hashmap.h>
#include <bcutils/Array.h>
#include <bcutils/bcutils.h>

#include "tfcount_cuda.h"

#define BIGGEST_RVD_SCORE_EVER 100
#define PADDED_RVD_WIDTH 32

int run_counting_task(Hashmap *kwargs) {

  // Options
  char *seq_filename = hashmap_get(kwargs, "seq_filename");
  char **rvd_strings = hashmap_get(kwargs, "rvd_strings");
  int num_rvd_pairs = *((int *) hashmap_get(kwargs, "num_rvd_pairs"));

  double weight = *((double *) hashmap_get(kwargs, "weight"));
  double cutoff = *((double *) hashmap_get(kwargs, "cutoff"));

  unsigned int *count_results_array = hashmap_get(kwargs, "count_results_array");

  unsigned int spacer_min = *((unsigned int *) hashmap_get(kwargs, "spacer_min"));
  unsigned int spacer_max = *((unsigned int *) hashmap_get(kwargs, "spacer_max"));
  
  Array *empty_array = array_new();

  unsigned int *rvd_pairs = calloc(2 * PADDED_RVD_WIDTH * num_rvd_pairs, sizeof(unsigned int));
  unsigned int *rvd_lens = calloc(2 * num_rvd_pairs, sizeof(unsigned int));
  double *rvd_cutoffs = calloc(2 * num_rvd_pairs, sizeof(double));

  // Get RVD/bp matching scores

  Hashmap *diresidue_probabilities = get_diresidue_probabilities(empty_array, weight);
  Hashmap *diresidue_scores = convert_probabilities_to_scores(diresidue_probabilities);
  hashmap_delete(diresidue_probabilities, NULL);

  // Spacer sizes

  unsigned int spacer_sizes[2];

  spacer_sizes[0] = spacer_min;
  spacer_sizes[1] = spacer_max;

  // Convert hashmap to int map
  
  hashmap_add(diresidue_scores, "XX", double_array(0, 0, 0, 0, BIGGEST_RVD_SCORE_EVER));

  double **scoring_matrix = calloc(64, sizeof(double*));

  Hashmap *rvd_to_int = hashmap_new(64);
  unsigned int *rvd_ints = calloc(64, sizeof(unsigned int));

  char **diresidues = hashmap_keys(diresidue_scores);

  for (unsigned int i = 0; i < hashmap_size(diresidue_scores); i++) {

    rvd_ints[i] = i;
    hashmap_add(rvd_to_int, diresidues[i], rvd_ints + i);

    scoring_matrix[i] = hashmap_get(diresidue_scores, diresidues[i]);
    scoring_matrix[i][4] = BIGGEST_RVD_SCORE_EVER;

  }
  
  unsigned int blank_rvd = *(unsigned int *)(hashmap_get(rvd_to_int, "XX"));
  
  // Process RVD sequences
  
  for (int i = 0; i < num_rvd_pairs; i++) {
    
    int first_pos = 2 * i;
    int second_pos = first_pos + 1;
    
    Array *rvd_seq = rvd_string_to_array(rvd_strings[first_pos]);
    Array *rvd_seq2 = rvd_string_to_array(rvd_strings[second_pos]);
    
    rvd_lens[first_pos] = array_size(rvd_seq);
    rvd_lens[second_pos] = array_size(rvd_seq2);
    
    rvd_cutoffs[first_pos] = cutoff * get_best_score(rvd_seq, diresidue_scores);
    rvd_cutoffs[second_pos] = cutoff * get_best_score(rvd_seq2, diresidue_scores);
    
    for (int i = 0; i < PADDED_RVD_WIDTH; i++) {
      if (i < array_size(rvd_seq)) {
        rvd_pairs[first_pos * PADDED_RVD_WIDTH + i] = *(unsigned int *)(hashmap_get(rvd_to_int, array_get(rvd_seq, i)));
      } else {
        rvd_pairs[first_pos * PADDED_RVD_WIDTH + i] = blank_rvd;
      }
    }

    for (int i = 0; i < PADDED_RVD_WIDTH; i++) {
      if (i < array_size(rvd_seq2)) {
        rvd_pairs[second_pos * PADDED_RVD_WIDTH + i] = *(unsigned int *)(hashmap_get(rvd_to_int, array_get(rvd_seq2, i)));
      } else {
        rvd_pairs[second_pos * PADDED_RVD_WIDTH + i] = blank_rvd;
      }
    }
    
  }
  
  // Transform RVD sequences to int sequences

  RunCountBindingSites(seq_filename, spacer_sizes, rvd_pairs, rvd_lens, rvd_cutoffs, num_rvd_pairs, scoring_matrix, hashmap_size(diresidue_scores), count_results_array);

  free(scoring_matrix);
  hashmap_delete(rvd_to_int, NULL);
  free(rvd_ints);
  free(diresidues);
  
  free(rvd_pairs);
  free(rvd_lens);
  free(rvd_cutoffs);
  
  array_delete(empty_array, NULL);

  if (diresidue_scores) {
    hashmap_delete(diresidue_scores, free);
  }
  
  return 0;

}
