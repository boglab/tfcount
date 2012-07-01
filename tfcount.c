#include <bcutils/Hashmap.h>
#include <bcutils/Array.h>

#include "tfcount_cuda.h"

int run_counting_task(Hashmap *kwargs) {

  // Options
  char *seq_filename = hashmap_get(kwargs, "seq_filename");
  char *rvd_string = hashmap_get(kwargs, "rvd_string");
  char *rvd_string2 = hashmap_get(kwargs, "rvd_string2");

  double weight = *((double *) hashmap_get(kwargs, "weight"));
  double cutoff = *((double *) hashmap_get(kwargs, "cutoff"));

  unsigned int **count_results_array = hashmap_get(kwargs, "count_results_array");

  unsigned int spacer_min = *((unsigned int *) hashmap_get(kwargs, "spacer_min"));
  unsigned int spacer_max = *((unsigned int *) hashmap_get(kwargs, "spacer_max"));

  // Process RVD sequences

  Array *rvd_seq = rvd_string_to_array(rvd_string);
  Array *rvd_seq2 = rvd_string_to_array(rvd_string2);

  unsigned int *rvd_seqs[2];

  rvd_seqs[0] = calloc(32, sizeof(unsigned int));
  rvd_seqs[1] = calloc(32, sizeof(unsigned int));

  Array *joined_rvd_seq = array_concat(rvd_seq, rvd_seq2);

  // Get RVD/bp matching scores

  Hashmap *diresidue_probabilities = get_diresidue_probabilities(joined_rvd_seq, weight);
  Hashmap *diresidue_scores = convert_probabilities_to_scores(diresidue_probabilities);
  hashmap_delete(diresidue_probabilities, NULL);

  // RVD Sequence lengths

  unsigned int rvd_seqs_lens[2];
  rvd_seqs_lens[0] = array_size(rvd_seq);
  rvd_seqs_lens[1] = array_size(rvd_seq2);

  // Compute optimal score for the RVD sequences

  double best_score = get_best_score(rvd_seq, diresidue_scores);
  double best_score2 = get_best_score(rvd_seq2, diresidue_scores);

  // Create cuttoffs

  double cutoffs[2];

  cutoffs[0] = cutoff * best_score;
  cutoffs[1] = cutoff * best_score2;

  // Spacer sizes

  unsigned int spacer_sizes[2];

  spacer_sizes[0] = spacer_min;
  spacer_sizes[1] = spacer_max;

  // Convert hashmap to int map
  
  hashmap_add(diresidue_scores, "XX", double_array(0, 0, 0, 0, 0));

  double **scoring_matrix = calloc(64, sizeof(double*));

  Hashmap *rvd_to_int = hashmap_new(64);
  unsigned int *rvd_ints = calloc(64, sizeof(unsigned int));

  char **diresidues = hashmap_keys(diresidue_scores);

  for (unsigned int i = 0; i < hashmap_size(diresidue_scores); i++) {

    rvd_ints[i] = i;
    hashmap_add(rvd_to_int, diresidues[i], rvd_ints + i);

    scoring_matrix[i] = hashmap_get(diresidue_scores, diresidues[i]);
    scoring_matrix[i][4] = (cutoffs[0] > cutoffs[1] ? cutoffs[0] : cutoffs[1]);

  }
  
  // Transform RVD sequences to int sequences

  for (int i = 0; i < 32; i++) {
    if (i < array_size(rvd_seq)) {
      rvd_seqs[0][i] = *(unsigned int *)(hashmap_get(rvd_to_int, array_get(rvd_seq, i)));
    } else {
      rvd_seqs[0][i] = *(unsigned int *)(hashmap_get(rvd_to_int, "XX"));
    }
  }

  for (int i = 0; i < 32; i++) {
    if (i < array_size(rvd_seq2)) {
      rvd_seqs[1][i] = *(unsigned int *)(hashmap_get(rvd_to_int, array_get(rvd_seq2, i)));
    } else {
      rvd_seqs[1][i] = *(unsigned int *)(hashmap_get(rvd_to_int, "XX"));
    }
  }

  RunCountBindingSites(seq_filename, spacer_sizes, rvd_seqs, rvd_seqs_lens, cutoffs, scoring_matrix, hashmap_size(diresidue_scores), count_results_array);

  free(rvd_seqs[0]);
  free(rvd_seqs[1]);
  free(scoring_matrix);
  hashmap_delete(rvd_to_int, NULL);
  free(rvd_ints);
  free(diresidues);
  
  if (rvd_seq) {
    array_delete(rvd_seq, free);
  }

  if (rvd_seq2) {
    array_delete(rvd_seq2, free);
  }

  if (joined_rvd_seq) {
    array_delete(joined_rvd_seq, NULL);
  }

  if (diresidue_scores) {
    hashmap_delete(diresidue_scores, free);
  }

}
