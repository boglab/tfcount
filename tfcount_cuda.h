#ifndef BTFCOUNT_TFCOUNT_CUDA
#define BTFCOUNT_TFCOUNT_CUDA

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

void RunPairedCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *spacer_sizes, unsigned int *rvd_pairs, unsigned int *rvd_lengths, double *cutoffs, unsigned int num_rvd_pairs, int c_upstream, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results);


#ifdef __cplusplus
}
#endif

#endif
