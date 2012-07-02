#ifndef BTFCOUNT_TFCOUNT_CUDA
#define BTFCOUNT_TFCOUNT_CUDA

#ifdef __cplusplus
extern "C" {
#endif

void RunCountBindingSites(char *seq_filename, unsigned int *spacer_sizes, unsigned int *rvd_pairs, unsigned int *rvd_lengths, double *cutoffs, unsigned int num_rvd_pairs, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results);


#ifdef __cplusplus
}
#endif

#endif
