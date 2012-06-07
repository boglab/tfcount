#ifndef OTFS_PAIRED_CUDA
#define OTFS_PAIRED_CUDA

#ifdef __cplusplus
extern "C" {
#endif

void CountBindingSites(char *reference_sequence, unsigned long reference_sequence_length, int *spacer_sizes, int **rvd_sequences, int *rvd_sequence_lengths, double *cutoffs, double **scoring_matrix, int scoring_matrix_length, int **results);

#ifdef __cplusplus
}
#endif

#endif
