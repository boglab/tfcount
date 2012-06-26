#ifndef OTFS_PAIRED_CUDA
#define OTFS_PAIRED_CUDA

#ifdef __cplusplus
extern "C" {
#endif

void RunCountBindingSites(char *seq_filename, unsigned int *spacer_sizes, unsigned int **rvd_sequences, unsigned int *rvd_sequence_lengths, double *cutoffs, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int **results);


#ifdef __cplusplus
}
#endif

#endif
