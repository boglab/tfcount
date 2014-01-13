#ifndef BTFCOUNT_TFCOUNT_CUDA
#define BTFCOUNT_TFCOUNT_CUDA

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

void RunCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *rvd_seqs, unsigned int *rvd_lengths, double *cutoffs, unsigned int num_rvd_seqs, int c_upstream, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results);
void RunPairedCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *spacer_sizes, unsigned int *rvd_pairs, unsigned int *rvd_lengths, double *cutoffs, unsigned int num_rvd_pairs, int c_upstream, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results);
void RunPairedFindBindingSitesKeepScores_init(unsigned int **d_rvd_pair_p, double **d_scoring_matrix_p, size_t *sm_pitch_p, unsigned char **d_prelim_results_p, int **d_prelim_results_indexes_p, char **d_reference_sequence_p, unsigned char **prelim_results_p, int **prelim_results_indexes_p, unsigned long *reference_window_size_p, int *score_block_x_p, int *score_block_y_p, unsigned int **rvd_pair, double **scoring_matrix, unsigned int *rvd_lengths, unsigned int scoring_matrix_length);
int RunPairedFindBindingSitesKeepScores(char *d_reference_sequence, unsigned int *d_rvd_pairs, double *d_scoring_matrix, size_t sm_pitch, unsigned char *d_prelim_results, int *d_prelim_results_indexes, unsigned char *prelim_results, int *prelim_results_indexes, unsigned long reference_window_size, int score_block_x, int score_block_y, unsigned int *rvd_lengths, char *ref_seq, unsigned long ref_seq_len, double *cutoffs, int c_upstream);
void RunPairedFindBindingSitesKeepScores_cleanup(char *d_reference_sequence, unsigned int *d_rvd_pairs, double *d_scoring_matrix, unsigned char *d_prelim_results, int *d_prelim_results_indexes, unsigned char *prelim_results, int *prelim_results_indexes);


#ifdef __cplusplus
}
#endif

#endif
