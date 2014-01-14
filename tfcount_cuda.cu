#undef _GLIBCXX_USE_INT128
#undef _GLIBCXX_ATOMIC_BUILTINS

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/partition.h>

extern "C" {
#include <bcutils/bcutils.h>
}

#include "tfcount_cuda.h"


// Sequence handling
#include <zlib.h>
#include <bcutils/kseq.h>
KSEQ_INIT(gzFile, gzread)

#define MAX_THREADS_PER_BLOCK 1024
#define SCORE_THREADS_PER_BLOCK 448
#define TALLY_THREADS_PER_BLOCK 768
#define MAX_BLOCKS_PER_GRID 65535

#define PADDED_RVD_WIDTH 32

#define cudaSafeCall(call){   \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "%s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

__device__ double ScoringMatrixVal(double *scoring_matrix, size_t pitch, unsigned int row, unsigned int column);
double *ScoringMatrixRow(double *scoring_matrix, size_t pitch, unsigned int row);

__global__ void ScoreBindingSites(char *input_sequence, unsigned long is_length, unsigned int *rvd_sequence, unsigned int rs_len, double cutoff, int c_upstream, unsigned int rvd_num, double *scoring_matrix, size_t sm_pitch, unsigned char *results) {

  int block_seq_index = SCORE_THREADS_PER_BLOCK * (blockIdx.y * gridDim.x + blockIdx.x);
  int thread_id = (blockDim.x * threadIdx.y) + threadIdx.x;
  int seq_index = block_seq_index + thread_id + 1;

  if (seq_index < 1 || seq_index >= is_length || seq_index + rs_len >= is_length - 1) return;

  char first = input_sequence[seq_index - 1];
  char last  = input_sequence[seq_index + rs_len];

  int first_t = c_upstream != 1 && (first == 'T' || first == 't');
  int first_c = c_upstream != 0 && (first == 'C' || first == 'c');
  int last_a  = c_upstream != 1 && (last == 'A' || last == 'a');
  int last_g  = c_upstream != 0 && (last == 'G' || last == 'g');


  if (first_c || first_t || last_g || last_a) {

    double thread_result_t = 0;
    double thread_result_a = 0;

    for (int i = 0; i < rs_len; i++) {

      int sm_col_t = 4;

      char base = input_sequence[seq_index + i];

      if (base == 'A' || base == 'a')
        sm_col_t = 0;
      if (base == 'C' || base == 'c')
        sm_col_t = 1;
      if (base == 'G' || base == 'g')
        sm_col_t = 2;
      if (base == 'T' || base == 't')
        sm_col_t = 3;

      int rvd_index_t = i;
      int rvd_index_a = rs_len - i - 1;

      thread_result_t += ScoringMatrixVal(scoring_matrix, sm_pitch, rvd_sequence[rvd_index_t], sm_col_t);

      int sm_col_a = (sm_col_t == 4 ? 4 : 3 - sm_col_t);

      thread_result_a += ScoringMatrixVal(scoring_matrix, sm_pitch, rvd_sequence[rvd_index_a], sm_col_a);

    }

    if (first_c || first_t)
      results[seq_index] |= (thread_result_t < cutoff ? 1UL : 0UL) << ((2 * rvd_num) + (first_c * 4));

    if (last_g || last_a)
      results[seq_index] |= (thread_result_a < cutoff ? 1UL : 0UL) << ((2 * rvd_num + 1) + (last_g * 4));

  }

}

__global__ void TallyResults(unsigned char *prelim_results, unsigned int pr_length, unsigned int rs_len, int c_upstream, unsigned int u_shift, unsigned int d_shift, unsigned int spacer_range_start, unsigned int spacer_range_end, unsigned int *second_results) {
    
  short thread_result = 0;
  
  int block_seq_index = TALLY_THREADS_PER_BLOCK * (blockIdx.y * gridDim.x + blockIdx.x);
  int seq_index = block_seq_index + (blockDim.x * threadIdx.y) + threadIdx.x;
  
  if (seq_index < 0 || seq_index >= pr_length) return;

  int first_t = (prelim_results[seq_index] & (1UL << u_shift)) > 0;
  int first_c = (prelim_results[seq_index] & (1UL << (u_shift + 4))) > 0;

  if (!((c_upstream != 0 && first_c) || (c_upstream != 1 && first_t))) return;
  
  for (int i = spacer_range_start; i <= spacer_range_end; i++) {
    
    if (seq_index + rs_len + i >= pr_length) continue;
    
    thread_result += ((prelim_results[seq_index + rs_len + i] & (1UL << (d_shift + (first_c * 4))) ) > 0);

  }
  
  second_results[seq_index] = thread_result;
  
}

__device__ double ScoringMatrixVal(double *scoring_matrix, size_t pitch, unsigned int row, unsigned int column) {
  
  return *((double*)((char*) scoring_matrix + row * pitch) + column);
  
}

double *ScoringMatrixRow(double *scoring_matrix, size_t pitch, unsigned int row) {
  return (double*)((char*) scoring_matrix + row * pitch);
}

struct first_t_or_c
{
  __host__ __device__ bool operator()(const unsigned char &x) {
      return (x & 1) || (x & (1 << 4));
  }
};



struct last_a_or_g
{
  __host__ __device__ bool operator()(const unsigned char &x) {
      return (x & (1 << 1)) || (x & (1 << 5));
  }
};

struct index_to_keep
{
  const int offset;

  index_to_keep(int _offset) : offset(_offset) {}

  __host__ __device__ int operator()(const unsigned char &x, const int &y) {
      if ((x & 1) || (x & (1 << 4)) || (x & (1 << 1)) || (x & (1 << 5))) {
        return y + offset;
      } else {
        return -1;
      }
  }
};

struct index_to_keep_paired
{
  const int offset;

  index_to_keep_paired(int _offset) : offset(_offset) {}

  __host__ __device__ int operator()(const unsigned char &x, const int &y) {
      if ((x & 1) || (x & (1 << 4)) || (x & (1 << 2)) || (x & (1 << 6))) {
        return y + offset;
      } else {
        return -1;
      }
  }
};

struct valid_index
{
  __host__ __device__ bool operator()(const int &x) {
       return x > -1;
  }
};

void RunCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *rvd_seqs, unsigned int *rvd_lengths, double *cutoffs, unsigned int num_rvd_seqs, int c_upstream, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results) {
  
  unsigned int *d_rvd_seqs;
  double *d_scoring_matrix;
  size_t sm_pitch;
  
  cudaSafeCall( cudaMalloc(&d_rvd_seqs, PADDED_RVD_WIDTH * num_rvd_seqs * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_seqs, rvd_seqs, PADDED_RVD_WIDTH * num_rvd_seqs * sizeof(unsigned int), cudaMemcpyHostToDevice) );
  
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 5 * sizeof(double), scoring_matrix_length * sizeof(double)) );
  
  for (unsigned int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(double) * 5, cudaMemcpyHostToDevice) );
  }
  
  dim3 score_threadsPerBlock(32, 14);
  
  gzFile seqfile = gzopen(seq_filename, "r");

  kseq_t *seq = kseq_init(seqfile);
  int result;

  while ((result = kseq_read(seq)) >= 0) {

    unsigned char *d_prelim_results;
    char *d_reference_sequence;

    char *reference_sequence = seq->seq.s;
    unsigned long reference_sequence_length = ((seq->seq.l + 31) / 32 ) * 32;
    
    for (unsigned long i = seq->seq.l; i < reference_sequence_length - 1; i++) {
      reference_sequence[i] = 'X';
    }
    
    reference_sequence[reference_sequence_length- 1] = '\0';

    logger(log_file, "Scanning %s for off-target sites (length %ld)", seq->name.s, seq->seq.l);

    cudaSafeCall( cudaMalloc(&d_reference_sequence, reference_sequence_length * sizeof(char)) );
    cudaSafeCall( cudaMemcpy(d_reference_sequence, reference_sequence, reference_sequence_length * sizeof(char), cudaMemcpyHostToDevice) );

    cudaSafeCall( cudaMalloc(&d_prelim_results, reference_sequence_length * sizeof(unsigned char)) );
    
    thrust::device_ptr<unsigned char> prelim_results_start(d_prelim_results);
    thrust::device_ptr<unsigned char> prelim_results_end(d_prelim_results + reference_sequence_length);

    int score_blocks_needed = (reference_sequence_length + SCORE_THREADS_PER_BLOCK - 1) / SCORE_THREADS_PER_BLOCK;

    int score_block_x = (score_blocks_needed >= MAX_BLOCKS_PER_GRID ? MAX_BLOCKS_PER_GRID : score_blocks_needed);
    int score_block_y = (score_blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;

    dim3 score_blocksPerGrid(score_block_x, score_block_y);
    
    for (int i = 0; i < num_rvd_seqs; i++) {
      
      cudaSafeCall( cudaMemset(d_prelim_results, '\0', reference_sequence_length * sizeof(unsigned char)) );
      
      ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_seqs + i * PADDED_RVD_WIDTH, rvd_lengths[i], cutoffs[i], c_upstream, 0, d_scoring_matrix, sm_pitch, d_prelim_results);
      cudaSafeCall( cudaGetLastError() );
      
      results[i] += thrust::count_if(prelim_results_start, prelim_results_end, first_t_or_c());
      results[i] += thrust::count_if(prelim_results_start, prelim_results_end, last_a_or_g());

      
    }

    cudaSafeCall( cudaFree(d_prelim_results) );
    cudaSafeCall( cudaFree(d_reference_sequence) );
    
  }

  kseq_destroy(seq);
  gzclose(seqfile);
  
  cudaSafeCall( cudaFree(d_rvd_seqs) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );
  
}

void RunPairedCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *spacer_sizes, unsigned int *rvd_pairs, unsigned int *rvd_lengths, double *cutoffs, unsigned int num_rvd_pairs, int c_upstream, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results) {
  
  unsigned int *d_rvd_pairs;
  double *d_scoring_matrix;
  size_t sm_pitch;
  
  cudaSafeCall( cudaMalloc(&d_rvd_pairs, 2 * PADDED_RVD_WIDTH * num_rvd_pairs * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_pairs, rvd_pairs, 2 * PADDED_RVD_WIDTH * num_rvd_pairs * sizeof(unsigned int), cudaMemcpyHostToDevice) );
  
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 5 * sizeof(double), scoring_matrix_length * sizeof(double)) );
  
  for (unsigned int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(double) * 5, cudaMemcpyHostToDevice) );
  }
  
  dim3 score_threadsPerBlock(32, 14);
  dim3 tally_threadsPerBlock(32, 24);
  
  gzFile seqfile = gzopen(seq_filename, "r");

  kseq_t *seq = kseq_init(seqfile);
  int result;

  while ((result = kseq_read(seq)) >= 0) {

    unsigned char *d_prelim_results;
    unsigned int *d_second_results;
    char *d_reference_sequence;

    char *reference_sequence = seq->seq.s;
    unsigned long reference_sequence_length = ((seq->seq.l + 31) / 32 ) * 32;
    
    for (unsigned long i = seq->seq.l; i < reference_sequence_length - 1; i++) {
      reference_sequence[i] = 'X';
    }
    
    reference_sequence[reference_sequence_length- 1] = '\0';

    logger(log_file, "Scanning %s for off-target sites (length %ld)", seq->name.s, seq->seq.l);

    cudaSafeCall( cudaMalloc(&d_reference_sequence, reference_sequence_length * sizeof(char)) );
    cudaSafeCall( cudaMemcpy(d_reference_sequence, reference_sequence, reference_sequence_length * sizeof(char), cudaMemcpyHostToDevice) );

    cudaSafeCall( cudaMalloc(&d_prelim_results, reference_sequence_length * sizeof(unsigned char)) );
    cudaSafeCall( cudaMalloc(&d_second_results, reference_sequence_length * sizeof(unsigned int)) );
    
    thrust::device_ptr<unsigned int> second_results_start(d_second_results);
    thrust::device_ptr<unsigned int> second_results_end(d_second_results + reference_sequence_length);

    int score_blocks_needed = (reference_sequence_length + SCORE_THREADS_PER_BLOCK - 1) / SCORE_THREADS_PER_BLOCK;

    int score_block_x = (score_blocks_needed >= MAX_BLOCKS_PER_GRID ? MAX_BLOCKS_PER_GRID : score_blocks_needed);
    int score_block_y = (score_blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;

    dim3 score_blocksPerGrid(score_block_x, score_block_y);

    int tally_blocks_needed = (reference_sequence_length + TALLY_THREADS_PER_BLOCK - 1) / TALLY_THREADS_PER_BLOCK;

    int tally_block_x = (tally_blocks_needed >= MAX_BLOCKS_PER_GRID ? MAX_BLOCKS_PER_GRID : tally_blocks_needed);
    int tally_block_y = (tally_blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;

    dim3 tally_blocksPerGrid(tally_block_x, tally_block_y);
    
    for (int i = 0; i < num_rvd_pairs; i++) {
      
      unsigned int *pair_final_results = results + (4 * i);
      unsigned int pair_temp_results[4];
      
      cudaSafeCall( cudaMemset(d_prelim_results, '\0', reference_sequence_length * sizeof(unsigned char)) );
      cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
      
      int first_index = 2 * i;
      int second_index = first_index + 1;

      ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_pairs + first_index * PADDED_RVD_WIDTH, rvd_lengths[first_index], cutoffs[first_index], c_upstream, 0, d_scoring_matrix, sm_pitch, d_prelim_results);
      cudaSafeCall( cudaGetLastError() );

      ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_pairs + second_index * PADDED_RVD_WIDTH, rvd_lengths[second_index], cutoffs[second_index], c_upstream, 1, d_scoring_matrix, sm_pitch, d_prelim_results);
      cudaSafeCall( cudaGetLastError() );

      TallyResults<<<tally_blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_lengths[first_index], c_upstream, 0, 1, spacer_sizes[0], spacer_sizes[1], d_second_results);
      cudaSafeCall( cudaGetLastError() );
      
      pair_temp_results[0] = thrust::reduce(second_results_start, second_results_end);
      cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
      
      TallyResults<<<tally_blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_lengths[first_index], c_upstream, 0, 3, spacer_sizes[0], spacer_sizes[1], d_second_results);
      cudaSafeCall( cudaGetLastError() );
      
      pair_temp_results[1] = thrust::reduce(second_results_start, second_results_end);
      cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
      
      TallyResults<<<tally_blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_lengths[second_index], c_upstream, 2, 1, spacer_sizes[0], spacer_sizes[1], d_second_results);
      cudaSafeCall( cudaGetLastError() );
      
      pair_temp_results[2] = thrust::reduce(second_results_start, second_results_end);
      cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
      
      TallyResults<<<tally_blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_lengths[second_index], c_upstream, 2, 3, spacer_sizes[0], spacer_sizes[1], d_second_results);
      cudaSafeCall( cudaGetLastError() );
      
      pair_temp_results[3] = thrust::reduce(second_results_start, second_results_end);
      
      pair_final_results[0] += pair_temp_results[0];
      pair_final_results[1] += pair_temp_results[1];
      pair_final_results[2] += pair_temp_results[2];
      pair_final_results[3] += pair_temp_results[3];
      
    }

    cudaSafeCall( cudaFree(d_prelim_results) );
    cudaSafeCall( cudaFree(d_second_results) );
    cudaSafeCall( cudaFree(d_reference_sequence) );
    
  }

  kseq_destroy(seq);
  gzclose(seqfile);
  
  cudaSafeCall( cudaFree(d_rvd_pairs) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );
  
}

void RunPairedFindBindingSitesKeepScores_init(unsigned int **d_rvd_pair_p, double **d_scoring_matrix_p, size_t *sm_pitch_p, unsigned char **d_prelim_results_p, int **d_prelim_results_indexes_p, char **d_reference_sequence_p, unsigned char **prelim_results_p, int **prelim_results_indexes_p, unsigned long *reference_window_size_p, int *score_block_x_p, int *score_block_y_p, unsigned int **rvd_pair, double **scoring_matrix, unsigned int *rvd_lengths, unsigned int scoring_matrix_length) {
  
  unsigned int *d_rvd_pair;
  double *d_scoring_matrix;
  size_t sm_pitch;
  unsigned char *d_prelim_results;
  int *d_prelim_results_indexes;
  unsigned char *prelim_results;
  int *prelim_results_indexes;
  char *d_reference_sequence;
  // must be divisible by 32, pref power of 2
  unsigned long reference_window_size = 134217728;// 2^27
  int score_block_x;
  int score_block_y;
  
  cudaSafeCall( cudaMalloc(&d_rvd_pair, 2 * PADDED_RVD_WIDTH * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_pair, rvd_pair[0], rvd_lengths[0] * sizeof(unsigned int), cudaMemcpyHostToDevice) );
  cudaSafeCall( cudaMemcpy(d_rvd_pair + PADDED_RVD_WIDTH, rvd_pair[1], rvd_lengths[1] * sizeof(unsigned int), cudaMemcpyHostToDevice) );
  
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 5 * sizeof(double), scoring_matrix_length * sizeof(double)) );
  
  for (unsigned int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(double) * 5, cudaMemcpyHostToDevice) );
  }
  
  cudaSafeCall( cudaMalloc(&d_reference_sequence, reference_window_size * sizeof(char)) );
  cudaSafeCall( cudaMalloc(&d_prelim_results, reference_window_size * sizeof(unsigned char)) );
  cudaSafeCall( cudaMalloc(&d_prelim_results_indexes, reference_window_size * sizeof(int)) );
  
  cudaSafeCall( cudaMallocHost(&prelim_results, 400000000 * sizeof(unsigned char)) );
  cudaSafeCall( cudaMallocHost(&prelim_results_indexes, 400000000 * sizeof(int)) );

  int score_blocks_needed = (reference_window_size + SCORE_THREADS_PER_BLOCK - 1) / SCORE_THREADS_PER_BLOCK;

  score_block_x = (score_blocks_needed >= MAX_BLOCKS_PER_GRID ? MAX_BLOCKS_PER_GRID : score_blocks_needed);
  score_block_y = (score_blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;
  
  *d_rvd_pair_p = d_rvd_pair;
  *d_scoring_matrix_p = d_scoring_matrix;
  *sm_pitch_p = sm_pitch;
  *d_prelim_results_p = d_prelim_results;
  *d_prelim_results_indexes_p = d_prelim_results_indexes;
  *prelim_results_p = prelim_results;
  *prelim_results_indexes_p = prelim_results_indexes;
  *d_reference_sequence_p = d_reference_sequence;
  *reference_window_size_p = reference_window_size;
  *score_block_x_p = score_block_x;
  *score_block_y_p = score_block_y;
  
}

int RunPairedFindBindingSitesKeepScores(char *d_reference_sequence, unsigned int *d_rvd_pairs, double *d_scoring_matrix, size_t sm_pitch, unsigned char *d_prelim_results, int *d_prelim_results_indexes, unsigned char *prelim_results, int *prelim_results_indexes, unsigned long reference_window_size, int score_block_x, int score_block_y, unsigned int *rvd_lengths, char *ref_seq, unsigned long ref_seq_len, double *cutoffs, int c_upstream) {
  
  dim3 score_threadsPerBlock(32, 14);
  dim3 score_blocksPerGrid(score_block_x, score_block_y);
  
  thrust::device_ptr<int> prelim_results_indexes_start(d_prelim_results_indexes);
  thrust::device_ptr<int> prelim_results_indexes_end(d_prelim_results_indexes + reference_window_size);
  thrust::device_ptr<unsigned char> prelim_results_start(d_prelim_results);
  thrust::device_ptr<unsigned char> prelim_results_end(d_prelim_results + reference_window_size);

  int keepers_end_pos = 0;

  int max_rvd_len = (rvd_lengths[0] > rvd_lengths[1]) ? rvd_lengths[0] : rvd_lengths[1];
  int usable_tile_size = reference_window_size - (((max_rvd_len - 1) + 31) / 32 ) * 32;

  int iterations_needed = (ref_seq_len + (usable_tile_size - 1)) / usable_tile_size;
  
  memset(prelim_results, '\0', 400000000 * sizeof(unsigned char));
  
  for (int i = 0; i < iterations_needed; i++) {

    int copy_offset = usable_tile_size * i;

    int copy_num;

    if (ref_seq_len - copy_offset <= reference_window_size)
      copy_num = ref_seq_len - copy_offset;
    else
      copy_num = reference_window_size;

    cudaSafeCall( cudaMemset(d_reference_sequence, 'X', reference_window_size * sizeof(unsigned char)) );
    cudaSafeCall( cudaMemset(d_reference_sequence + reference_window_size - 1, '\0', 1 * sizeof(unsigned char)) );
    cudaSafeCall( cudaMemcpy(d_reference_sequence, ref_seq + copy_offset, copy_num * sizeof(char), cudaMemcpyHostToDevice) );

    cudaSafeCall( cudaMemset(d_prelim_results, '\0', reference_window_size * sizeof(unsigned char)) );
    cudaSafeCall( cudaMemset(d_prelim_results_indexes, '\0', reference_window_size * sizeof(unsigned int)) );
    
    ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_window_size, d_rvd_pairs + 0, rvd_lengths[0], cutoffs[0], c_upstream, 0, d_scoring_matrix, sm_pitch, d_prelim_results);
    cudaSafeCall( cudaGetLastError() );

    ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_window_size, d_rvd_pairs + PADDED_RVD_WIDTH, rvd_lengths[1], cutoffs[1], c_upstream, 1, d_scoring_matrix, sm_pitch, d_prelim_results);
    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall( cudaMemcpy(prelim_results + copy_offset, d_prelim_results, copy_num * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaGetLastError() );

    thrust::sequence(prelim_results_indexes_start, prelim_results_indexes_end);
    thrust::transform(prelim_results_start, prelim_results_end, prelim_results_indexes_start, prelim_results_indexes_start, index_to_keep_paired(copy_offset));
    thrust::sort(prelim_results_indexes_start, prelim_results_indexes_end, thrust::greater<int>());
    thrust::device_ptr<int> keepers_end = thrust::min_element(prelim_results_indexes_start, prelim_results_indexes_end);

    cudaSafeCall( cudaMemcpy(prelim_results_indexes + keepers_end_pos, d_prelim_results_indexes, (keepers_end - prelim_results_indexes_start) * sizeof(int), cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaGetLastError() );

    keepers_end_pos += (keepers_end - prelim_results_indexes_start);

  }

  return keepers_end_pos;

}

void RunPairedFindBindingSitesKeepScores_cleanup(char *d_reference_sequence, unsigned int *d_rvd_pairs, double *d_scoring_matrix, unsigned char *d_prelim_results, int *d_prelim_results_indexes, unsigned char *prelim_results, int *prelim_results_indexes) {
  cudaSafeCall( cudaFree(d_prelim_results) );
  cudaSafeCall( cudaFree(d_prelim_results_indexes) );
  cudaSafeCall( cudaFreeHost(prelim_results) );
  cudaSafeCall( cudaFreeHost(prelim_results_indexes) );
  cudaSafeCall( cudaFree(d_reference_sequence) );
  cudaSafeCall( cudaFree(d_rvd_pairs) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );
}

void RunFindBindingSitesKeepScores_init(unsigned int **d_rvd_seq_p, double **d_scoring_matrix_p, size_t *sm_pitch_p, unsigned char **d_prelim_results_p, int **d_prelim_results_indexes_p, char **d_reference_sequence_p, unsigned char **prelim_results_p, int **prelim_results_indexes_p, unsigned long *reference_window_size_p, int *score_block_x_p, int *score_block_y_p, unsigned int *rvd_seq, double **scoring_matrix, unsigned int rvd_length, unsigned int scoring_matrix_length) {
  
  unsigned int *d_rvd_seq;
  double *d_scoring_matrix;
  size_t sm_pitch;
  unsigned char *d_prelim_results;
  int *d_prelim_results_indexes;
  unsigned char *prelim_results;
  int *prelim_results_indexes;
  char *d_reference_sequence;
  // must be divisible by 32, pref power of 2
  unsigned long reference_window_size = 134217728;// 2^27
  int score_block_x;
  int score_block_y;
  
  cudaSafeCall( cudaMalloc(&d_rvd_seq, 2 * PADDED_RVD_WIDTH * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_seq, rvd_seq, rvd_length * sizeof(unsigned int), cudaMemcpyHostToDevice) );
  
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 5 * sizeof(double), scoring_matrix_length * sizeof(double)) );
  
  for (unsigned int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(double) * 5, cudaMemcpyHostToDevice) );
  }
  
  cudaSafeCall( cudaMalloc(&d_reference_sequence, reference_window_size * sizeof(char)) );
  cudaSafeCall( cudaMalloc(&d_prelim_results, reference_window_size * sizeof(unsigned char)) );
  cudaSafeCall( cudaMalloc(&d_prelim_results_indexes, reference_window_size * sizeof(int)) );
  
  cudaSafeCall( cudaMallocHost(&prelim_results, 400000000 * sizeof(unsigned char)) );
  cudaSafeCall( cudaMallocHost(&prelim_results_indexes, 400000000 * sizeof(int)) );

  int score_blocks_needed = (reference_window_size + SCORE_THREADS_PER_BLOCK - 1) / SCORE_THREADS_PER_BLOCK;

  score_block_x = (score_blocks_needed >= MAX_BLOCKS_PER_GRID ? MAX_BLOCKS_PER_GRID : score_blocks_needed);
  score_block_y = (score_blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;
  
  *d_rvd_seq_p = d_rvd_seq;
  *d_scoring_matrix_p = d_scoring_matrix;
  *sm_pitch_p = sm_pitch;
  *d_prelim_results_p = d_prelim_results;
  *d_prelim_results_indexes_p = d_prelim_results_indexes;
  *prelim_results_p = prelim_results;
  *prelim_results_indexes_p = prelim_results_indexes;
  *d_reference_sequence_p = d_reference_sequence;
  *reference_window_size_p = reference_window_size;
  *score_block_x_p = score_block_x;
  *score_block_y_p = score_block_y;
  
}

int RunFindBindingSitesKeepScores(char *d_reference_sequence, unsigned int *d_rvd_seq, double *d_scoring_matrix, size_t sm_pitch, unsigned char *d_prelim_results, int *d_prelim_results_indexes, unsigned char *prelim_results, int *prelim_results_indexes, unsigned long reference_window_size, int score_block_x, int score_block_y, unsigned int rvd_length, char *ref_seq, unsigned long ref_seq_len, double cutoff, int c_upstream) {
  
  dim3 score_threadsPerBlock(32, 14);
  dim3 score_blocksPerGrid(score_block_x, score_block_y);
  
  thrust::device_ptr<int> prelim_results_indexes_start(d_prelim_results_indexes);
  thrust::device_ptr<int> prelim_results_indexes_end(d_prelim_results_indexes + reference_window_size);
  thrust::device_ptr<unsigned char> prelim_results_start(d_prelim_results);
  thrust::device_ptr<unsigned char> prelim_results_end(d_prelim_results + reference_window_size);

  int keepers_end_pos = 0;

  int usable_tile_size = reference_window_size - (((rvd_length - 1) + 31) / 32 ) * 32;

  int iterations_needed = (ref_seq_len + (usable_tile_size - 1)) / usable_tile_size;
  
  memset(prelim_results, '\0', 400000000 * sizeof(unsigned char));
  
  for (int i = 0; i < iterations_needed; i++) {

    int copy_offset = usable_tile_size * i;

    int copy_num;

    if (ref_seq_len - copy_offset <= reference_window_size)
      copy_num = ref_seq_len - copy_offset;
    else
      copy_num = reference_window_size;

    cudaSafeCall( cudaMemset(d_reference_sequence, 'X', reference_window_size * sizeof(unsigned char)) );
    cudaSafeCall( cudaMemset(d_reference_sequence + reference_window_size - 1, '\0', 1 * sizeof(unsigned char)) );
    cudaSafeCall( cudaMemcpy(d_reference_sequence, ref_seq + copy_offset, copy_num * sizeof(char), cudaMemcpyHostToDevice) );

    cudaSafeCall( cudaMemset(d_prelim_results, '\0', reference_window_size * sizeof(unsigned char)) );
    cudaSafeCall( cudaMemset(d_prelim_results_indexes, '\0', reference_window_size * sizeof(unsigned int)) );
    
    ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_window_size, d_rvd_seq, rvd_length, cutoff, c_upstream, 0, d_scoring_matrix, sm_pitch, d_prelim_results);
    cudaSafeCall( cudaGetLastError() );

    cudaSafeCall( cudaMemcpy(prelim_results + copy_offset, d_prelim_results, copy_num * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaGetLastError() );

    thrust::sequence(prelim_results_indexes_start, prelim_results_indexes_end);
    thrust::transform(prelim_results_start, prelim_results_end, prelim_results_indexes_start, prelim_results_indexes_start, index_to_keep(copy_offset));
    thrust::sort(prelim_results_indexes_start, prelim_results_indexes_end, thrust::greater<int>());
    thrust::device_ptr<int> keepers_end = thrust::min_element(prelim_results_indexes_start, prelim_results_indexes_end);

    cudaSafeCall( cudaMemcpy(prelim_results_indexes + keepers_end_pos, d_prelim_results_indexes, (keepers_end - prelim_results_indexes_start) * sizeof(int), cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaGetLastError() );

    keepers_end_pos += (keepers_end - prelim_results_indexes_start);

  }

  return keepers_end_pos;

}

void RunFindBindingSitesKeepScores_cleanup(char *d_reference_sequence, unsigned int *d_rvd_seq, double *d_scoring_matrix, unsigned char *d_prelim_results, int *d_prelim_results_indexes, unsigned char *prelim_results, int *prelim_results_indexes) {
  cudaSafeCall( cudaFree(d_prelim_results) );
  cudaSafeCall( cudaFree(d_prelim_results_indexes) );
  cudaSafeCall( cudaFreeHost(prelim_results) );
  cudaSafeCall( cudaFreeHost(prelim_results_indexes) );
  cudaSafeCall( cudaFree(d_reference_sequence) );
  cudaSafeCall( cudaFree(d_rvd_seq) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );
}


