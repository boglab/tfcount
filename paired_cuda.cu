#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "paired_cuda.h"

#define MAX_THREADS_PER_BLOCK 256
#define MAX_BLOCKS_PER_GRID 65535

#define cudaSafeCall(call){   \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "%s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

#define imin(a,b) (a<b?a:b)

__device__ double ScoringMatrixVal(double *scoring_matrix, size_t pitch, int row, int column);
double *ScoringMatrixRow(double *scoring_matrix, size_t pitch, int row);

template< int STRAND, int RVD_NUM >
__global__ void ScoreBindingSites(char *input_sequence, int is_length, unsigned int *rvd_sequence, int rs_len, double cutoff, double *scoring_matrix, size_t sm_pitch, unsigned char *results) {
  
  // replace non-existant variable
  __shared__ double cache[MAX_THREADS_PER_BLOCK];

  int cache_index = threadIdx.x;

  cache[cache_index] = 0;

  __syncthreads();
  
  int seq_index = blockIdx.y * gridDim.x + blockIdx.x;
  int rvd_index = (STRAND == 1 ? rs_len - threadIdx.x - 1 : threadIdx.x);

  if (rvd_index < 0 || rvd_index >= rs_len) return;
  if (seq_index < 1 || seq_index >= is_length  || seq_index + rs_len >= is_length - 1) return;
  if (STRAND == 0 && input_sequence[seq_index - 1] != 'T' && input_sequence[seq_index - 1] != 't') return;
  if (STRAND == 1 && input_sequence[seq_index + rs_len] != 'A' && input_sequence[seq_index + rs_len] != 'a') return;
  
  int sm_col = 0;

  char base = input_sequence[seq_index + threadIdx.x];

  if (base == 'A' || base == 'a')    
    sm_col = (STRAND == 1 ? 3 : 0);
  if (base == 'C' || base == 'c')
    sm_col = (STRAND == 1 ? 2 : 1);
  if (base == 'G' || base == 'g')
    sm_col = (STRAND == 1 ? 1 : 2);
  if (base == 'T' || base == 't')
    sm_col = (STRAND == 1 ? 0 : 3);
  
  cache[cache_index] = ScoringMatrixVal(scoring_matrix, sm_pitch, rvd_sequence[rvd_index], sm_col);
  
  __syncthreads();
  
  // for reductions, threadsPerBlock must be a power of 2
  // because of the following code
  
  int i = blockDim.x / 2;
  
  while (i != 0) {
      if (threadIdx.x < i)
          cache[cache_index] += cache[cache_index + i];
      __syncthreads();
      i /= 2;
  }
  
  if (threadIdx.x == 0 && cache[cache_index] < cutoff) {// && cache[0] < cutoff)
    results[seq_index] |= (1UL << (2 * RVD_NUM + STRAND));
  }
  
  __syncthreads();
  
}

template< int COMBO >
__global__ void TallyResults(unsigned char *prelim_results, int pr_length, int rs_len, int spacer_range_start, int spacer_range_end, unsigned int *final_results) {
  
  __shared__ int cache[MAX_THREADS_PER_BLOCK];

  int cache_index = threadIdx.x;

  cache[cache_index] = 0;
  
  int spacer_size = spacer_range_start + threadIdx.x;
  int seq_index = blockIdx.y * gridDim.x + blockIdx.x;
  
  if (spacer_size > spacer_range_end) return;
  if (seq_index >= pr_length  || seq_index + spacer_size + rs_len >= pr_length) return;
  
  // 0 f rvd1 r rvd1 
  // 1 f rvd1 r rvd2 
  // 2 f rvd2 r rvd1 
  // 3 f rvd2 r rvd2 

  int u_shift;
  int d_shift;

  if (COMBO == 0) {
    u_shift = 0;
    d_shift = 1;
  } else if (COMBO == 1) {
    u_shift = 0;
    d_shift = 3;
  } else if (COMBO == 2) {
    u_shift = 2;
    d_shift = 1;
  } else {
    u_shift = 2;
    d_shift = 3;
  }
  
  cache[cache_index] = prelim_results[seq_index] & (1UL << u_shift) &&
                       prelim_results[seq_index + rs_len + spacer_size] & (1UL << d_shift);
  
  if (seq_index == 17505324 && spacer_size == 18)
    while (true) break;
  
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  // because of the following code

  int i = blockDim.x / 2;

  while (i != 0) {
      if (threadIdx.x < i)
          cache[cache_index] += cache[cache_index + i];
      __syncthreads();
      i /= 2;
  }

  if (threadIdx.x == 0)
    atomicAdd(final_results + COMBO, cache[cache_index]);
  
}

__device__ double ScoringMatrixVal(double *scoring_matrix, size_t pitch, int row, int column) {
  
  return *((double*)((char*) scoring_matrix + row * pitch) + column);
  
}

double *ScoringMatrixRow(double *scoring_matrix, size_t pitch, int row) {
  return (double*)((char*) scoring_matrix + row * pitch);
}

void printDeviceMatrix(double *matrix, int width, int length) {
  for (int y = 0; y < length; y++) {
    double *row = ScoringMatrixRow(matrix, width, y);
    printf("[%.2f, %.2f, %.2f, %.2f]\n",
           row[0],
           row[1],
           row[2],
           row[3]);
  }
}

void printHostMatrix(double **array, int width, int length) {
  for (int y = 0; y < length; y++) {
    printf("[%.2f, %.2f, %.2f, %.2f]\n",
           array[y][0],
           array[y][1],
           array[y][2],
           array[y][3]);
  }
}

void RunCountBindingSites(char *reference_sequence, unsigned long reference_sequence_length, int *spacer_sizes, unsigned int **rvd_sequences, int *rvd_sequence_lengths, double *cutoffs, double **scoring_matrix, int scoring_matrix_length, unsigned int **results) {
   
  //size_t free, total;
  
  unsigned char *d_prelim_results;
  char *d_reference_sequence;
  unsigned int *d_rvd_sequence;
  unsigned int *d_rvd_sequence2;
  
  unsigned int *d_results;
  
  double *d_scoring_matrix;

  //double *h_scoring_matrix;

  //double *d_prelim_result_scores;
  //unsigned char *h_prelim_results;
  //double *h_prelim_result_scores;

  //unsigned char *h_prelim_results;
  
  // Input Sequence
  
  //cudaMemGetInfo(&free, &total);
  //printf("%d KB free of total %d KB\n", free/1024, total/1024);
  
  cudaSafeCall( cudaMalloc(&d_reference_sequence, reference_sequence_length * sizeof(char)) );
  cudaSafeCall( cudaMemcpy(d_reference_sequence, reference_sequence, reference_sequence_length * sizeof(char), cudaMemcpyHostToDevice) );

  cudaSafeCall( cudaMalloc(&d_prelim_results, reference_sequence_length * sizeof(unsigned char)) );
  cudaSafeCall( cudaMemset(d_prelim_results, '\0', reference_sequence_length * sizeof(unsigned char)) );


  //cudaSafeCall( cudaMalloc(&d_prelim_result_scores, reference_sequence_length * sizeof(double)) );
  //cudaSafeCall( cudaMemset(d_prelim_result_scores, '\0', reference_sequence_length * sizeof(double)) );

  //h_prelim_result_scores = (double *) malloc(reference_sequence_length * sizeof(double));
  //h_prelim_results = (unsigned char *) malloc(reference_sequence_length * sizeof(unsigned char));

  
  printf("Loaded input sequence\n");
  
  // RVD Sequences
  //cudaMemGetInfo(&free, &total);
  //printf("%d KB free of total %d KB\n", free/1024, total/1024);

  cudaSafeCall( cudaMalloc(&d_rvd_sequence, rvd_sequence_lengths[0] * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_sequence, rvd_sequences[0], rvd_sequence_lengths[0] * sizeof(unsigned int), cudaMemcpyHostToDevice) );

  cudaSafeCall( cudaMalloc(&d_rvd_sequence2, rvd_sequence_lengths[1] * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_sequence2, rvd_sequences[1], rvd_sequence_lengths[1] * sizeof(unsigned int), cudaMemcpyHostToDevice) );

  // Results
  cudaSafeCall( cudaMalloc(&d_results, 4 * sizeof(unsigned int)) );
  
  // Scoring Matrix
  
  size_t sm_pitch;
  
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 4 * sizeof(double), scoring_matrix_length * sizeof(double)) );
  
  for (int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(double) * 4, cudaMemcpyHostToDevice) );
  }

  //h_scoring_matrix = (double *) malloc(sm_pitch * scoring_matrix_length * sizeof(double));

  //cudaSafeCall( cudaMemcpy(h_scoring_matrix, d_scoring_matrix, sm_pitch * scoring_matrix_length * sizeof(double), cudaMemcpyDeviceToHost) );

  
  unsigned int h_results[4];
  
  h_results[0] = 0;
  h_results[1] = 0;
  h_results[2] = 0;
  h_results[3] = 0;
  
  cudaSafeCall( cudaMemcpy(d_results, h_results, sizeof(unsigned int) * 4, cudaMemcpyHostToDevice) );
  
  //(reference_sequence_length / threadsPerBlock.x) + 1
  
  int blocks_needed = reference_sequence_length;
  int block_y = (blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;

  //dim3 speshul_blocksPerGrid(1, 1);
  dim3 blocksPerGrid(MAX_BLOCKS_PER_GRID, block_y);

  //printf("blocksPerGrid %dx%d\n", blocksPerGrid.x, blocksPerGrid.y);

  dim3 rvd_threadsPerBlock(((rvd_sequence_lengths[0] + 15) / 16) * 16, 1);

  //printf("rvd_threadsPerBlock %dx%d\n", rvd_threadsPerBlock.x, rvd_threadsPerBlock.y);

  dim3 rvd2_threadsPerBlock(((rvd_sequence_lengths[1] + 15) / 16) * 16, 1);

  //printf("rvd2_threadsPerBlock %dx%d\n", rvd2_threadsPerBlock.x, rvd2_threadsPerBlock.y);

//  ScoreBindingSites <0, 0> <<<blocksPerGrid, rvd_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_sequence, rvd_sequence_lengths[0], cutoffs[0], d_scoring_matrix, sm_pitch, d_prelim_results, d_prelim_result_scores);

  ScoreBindingSites <0, 0> <<<blocksPerGrid, rvd_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_sequence, rvd_sequence_lengths[0], cutoffs[0], d_scoring_matrix, sm_pitch, d_prelim_results);
  //cudaSafeCall( cudaMemcpy(h_prelim_result_scores, d_prelim_result_scores, reference_sequence_length * sizeof(double), cudaMemcpyDeviceToHost) );
  //cudaSafeCall( cudaMemcpy(h_prelim_results, d_prelim_results, reference_sequence_length * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );

  ScoreBindingSites <1, 0> <<<blocksPerGrid, rvd_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_sequence, rvd_sequence_lengths[0], cutoffs[0], d_scoring_matrix, sm_pitch, d_prelim_results);
  //cudaSafeCall( cudaMemcpy(h_prelim_result_scores, d_prelim_result_scores, reference_sequence_length * sizeof(double), cudaMemcpyDeviceToHost) );
  //cudaSafeCall( cudaMemcpy(h_prelim_results, d_prelim_results, reference_sequence_length * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );

  ScoreBindingSites <0, 1> <<<blocksPerGrid, rvd2_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_sequence2, rvd_sequence_lengths[1], cutoffs[1], d_scoring_matrix, sm_pitch, d_prelim_results);
  //cudaSafeCall( cudaMemcpy(h_prelim_result_scores, d_prelim_result_scores, reference_sequence_length * sizeof(double), cudaMemcpyDeviceToHost) );
  //cudaSafeCall( cudaMemcpy(h_prelim_results, d_prelim_results, reference_sequence_length * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );

  ScoreBindingSites <1, 1> <<<blocksPerGrid, rvd2_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_sequence2, rvd_sequence_lengths[1], cutoffs[1], d_scoring_matrix, sm_pitch, d_prelim_results);
  //cudaSafeCall( cudaMemcpy(h_prelim_result_scores, d_prelim_result_scores, reference_sequence_length * sizeof(double), cudaMemcpyDeviceToHost) );
  //cudaSafeCall( cudaMemcpy(h_prelim_results, d_prelim_results, reference_sequence_length * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );
  
  //h_prelim_results = (unsigned char *) malloc(sizeof(unsigned char) * reference_sequence_length);

  //cudaSafeCall( cudaMemcpy(h_prelim_results, d_prelim_results, sizeof(unsigned char) * reference_sequence_length, cudaMemcpyDeviceToHost) );

  dim3 tally_threadsPerBlock(((spacer_sizes[1] - spacer_sizes[0] + 1) + 15) / 16 * 16, 1);
  
  TallyResults <0> <<<blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[0], spacer_sizes[0], spacer_sizes[1], d_results);
  //cudaSafeCall( cudaMemcpy(h_results, d_results, 4 * sizeof(int), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );
  
  TallyResults <1> <<<blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[0], spacer_sizes[0], spacer_sizes[1], d_results);
  //cudaSafeCall( cudaMemcpy(h_results, d_results, 4 * sizeof(int), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );
  
  TallyResults <2> <<<blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[1], spacer_sizes[0], spacer_sizes[1], d_results);
  //cudaSafeCall( cudaMemcpy(h_results, d_results, 4 * sizeof(int), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );
  
  TallyResults <3> <<<blocksPerGrid, tally_threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[1], spacer_sizes[0], spacer_sizes[1], d_results);
  //cudaSafeCall( cudaMemcpy(h_results, d_results, 4 * sizeof(int), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );
  
  cudaSafeCall( cudaMemcpy(h_results, d_results, 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
  cudaSafeCall( cudaGetLastError() );


  results[0][0] += h_results[0];
  results[0][1] += h_results[1];
  results[1][0] += h_results[2];
  results[1][1] += h_results[3];
  
  printf("%d %d %d %d\n", h_results[0], h_results[1], h_results[2], h_results[3]);
  
  cudaSafeCall( cudaFree(d_prelim_results) );
  cudaSafeCall( cudaFree(d_reference_sequence) );
  cudaSafeCall( cudaFree(d_rvd_sequence) );
  cudaSafeCall( cudaFree(d_rvd_sequence2) );
  cudaSafeCall( cudaFree(d_results) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );
  
}

