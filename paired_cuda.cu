#undef _GLIBCXX_USE_INT128
#undef _GLIBCXX_ATOMIC_BUILTINS

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "paired_cuda.h"

// Sequence handling
#include <zlib.h>
#include "kseq.h"
KSEQ_INIT(gzFile, gzread)

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65535

#define cudaSafeCall(call){   \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "%s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

__device__ double ScoringMatrixVal(double *scoring_matrix, size_t pitch, unsigned int row, unsigned int column);
double *ScoringMatrixRow(double *scoring_matrix, size_t pitch, unsigned int row);

//template< unsigned int STRAND, unsigned int RVD_NUM >
__global__ void ScoreBindingSites(char *input_sequence, unsigned long is_length, unsigned int *rvd_sequence, unsigned int rs_len, double cutoff, unsigned int strand, unsigned int rvd_num, double *scoring_matrix, size_t sm_pitch, unsigned char *results) {
   
  int block_seq_index = MAX_THREADS_PER_BLOCK * (blockIdx.y * gridDim.x + blockIdx.x);
  int seq_index = block_seq_index + (blockDim.x * threadIdx.y) + threadIdx.x;
  
  if (seq_index < 1 || seq_index >= is_length || seq_index + rs_len >= is_length - 1) return;
  
  if (input_sequence[seq_index - 1] == 'T' || input_sequence[seq_index - 1] == 't') {
    
    double thread_result = 0;
    
    for (int i = 0; i < rs_len; i++) {
      
      int rvd_index = i;
      
      int sm_col = 0;
      
      char base = input_sequence[seq_index + i];
      
      if (base == 'A' || base == 'a')    
        sm_col = 0;
      if (base == 'C' || base == 'c')
        sm_col = 1;
      if (base == 'G' || base == 'g')
        sm_col = 2;
      if (base == 'T' || base == 't')
        sm_col = 3;
      
      thread_result += ScoringMatrixVal(scoring_matrix, sm_pitch, rvd_sequence[rvd_index], sm_col);
      
    }
    
    results[seq_index] |= (thread_result < cutoff ? 1UL : 0UL) << (2 * rvd_num + 0);
    
  } 
  
  if (input_sequence[seq_index + rs_len] == 'A' || input_sequence[seq_index + rs_len] == 'a') {
    
    double thread_result = 0;
    
    for (int i = 0; i < rs_len; i++) {
      
      int rvd_index = rs_len - i - 1;
      
      int sm_col = 0;
      
      char base = input_sequence[seq_index + i];
      
      if (base == 'A' || base == 'a')    
        sm_col = 3;
      if (base == 'C' || base == 'c')
        sm_col = 2;
      if (base == 'G' || base == 'g')
        sm_col = 1;
      if (base == 'T' || base == 't')
        sm_col = 0;
      
      thread_result += ScoringMatrixVal(scoring_matrix, sm_pitch, rvd_sequence[rvd_index], sm_col);
      
    }
    
    results[seq_index] |= (thread_result < cutoff ? 1UL : 0UL) << (2 * rvd_num + 1);
    
  }
  
}

__global__ void TallyResults(unsigned char *prelim_results, unsigned int pr_length, unsigned int rs_len, unsigned int u_shift, unsigned int d_shift, unsigned int spacer_range_start, unsigned int spacer_range_end, unsigned int *second_results) {
    
  int thread_result = 0;
  
  int block_seq_index = MAX_THREADS_PER_BLOCK * (blockIdx.y * gridDim.x + blockIdx.x);
  int seq_index = block_seq_index + (blockDim.x * threadIdx.y) + threadIdx.x;
  
  if (seq_index < 0 || seq_index >= pr_length) return;
  
  for (int i = spacer_range_start; i <= spacer_range_end; i++) {
    
    if (seq_index + rs_len + i >= pr_length) continue;
    
    thread_result += ((prelim_results[seq_index] & (1UL << u_shift)) && 
                      (prelim_results[seq_index + rs_len + i] & (1UL << d_shift)));
  }
  
  second_results[seq_index] = thread_result;
  
}

__device__ double ScoringMatrixVal(double *scoring_matrix, size_t pitch, unsigned int row, unsigned int column) {
  
  return *((double*)((char*) scoring_matrix + row * pitch) + column);
  
}

double *ScoringMatrixRow(double *scoring_matrix, size_t pitch, unsigned int row) {
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
    printf("[%.2f, %.2f, %.2f, %.2f, %.2f]\n",
           array[y][0],
           array[y][1],
           array[y][2],
           array[y][3],
           array[y][4]);
  }
}

void printRvdArray(unsigned int *array) {
  for (int y = 0; y < 32; y++) {
    printf("%du ", array[y]);
  }
  printf("\n");
}

void RunCountBindingSites(char *seq_filename, unsigned int *spacer_sizes, unsigned int **rvd_sequences, unsigned int *rvd_sequence_lengths, double *cutoffs, double **scoring_matrix, unsigned int scoring_matrix_length, unsigned int **results) {
  
  unsigned int *d_rvd_sequence;
  unsigned int *d_rvd_sequence2;
  double *d_scoring_matrix;
  size_t sm_pitch;
  cudaEvent_t start, stop;
  float elapsed;
  
  cudaSafeCall( cudaMalloc(&d_rvd_sequence, 32 * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_sequence, rvd_sequences[0], 32 * sizeof(unsigned int), cudaMemcpyHostToDevice) );

  cudaSafeCall( cudaMalloc(&d_rvd_sequence2, 32 * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_sequence2, rvd_sequences[1], 32 * sizeof(unsigned int), cudaMemcpyHostToDevice) );
    
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 4 * sizeof(double), scoring_matrix_length * sizeof(double)) );
  
  for (unsigned int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(double) * 4, cudaMemcpyHostToDevice) );
  }
  
  dim3 threadsPerBlock(32, 32);
  
  gzFile seqfile = gzopen(seq_filename, "r");

  kseq_t *seq = kseq_init(seqfile);
  int result;

  while ((result = kseq_read(seq)) >= 0) {
    
    unsigned char *d_prelim_results;
    unsigned int *d_second_results;
    char *d_reference_sequence;
    unsigned int h_results[4];
    
    char *reference_sequence = seq->seq.s;
    
    for (int i = seq->seq.l; i < seq->seq.m; i++) {
      reference_sequence[i] = 'X';
    }
    
    reference_sequence[seq->seq.m - 1] = '\0';
    
    unsigned long reference_sequence_length = seq->seq.m;
    
    cudaSafeCall( cudaMalloc(&d_reference_sequence, reference_sequence_length * sizeof(char)) );
    cudaSafeCall( cudaMemcpy(d_reference_sequence, reference_sequence, reference_sequence_length * sizeof(char), cudaMemcpyHostToDevice) );
  
    cudaSafeCall( cudaMalloc(&d_prelim_results, reference_sequence_length * sizeof(unsigned char)) );
    cudaSafeCall( cudaMemset(d_prelim_results, '\0', reference_sequence_length * sizeof(unsigned char)) );
    
    cudaSafeCall( cudaMalloc(&d_second_results, reference_sequence_length * sizeof(unsigned int)) );
    cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
    
    thrust::device_ptr<unsigned int> second_results_start(d_second_results);
    thrust::device_ptr<unsigned int> second_results_end(d_second_results + reference_sequence_length);

    int blocks_needed = reference_sequence_length / 1024;
    int block_y = (blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;
    dim3 blocksPerGrid(MAX_BLOCKS_PER_GRID, block_y);
    
    cudaSafeCall( cudaEventCreate(&start) );
    cudaSafeCall( cudaEventCreate(&stop) );
    
    cudaSafeCall( cudaEventRecord(start, 0) );
    
    ScoreBindingSites <<<blocksPerGrid, threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_sequence, rvd_sequence_lengths[0], cutoffs[0], 0, 0, d_scoring_matrix, sm_pitch, d_prelim_results);
    cudaSafeCall( cudaGetLastError() );

    ScoreBindingSites <<<blocksPerGrid, threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, d_rvd_sequence2, rvd_sequence_lengths[1], cutoffs[1], 0, 1, d_scoring_matrix, sm_pitch, d_prelim_results);
    cudaSafeCall( cudaGetLastError() );
    
    cudaSafeCall( cudaDeviceSynchronize() );
    
    cudaSafeCall( cudaEventRecord(stop, 0) );
    cudaSafeCall( cudaEventSynchronize(stop) );
    cudaSafeCall( cudaEventElapsedTime(&elapsed, start, stop) );
    
    printf("%.2f ms to score binding sites\n", elapsed);
    
    cudaSafeCall( cudaEventDestroy(stop) );
    cudaSafeCall( cudaEventDestroy(start) );
    
    
    cudaSafeCall( cudaEventCreate(&start) );
    cudaSafeCall( cudaEventCreate(&stop) );
    
    cudaSafeCall( cudaEventRecord(start, 0) );
    
    TallyResults<<<blocksPerGrid, threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[0], 0, 1, spacer_sizes[0], spacer_sizes[1], d_second_results);
    cudaSafeCall( cudaGetLastError() );
    
    h_results[0] = thrust::reduce(second_results_start, second_results_end);
    cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
    
    TallyResults<<<blocksPerGrid, threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[0], 0, 3, spacer_sizes[0], spacer_sizes[1], d_second_results);
    cudaSafeCall( cudaGetLastError() );
    
    h_results[1] = thrust::reduce(second_results_start, second_results_end);
    cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
    
    TallyResults<<<blocksPerGrid, threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[1], 2, 1, spacer_sizes[0], spacer_sizes[1], d_second_results);
    cudaSafeCall( cudaGetLastError() );
    
    h_results[2] = thrust::reduce(second_results_start, second_results_end);
    cudaSafeCall( cudaMemset(d_second_results, '\0', reference_sequence_length * sizeof(unsigned int)) );
    
    TallyResults<<<blocksPerGrid, threadsPerBlock>>>(d_prelim_results, reference_sequence_length, rvd_sequence_lengths[1], 2, 3, spacer_sizes[0], spacer_sizes[1], d_second_results);
    cudaSafeCall( cudaGetLastError() );
    
    h_results[3] = thrust::reduce(second_results_start, second_results_end);
    
    cudaSafeCall( cudaDeviceSynchronize() );
    
    cudaSafeCall( cudaEventRecord(stop, 0) );
    cudaSafeCall( cudaEventSynchronize(stop) );
    
    cudaSafeCall( cudaEventElapsedTime(&elapsed, start, stop) );
    
    printf("%.2f ms to tally results\n", elapsed);
    
    cudaSafeCall( cudaEventDestroy(stop) );
    cudaSafeCall( cudaEventDestroy(start) );
  
    results[0][0] += h_results[0];
    results[0][1] += h_results[1];
    results[1][0] += h_results[2];
    results[1][1] += h_results[3];
    
    printf("%d %d %d %d\n", h_results[0], h_results[1], h_results[2], h_results[3]);
    
    cudaSafeCall( cudaFree(d_prelim_results) );
    cudaSafeCall( cudaFree(d_second_results) );
    cudaSafeCall( cudaFree(d_reference_sequence) );
    
  }

  kseq_destroy(seq);
  gzclose(seqfile);
  
  cudaSafeCall( cudaFree(d_rvd_sequence) );
  cudaSafeCall( cudaFree(d_rvd_sequence2) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );
  
}
