#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "paired_cuda.h"

#define CHUNK_SIZE 256

__global__ void MyKernel(char *input_sequence, int spacer_size_start, int **rvd_sequences, int *rvd_sequence_lengths, double *cutoffs, double **scoring_matrix, int **results);

void CountBindingSites(char *reference_sequence, unsigned long reference_sequence_length, int *spacer_sizes, int **rvd_sequences, int *rvd_sequence_lengths, double *cutoffs, double **scoring_matrix, int scoring_matrix_length, int **results) {
  
  char *d_reference_sequence;
  int **d_rvd_sequences;
  int *d_rvd_sequence_lengths;
  
  int **d_results;
  double *d_cutoffs;
  
  double **d_scoring_matrix;
  
  // Input Sequence
  cudaMalloc(&d_reference_sequence, reference_sequence_length * sizeof(char));
  cudaMemcpy(d_reference_sequence, reference_sequence, reference_sequence_length * sizeof(char), cudaMemcpyHostToDevice);
  
  // RVD Sequences
  
  cudaMalloc(&d_rvd_sequences, 2 * sizeof(int *));
  
  cudaMalloc(&(d_rvd_sequences[0]), rvd_sequence_lengths[0] * sizeof(int));
  cudaMemcpy(d_rvd_sequences[0], rvd_sequences[0], sizeof(int) * rvd_sequence_lengths[0], cudaMemcpyHostToDevice);
  
  cudaMalloc(&(d_rvd_sequences[1]), rvd_sequence_lengths[1] * sizeof(int));
  cudaMemcpy(d_rvd_sequences[1], rvd_sequences[1], sizeof(int) * rvd_sequence_lengths[1], cudaMemcpyHostToDevice);
  
  // RVD Sequence Lengths
  cudaMalloc(&d_rvd_sequence_lengths, 2 * sizeof(int));
  cudaMemcpy(d_rvd_sequence_lengths, rvd_sequence_lengths, sizeof(int) * 2, cudaMemcpyHostToDevice);
  
  // Results
  cudaMalloc(&d_results, 2 * sizeof(int *));
  cudaMalloc(&(d_results[0]), 2 * sizeof(int));
  cudaMalloc(&(d_results[1]), 2 * sizeof(int));
  
  d_results[0][0] = 0;
  d_results[0][1] = 0;
  d_results[1][0] = 0;
  d_results[0][1] = 0;
  
  // Cutoffs
  cudaMalloc(&d_cutoffs, 2 * sizeof(double));
  cudaMemcpy(d_cutoffs, cutoffs, sizeof(double) * 2, cudaMemcpyHostToDevice);
  
  // Scoring Matrix
  cudaMalloc(&d_scoring_matrix, scoring_matrix_length * sizeof(double *));
  
  for (int i = 0; i < scoring_matrix_length; i++) {
    cudaMalloc(&(d_scoring_matrix[i]), sizeof(double) * 4);
    cudaMemcpy(d_scoring_matrix[i], scoring_matrix, sizeof(double) * 4, cudaMemcpyHostToDevice);
  }
  
  // Run kernel
  dim3 threadsPerBlock(CHUNK_SIZE, spacer_sizes[1] - spacer_sizes[0] + 1);
  dim3 numBlocks((reference_sequence_length/threadsPerBlock.x) + 1, 1);
  
  MyKernel<<<numBlocks, threadsPerBlock>>>(d_reference_sequence, spacer_sizes[0], d_rvd_sequences, d_rvd_sequence_lengths, d_cutoffs, d_scoring_matrix, d_results);
  
  cudaMemcpy(results[0], d_results[0], 2 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(results[1], d_results[1], 2 * sizeof(int), cudaMemcpyDeviceToHost);
  
}


__global__ void MyKernel(char *input_sequence, int spacer_size_start, int **rvd_sequences, int *rvd_sequence_lengths, double *cutoffs, double **scoring_matrix, int **results) {
  
  __shared__ char shared_input_sequence[CHUNK_SIZE * 2];

  int offset = blockIdx.x * blockDim.x;

  shared_input_sequence[2 * threadIdx.x] = input_sequence[offset + 2 * threadIdx.x];
  shared_input_sequence[2 * threadIdx.x + 1] = input_sequence[offset + 2 * threadIdx.x + 1];
  
  __syncthreads();


  for (int f_idx = 0; f_idx < 2; f_idx++) {

    for (int r_idx = 0; r_idx < 2; r_idx++) {

      if (shared_input_sequence[threadIdx.x] == 'T' &&
          shared_input_sequence[rvd_sequence_lengths[f_idx] + spacer_size_start + threadIdx.y + rvd_sequence_lengths[r_idx]] == 'A') {

        double forward_score = 0;

        for (int i = 0; i < rvd_sequence_lengths[f_idx]; i++) {

          int base = shared_input_sequence[threadIdx.x + 1 + i];

          if (base == 'A' || base == 'a')
            forward_score += scoring_matrix[rvd_sequences[f_idx][i]][0];
          if (base == 'C' || base == 'c')
            forward_score += scoring_matrix[rvd_sequences[f_idx][i]][1];
          if (base == 'G' || base == 'g')
            forward_score += scoring_matrix[rvd_sequences[f_idx][i]][2];
          if (base == 'T' || base == 't')
            forward_score += scoring_matrix[rvd_sequences[f_idx][i]][3];

        }

        double reverse_score = 0;

        for (int i = 0; i < rvd_sequence_lengths[r_idx]; i++) {

          int base = shared_input_sequence[rvd_sequence_lengths[f_idx] + spacer_size_start + threadIdx.y + rvd_sequence_lengths[r_idx] - 1 - i];

          if (base == 'A' || base == 'a')
            reverse_score += scoring_matrix[rvd_sequences[r_idx][i]][3];
          if (base == 'C' || base == 'c')
            reverse_score += scoring_matrix[rvd_sequences[r_idx][i]][2];
          if (base == 'G' || base == 'g')
            reverse_score += scoring_matrix[rvd_sequences[r_idx][i]][1];
          if (base == 'T' || base == 't')
            reverse_score += scoring_matrix[rvd_sequences[r_idx][i]][0];

        }

        if (forward_score < cutoffs[f_idx] && forward_score < cutoffs[r_idx]) {
          results[f_idx][r_idx]++;
        }

      }

    }

  }
  
}
