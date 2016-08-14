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
#include <bcutils/kseq_cuda.h>
KSEQ_INIT(gzFile, gzread)

#define MAX_THREADS_PER_BLOCK 1024
#define SCORE_THREADS_PER_BLOCK 512
#define TALLY_THREADS_PER_BLOCK 512
#define MAX_BLOCKS_PER_GRID 65535

#define MAX_COUNT_REF_LEN 320000000

#define PADDED_RVD_WIDTH 32

#define cudaSafeCall(call){   \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "%s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

__device__ float ScoringMatrixVal(const float * __restrict__ scoring_matrix, size_t pitch, unsigned int row, unsigned int column);
float *ScoringMatrixRow(float *scoring_matrix, size_t pitch, unsigned int row);

texture<float, cudaTextureType2D, cudaReadModeElementType> texRefSM;
texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> texRefRS;

__global__ void ScoreBindingSites(const char * __restrict__ input_sequence, unsigned long is_length, int rvd_offset, unsigned int rs_len, float cutoff, int c_upstream, unsigned int rvd_num, unsigned char * __restrict__ results) {

  int block_seq_index = SCORE_THREADS_PER_BLOCK * (blockIdx.y * gridDim.x + blockIdx.x);
  int thread_id = (blockDim.x * threadIdx.y) + threadIdx.x;
  int seq_index = block_seq_index + thread_id + 1;

  if (seq_index < 1 || seq_index >= is_length || seq_index + rs_len >= is_length - 1) return;

  char first = input_sequence[seq_index - 1];
  char last  = input_sequence[seq_index + rs_len];

  if (first == 4 and last == 4) return;

  float thread_result_t = 0.0f;
  float thread_result_a = 0.0f;

  for (int i = 0; i < rs_len; i++) {

    if (thread_result_t > cutoff && thread_result_a > cutoff) continue;

    int sm_col_t = input_sequence[seq_index + i];

    thread_result_t += tex2D(texRefSM, sm_col_t, tex1Dfetch(texRefRS, rvd_offset + i));

    int sm_col_a = (sm_col_t == 4 ? 4 : 3 - sm_col_t);

    thread_result_a += tex2D(texRefSM, sm_col_a, tex1Dfetch(texRefRS, rvd_offset + rs_len - i - 1));

  }

  int first_t = c_upstream != 1 && first == 3;
  int first_c = c_upstream != 0 && first == 1;
  int last_a  = c_upstream != 1 && last == 0;
  int last_g  = c_upstream != 0 && last == 2;

  results[seq_index] |= (thread_result_t < cutoff && (first_c || first_t)) << ((2 * rvd_num) + (first_c * 4));
  results[seq_index] |= (thread_result_a < cutoff && (last_g || last_a)) << ((2 * rvd_num + 1) + (last_g * 4));

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

__device__ float ScoringMatrixVal(const float * __restrict__ scoring_matrix, size_t pitch, unsigned int row, unsigned int column) {
  
  return *((float*)((char*) scoring_matrix + row * pitch) + column);
  
}

float *ScoringMatrixRow(float *scoring_matrix, size_t pitch, unsigned int row) {
  return (float*)((char*) scoring_matrix + row * pitch);
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

struct chars_to_smcols
{
  __host__ __device__ char operator()(const char &x) {
    if( x == 'A' || x == 'a')
      return 0;
    if( x == 'C' || x == 'c')
      return 1;
    if( x == 'G' || x == 'g')
      return 2;
    if( x == 'T' || x == 't')
      return 3;
    return 4;
  }
};

void RunCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *rvd_seqs, unsigned int *rvd_lengths, float *cutoffs, unsigned int num_rvd_seqs, int c_upstream, float **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results) {
  
  unsigned int *d_rvd_seqs;
  float *d_scoring_matrix;
  size_t sm_pitch;
  
  char *reference_buffer;
  size_t reference_buffer_len = MAX_COUNT_REF_LEN;
  kroundup32(reference_buffer_len);
  cudaSafeCall( cudaMallocHost(&reference_buffer, reference_buffer_len * sizeof(char)) );
  
  cudaSafeCall( cudaMalloc(&d_rvd_seqs, PADDED_RVD_WIDTH * num_rvd_seqs * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_seqs, rvd_seqs, PADDED_RVD_WIDTH * num_rvd_seqs * sizeof(unsigned int), cudaMemcpyHostToDevice) );
  
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 5 * sizeof(float), scoring_matrix_length * sizeof(float)) );
  
  for (int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(float) * 5, cudaMemcpyHostToDevice) );
  }
  
  cudaChannelFormatDesc channelDescSM = cudaCreateChannelDesc<float>();
  size_t offset;

  texRefSM.addressMode[0] = cudaAddressModeClamp;
  texRefSM.addressMode[1] = cudaAddressModeClamp;
  texRefSM.normalized = 0;
  texRefSM.filterMode = cudaFilterModePoint;

  cudaBindTexture2D(&offset, &texRefSM, d_scoring_matrix, &channelDescSM, 5, scoring_matrix_length, sm_pitch);
  
  dim3 score_threadsPerBlock(32, 16);
  
  gzFile seqfile = gzopen(seq_filename, "r");

  kseq_t *seq = kseq_init(seqfile);
  int result;
  
  unsigned char *d_prelim_results;
  char *d_reference_sequence;
  
  cudaSafeCall( cudaMalloc(&d_reference_sequence, MAX_COUNT_REF_LEN * sizeof(char)) );
  cudaSafeCall( cudaMalloc(&d_prelim_results, MAX_COUNT_REF_LEN * sizeof(unsigned char)) );
  
  cudaChannelFormatDesc channelDescRS = cudaCreateChannelDesc<unsigned int>();
  texRefRS.addressMode[0] = cudaAddressModeClamp;
  texRefRS.normalized = 0;
  texRefRS.filterMode = cudaFilterModePoint;

  cudaBindTexture(&offset, &texRefRS, d_rvd_seqs, &channelDescRS, PADDED_RVD_WIDTH * num_rvd_seqs * sizeof(unsigned int));
  
  result = kseq_read(seq, reference_buffer, reference_buffer_len);

  while (result >= 0) {

    char *reference_sequence = seq->seq.s;
    unsigned long reference_sequence_length = ((seq->seq.l + 31) / 32 ) * 32;

    cudaSafeCall( cudaMemset(d_reference_sequence, 4, MAX_COUNT_REF_LEN * sizeof(char)) );
    cudaSafeCall( cudaMemset(d_reference_sequence + MAX_COUNT_REF_LEN - 1, '\0', 1 * sizeof(char)) );
    cudaSafeCall( cudaMemcpy(d_reference_sequence, reference_sequence, reference_sequence_length * sizeof(char), cudaMemcpyHostToDevice) );
    
    thrust::device_ptr<char> reference_sequence_start(d_reference_sequence);
    thrust::device_ptr<char> reference_sequence_end(d_reference_sequence + reference_sequence_length);

    thrust::transform(reference_sequence_start, reference_sequence_end, reference_sequence_start, chars_to_smcols());

    logger(log_file, "Scanning %s for off-target sites (length %ld)", seq->name.s, seq->seq.l);
    
    thrust::device_ptr<unsigned char> prelim_results_start(d_prelim_results);
    thrust::device_ptr<unsigned char> prelim_results_end(d_prelim_results + reference_sequence_length);

    int score_blocks_needed = (reference_sequence_length + SCORE_THREADS_PER_BLOCK - 1) / SCORE_THREADS_PER_BLOCK;

    int score_block_x = (score_blocks_needed >= MAX_BLOCKS_PER_GRID ? MAX_BLOCKS_PER_GRID : score_blocks_needed);
    int score_block_y = (score_blocks_needed + (MAX_BLOCKS_PER_GRID - 1)) / MAX_BLOCKS_PER_GRID;

    dim3 score_blocksPerGrid(score_block_x, score_block_y);
    
    for (int i = 0; i < num_rvd_seqs; i++) {
      
      cudaSafeCall( cudaMemset(d_prelim_results, '\0', reference_sequence_length * sizeof(unsigned char)) );
      
      ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, i * PADDED_RVD_WIDTH, rvd_lengths[i], cutoffs[i], c_upstream, 0, d_prelim_results);
      if (i == 0) result = kseq_read(seq, reference_buffer, reference_buffer_len);

      cudaSafeCall( cudaGetLastError() );
      
      results[i] += thrust::count_if(prelim_results_start, prelim_results_end, first_t_or_c());
      results[i] += thrust::count_if(prelim_results_start, prelim_results_end, last_a_or_g());

      
    }
    
  }

  kseq_destroy(seq, reference_buffer);
  gzclose(seqfile);
  
  cudaSafeCall( cudaUnbindTexture(texRefSM) );
  cudaSafeCall( cudaUnbindTexture(texRefRS) );
  
  cudaSafeCall( cudaFreeHost(reference_buffer) );
  
  cudaSafeCall( cudaFree(d_rvd_seqs) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );
  
  cudaSafeCall( cudaFree(d_prelim_results) );
  
  cudaSafeCall( cudaFree(d_reference_sequence) );
  
}

void RunPairedCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *spacer_sizes, unsigned int *rvd_pairs, unsigned int *rvd_lengths, float *cutoffs, unsigned int num_rvd_pairs, int c_upstream, float **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results) {
  
  unsigned int *d_rvd_pairs;
  float *d_scoring_matrix;
  size_t sm_pitch;
  
  char *reference_buffer;
  size_t reference_buffer_len = MAX_COUNT_REF_LEN;
  kroundup32(reference_buffer_len);
  cudaSafeCall( cudaMallocHost(&reference_buffer, reference_buffer_len * sizeof(char)) );

  cudaSafeCall( cudaMalloc(&d_rvd_pairs, 2 * PADDED_RVD_WIDTH * num_rvd_pairs * sizeof(unsigned int)));
  cudaSafeCall( cudaMemcpy(d_rvd_pairs, rvd_pairs, 2 * PADDED_RVD_WIDTH * num_rvd_pairs * sizeof(unsigned int), cudaMemcpyHostToDevice) );
  
  cudaSafeCall( cudaMallocPitch(&d_scoring_matrix, &sm_pitch, 5 * sizeof(float), scoring_matrix_length * sizeof(float)) );
  
  for (int i = 0; i < scoring_matrix_length; i++) {
    cudaSafeCall( cudaMemcpy(ScoringMatrixRow(d_scoring_matrix, sm_pitch, i), scoring_matrix[i], sizeof(float) * 5, cudaMemcpyHostToDevice) );
  }
  
  cudaChannelFormatDesc channelDescSM = cudaCreateChannelDesc<float>();
  size_t offset;

  texRefSM.addressMode[0] = cudaAddressModeClamp;
  texRefSM.addressMode[1] = cudaAddressModeClamp;
  texRefSM.normalized = 0;
  texRefSM.filterMode = cudaFilterModePoint;

  cudaBindTexture2D(&offset, &texRefSM, d_scoring_matrix, &channelDescSM, 5, scoring_matrix_length, sm_pitch);
  
  dim3 score_threadsPerBlock(32, 16);
  dim3 tally_threadsPerBlock(32, 16);
  
  gzFile seqfile = gzopen(seq_filename, "r");

  kseq_t *seq = kseq_init(seqfile);
  int result;
  
  unsigned char *d_prelim_results;
  unsigned int *d_second_results;
  char *d_reference_sequence;

  cudaSafeCall( cudaMalloc(&d_reference_sequence, MAX_COUNT_REF_LEN * sizeof(char)) );
  cudaSafeCall( cudaMalloc(&d_prelim_results, MAX_COUNT_REF_LEN * sizeof(unsigned char)) );
  cudaSafeCall( cudaMalloc(&d_second_results, MAX_COUNT_REF_LEN * sizeof(unsigned int)) );

  cudaChannelFormatDesc channelDescRS = cudaCreateChannelDesc<unsigned int>();
  texRefRS.addressMode[0] = cudaAddressModeClamp;
  texRefRS.normalized = 0;
  texRefRS.filterMode = cudaFilterModePoint;

  cudaBindTexture(&offset, &texRefRS, d_rvd_pairs, &channelDescRS, 2 * PADDED_RVD_WIDTH * num_rvd_pairs * sizeof(unsigned int));

  result = kseq_read(seq, reference_buffer, reference_buffer_len);

  while (result >= 0) {

    char *reference_sequence = seq->seq.s;
    int reference_sequence_length = ((seq->seq.l + 31) / 32 ) * 32;

    cudaSafeCall( cudaMemset(d_reference_sequence, 4, MAX_COUNT_REF_LEN * sizeof(char)) );
    cudaSafeCall( cudaMemset(d_reference_sequence + MAX_COUNT_REF_LEN - 1, '\0', 1 * sizeof(char)) );
    cudaSafeCall( cudaMemcpy(d_reference_sequence, reference_sequence, reference_sequence_length * sizeof(char), cudaMemcpyHostToDevice) );
    
    thrust::device_ptr<char> reference_sequence_start(d_reference_sequence);
    thrust::device_ptr<char> reference_sequence_end(d_reference_sequence + reference_sequence_length);

    thrust::transform(reference_sequence_start, reference_sequence_end, reference_sequence_start, chars_to_smcols());
    
    logger(log_file, "Scanning %s for off-target sites (length %ld)", seq->name.s, seq->seq.l);
    
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

      ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, first_index * PADDED_RVD_WIDTH, rvd_lengths[first_index], cutoffs[first_index], c_upstream, 0, d_prelim_results);
      if (i == 0) result = kseq_read(seq, reference_buffer, reference_buffer_len);
      cudaSafeCall( cudaGetLastError() );

      ScoreBindingSites <<<score_blocksPerGrid, score_threadsPerBlock>>>(d_reference_sequence, reference_sequence_length, second_index * PADDED_RVD_WIDTH, rvd_lengths[second_index], cutoffs[second_index], c_upstream, 1, d_prelim_results);
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
    
  }

  kseq_destroy(seq, reference_buffer);
  gzclose(seqfile);
  
  cudaSafeCall( cudaUnbindTexture(texRefSM) );
  cudaSafeCall( cudaUnbindTexture(texRefRS) );
  
  cudaSafeCall( cudaFreeHost(reference_buffer) );

  cudaSafeCall( cudaFree(d_rvd_pairs) );
  cudaSafeCall( cudaFree(d_scoring_matrix) );

  cudaSafeCall( cudaFree(d_prelim_results) );
  cudaSafeCall( cudaFree(d_second_results) );

  cudaSafeCall( cudaFree(d_reference_sequence) );

  cudaSafeCall( cudaDeviceReset() );
  
}
