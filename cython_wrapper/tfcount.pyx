from libc.stdlib cimport malloc, free, calloc
from libc.stdio cimport FILE, stdout, fopen, fclose
from cython.operator cimport dereference as deref

ctypedef void (*valuefreefunc)(void *)

cdef extern from "Hashmap.h":

        ctypedef struct Hashmap:
                pass

        Hashmap *hashmap_new(int size)
        int hashmap_add(Hashmap *hash, char *key, void *value)
        void hashmap_delete(Hashmap *hash, valuefreefunc)
        int hashmap_size(Hashmap *hash)
        void *hashmap_get(Hashmap *hash, char *key)
        char **hashmap_keys(Hashmap *hash)

cdef extern from "Array.h":
        
        ctypedef struct Array:
                pass
        
        Array *array_new()
        void *array_get(Array *r, int index)
        void array_delete(Array *r, valuefreefunc)

cdef extern from "bcutils.h":
        double *double_array(double a, double c, double g, double t, double dummy)
        Hashmap *get_diresidue_probabilities(Array *rvdseq, double w)
        Hashmap *convert_probabilities_to_scores(Hashmap *diresidue_probabilities)

cdef extern from "tfcount_cuda.h":
        void RunCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *rvd_seqs, unsigned int *rvd_lengths, float *cutoffs, unsigned int num_rvd_seqs, int c_upstream, float **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results)
        void RunPairedCountBindingSites(char *seq_filename, FILE *log_file, unsigned int *spacer_sizes, unsigned int *rvd_pairs, unsigned int *rvd_lengths, float *cutoffs, unsigned int num_rvd_pairs, int c_upstream, float **scoring_matrix, unsigned int scoring_matrix_length, unsigned int *results)

cdef get_best_score(rvd_seq, Hashmap *rvdscores):
        
        cdef:
                int i,j
                double best_score = 0.0
                double min_score = -1.0
                double *scores
        
        for i in range(len(rvd_seq)):
                scores = <double*> hashmap_get(rvdscores, rvd_seq[i])
                if scores == NULL:
                        return -1.0
                for j in range(4):
                        if j == 0 or scores[j] < min_score:
                                min_score = scores[j]
                best_score += min_score
        
        return best_score

def TargetFinderCountTask(char *seq_filename, char *log_filepath, int c_upstream, double cutoff, rvd_seqs_list):
        
        cdef:
                int i, j, first_pos, second_pos
                double weight = 0.9
                FILE *log_file = stdout
                int BIGGEST_RVD_SCORE_EVER = 100
                int PADDED_RVD_WIDTH = 32
                
                unsigned int num_seqs = len(rvd_seqs_list)
                unsigned int *rvd_seqs = <unsigned int *> calloc(PADDED_RVD_WIDTH * num_seqs, sizeof(unsigned int))
                unsigned int *rvd_lens = <unsigned int *> calloc(num_seqs, sizeof(unsigned int))
                float *rvd_cutoffs = <float*> calloc(num_seqs, sizeof(float))
                
                unsigned int *count_results_array = <unsigned int *> calloc(num_seqs, sizeof(unsigned int))
                
                Array *empty_array = array_new()
                Hashmap *diresidue_probabilities = get_diresidue_probabilities(empty_array, weight)
                Hashmap *diresidue_scores = convert_probabilities_to_scores(diresidue_probabilities)
                double *rvd_score_ptr
                
        hashmap_delete(diresidue_probabilities, NULL)
        hashmap_add(diresidue_scores, "XX", double_array(0, 0, 0, 0, BIGGEST_RVD_SCORE_EVER))
        
        cdef:
                float *scoring_matrix_bak = <float*> calloc(hashmap_size(diresidue_scores) * 5, sizeof(float))
                float **scoring_matrix = <float**> calloc(hashmap_size(diresidue_scores), sizeof(float*))

        cdef char **diresidues = hashmap_keys(diresidue_scores)
        
        rvd_to_int = {}
        
        for i in range(hashmap_size(diresidue_scores)):
                rvd_to_int[diresidues[i]] = i
                rvd_score_ptr = <double*> hashmap_get(diresidue_scores, diresidues[i])
                scoring_matrix[i] = scoring_matrix_bak+5*i
                scoring_matrix[i][0] = <float> rvd_score_ptr[0]
                scoring_matrix[i][1] = <float> rvd_score_ptr[1]
                scoring_matrix[i][2] = <float> rvd_score_ptr[2]
                scoring_matrix[i][3] = <float> rvd_score_ptr[3]
                scoring_matrix[i][4] = BIGGEST_RVD_SCORE_EVER
        
        cdef unsigned int blank_rvd = rvd_to_int["XX"]
        
        for i in range(num_seqs):
                
                rvd_seq = rvd_seqs_list[i].split(" ")
                
                rvd_lens[i]= len(rvd_seq)
                
                rvd_cutoffs[i] = cutoff * get_best_score(rvd_seq, diresidue_scores)
                
                for j in range(PADDED_RVD_WIDTH):
                        if j < len(rvd_seq):
                                rvd_seqs[i * PADDED_RVD_WIDTH + j] = rvd_to_int[rvd_seq[j]]
                        else:
                                rvd_seqs[i * PADDED_RVD_WIDTH + j] = blank_rvd
        
        if log_filepath and log_filepath != b"NA":
                log_file = fopen(log_filepath, "a")
        
        RunCountBindingSites(seq_filename, log_file, rvd_seqs, rvd_lens, rvd_cutoffs, num_seqs, c_upstream, scoring_matrix, hashmap_size(diresidue_scores), count_results_array)
        
        
        if log_file != stdout:
                fclose(log_file)
        
        free(scoring_matrix)
        free(scoring_matrix_bak)
        free(diresidues)
        free(rvd_seqs)
        free(rvd_lens)
        free(rvd_cutoffs)
        array_delete(empty_array, NULL)
        hashmap_delete(diresidue_scores, free)
        
        count_results_list = []
        
        cdef unsigned int *pair_results_array
        cdef unsigned int pair_results_total
        
        for i in range(num_seqs):
                count_results_list.append(count_results_array[i])
        
        free(count_results_array)
        
        return count_results_list

def PairedTargetFinderCountTask(char *seq_filename, char *log_filepath, int c_upstream, double cutoff, unsigned int spacer_min, unsigned int spacer_max, rvd_pairs_list):
        
        cdef:
                int i, j, first_pos, second_pos
                double weight = 0.9
                FILE *log_file = stdout
                int BIGGEST_RVD_SCORE_EVER = 100
                int PADDED_RVD_WIDTH = 32
                
                unsigned int num_pairs = len(rvd_pairs_list)
                unsigned int *rvd_pairs = <unsigned int *> calloc(2 * PADDED_RVD_WIDTH * num_pairs, sizeof(unsigned int))
                unsigned int *rvd_lens = <unsigned int *> calloc(2 * num_pairs, sizeof(unsigned int))
                float *rvd_cutoffs = <float*> calloc(2 * num_pairs, sizeof(float))
                
                unsigned int *spacer_sizes = [spacer_min, spacer_max]
                unsigned int *count_results_array = <unsigned int *> calloc(4 * num_pairs, sizeof(unsigned int))
                
                Array *empty_array = array_new()
                Hashmap *diresidue_probabilities = get_diresidue_probabilities(empty_array, weight)
                Hashmap *diresidue_scores = convert_probabilities_to_scores(diresidue_probabilities)
                double *rvd_score_ptr
                
        hashmap_delete(diresidue_probabilities, NULL)
        hashmap_add(diresidue_scores, "XX", double_array(0, 0, 0, 0, BIGGEST_RVD_SCORE_EVER))
        
        cdef:
                float *scoring_matrix_bak = <float*> calloc(hashmap_size(diresidue_scores) * 5, sizeof(float))
                float **scoring_matrix = <float**> calloc(hashmap_size(diresidue_scores), sizeof(float*))
        
        cdef char **diresidues = hashmap_keys(diresidue_scores)
        
        rvd_to_int = {}
        
        for i in range(hashmap_size(diresidue_scores)):
                rvd_to_int[diresidues[i]] = i
                rvd_score_ptr = <double*> hashmap_get(diresidue_scores, diresidues[i])
                scoring_matrix[i] = scoring_matrix_bak+5*i
                scoring_matrix[i][0] = <float> rvd_score_ptr[0]
                scoring_matrix[i][1] = <float> rvd_score_ptr[1]
                scoring_matrix[i][2] = <float> rvd_score_ptr[2]
                scoring_matrix[i][3] = <float> rvd_score_ptr[3]
                scoring_matrix[i][4] = BIGGEST_RVD_SCORE_EVER
        
        cdef unsigned int blank_rvd = rvd_to_int["XX"]
        
        for i in range(num_pairs):
                
                first_pos = 2 * i
                second_pos = first_pos + 1
                
                rvd_seq = rvd_pairs_list[i][0].split(" ")
                rvd_seq_2 = rvd_pairs_list[i][1].split(" ")
                
                rvd_lens[first_pos]= len(rvd_seq)
                rvd_lens[second_pos]= len(rvd_seq_2)
                
                rvd_cutoffs[first_pos] = cutoff * get_best_score(rvd_seq, diresidue_scores)
                rvd_cutoffs[second_pos] = cutoff * get_best_score(rvd_seq_2, diresidue_scores)
                
                for j in range(PADDED_RVD_WIDTH):
                        if j < len(rvd_seq):
                                rvd_pairs[first_pos * PADDED_RVD_WIDTH + j] = rvd_to_int[rvd_seq[j]]
                        else:
                                rvd_pairs[first_pos * PADDED_RVD_WIDTH + j] = blank_rvd
                
                for j in range(PADDED_RVD_WIDTH):
                        if j < len(rvd_seq_2):
                                rvd_pairs[second_pos * PADDED_RVD_WIDTH + j] = rvd_to_int[rvd_seq_2[j]]
                        else:
                                rvd_pairs[second_pos * PADDED_RVD_WIDTH + j] = blank_rvd
        
        if log_filepath and log_filepath != b"NA":
                log_file = fopen(log_filepath, "a")
        
        RunPairedCountBindingSites(seq_filename, log_file, spacer_sizes, rvd_pairs, rvd_lens, rvd_cutoffs, num_pairs, c_upstream, scoring_matrix, hashmap_size(diresidue_scores), count_results_array)
        
        if log_file != stdout:
                fclose(log_file)
        
        free(scoring_matrix)
        free(scoring_matrix_bak)
        free(diresidues)
        free(rvd_pairs)
        free(rvd_lens)
        free(rvd_cutoffs)
        array_delete(empty_array, NULL)
        hashmap_delete(diresidue_scores, free)
        
        count_results_list = []
        
        cdef unsigned int *pair_results_array
        cdef unsigned int pair_results_total
        
        for i in range(num_pairs):
                
                pair_results_array = count_results_array + (4 * i)
                pair_results_total = 0
                
                pair_results_list = []
                
                for j in range(4):
                        
                        pair_results_total += pair_results_array[j]
                        pair_results_list.append(pair_results_array[j])
                
                pair_results_list.append(pair_results_total)
                
                count_results_list.append(pair_results_list)
        
        free(count_results_array)
        
        return count_results_list

