from libc.stdlib cimport malloc, free, calloc

ctypedef void (*valuefreefunc)(void *)

cdef extern from "Hashmap.h":

        ctypedef struct Hashmap:
                pass

        Hashmap *hashmap_new(int size)
        int hashmap_add(Hashmap *hash, char *key, void *value)
        void hashmap_delete(Hashmap *hash, valuefreefunc)

cdef extern from "tfcount.h":
        int run_counting_task(Hashmap *kwargs)

def TargetFinderCountTask(char *seq_filename, int c_upstream, double cutoff, unsigned int spacer_min, unsigned int spacer_max, rvd_pairs):

        cdef int i
        cdef int j
        
        cdef Hashmap *tfcount_kwargs = hashmap_new(32)
        
        cdef double weight = 0.9
        
        cdef int num_pairs = len(rvd_pairs)
        
        cdef char **rvd_strings = <char **> calloc(2 * num_pairs, sizeof(char *))
        
        for i in range(num_pairs):
                rvd_strings[2 * i] = rvd_pairs[i][0]
                rvd_strings[2 * i + 1] = rvd_pairs[i][1]
        
        cdef unsigned int *count_results_array = <unsigned int *> calloc(4 * num_pairs, sizeof(unsigned int))
        
        hashmap_add(tfcount_kwargs, "seq_filename", seq_filename)
        hashmap_add(tfcount_kwargs, "rvd_strings", rvd_strings)
        hashmap_add(tfcount_kwargs, "weight", &weight)
        hashmap_add(tfcount_kwargs, "cutoff", &cutoff)
        hashmap_add(tfcount_kwargs, "c_upstream", &c_upstream)
        hashmap_add(tfcount_kwargs, "spacer_min", &spacer_min)
        hashmap_add(tfcount_kwargs, "spacer_max", &spacer_max)
        
        hashmap_add(tfcount_kwargs, "num_rvd_pairs", &num_pairs)
        hashmap_add(tfcount_kwargs, "count_results_array", count_results_array)
        
        # add a way to pass in an error string
        
        cdef int task_result = run_counting_task(tfcount_kwargs)
        
        hashmap_delete(tfcount_kwargs, NULL)
        
        count_results_list = []
        
        cdef unsigned int *pair_results_array
        cdef int pair_results_total
        
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
        free(rvd_strings)
        
        return count_results_list
