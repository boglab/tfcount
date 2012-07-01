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

def TargetFinderCountTask(char *seq_filename, char *rvd_string, char *rvd_string2, int c_upstream, double cutoff, unsigned int spacer_min, unsigned int spacer_max):

        cdef Hashmap *tfcount_kwargs = hashmap_new(32)
        
        cdef double weight = 0.9
        
        cdef unsigned int *count_results_array_f = <unsigned int *> calloc(2, sizeof(unsigned int))
        cdef unsigned int *count_results_array_r = <unsigned int *> calloc(2, sizeof(unsigned int))
        cdef unsigned int **count_results_array = <unsigned int **> calloc(2, sizeof(unsigned int*))
        
        count_results_array[0] = count_results_array_f
        count_results_array[1] = count_results_array_r
        
        cdef int i
        cdef int j
        
        hashmap_add(tfcount_kwargs, "seq_filename", seq_filename)
        hashmap_add(tfcount_kwargs, "rvd_string", rvd_string)
        hashmap_add(tfcount_kwargs, "rvd_string2", rvd_string2)
        hashmap_add(tfcount_kwargs, "weight", &weight)
        hashmap_add(tfcount_kwargs, "cutoff", &cutoff)
        hashmap_add(tfcount_kwargs, "c_upstream", &c_upstream)
        hashmap_add(tfcount_kwargs, "spacer_min", &spacer_min)
        hashmap_add(tfcount_kwargs, "spacer_max", &spacer_max)
        
        hashmap_add(tfcount_kwargs, "count_results_array", count_results_array)
        
        # add a way to pass in an error string
        
        cdef int task_result = run_counting_task(tfcount_kwargs)
        
        hashmap_delete(tfcount_kwargs, NULL)
        
        count_results_list = []
        cdef int count_results_total = 0
        
        for i in range(2):
                for j in range(2):
                        count_results_total += count_results_array[i][j]
                        count_results_list.append(count_results_array[i][j])
        
        count_results_list.append(count_results_total)
        
        free(count_results_array_f)
        free(count_results_array_r)
        free(count_results_array)
        
        return count_results_list
