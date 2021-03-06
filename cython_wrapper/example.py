from btfcount import TargetFinderCountTask

#organism = "caenorhabditis_elegans"
organism = "arabidopsis_thaliana"
#organism = "drosophila_melanogaster"
#organism = "oryza_sativa"
#organism = "danio_rerio"
#organism = "mus_musculus"
#organism = "homo_sapiens"

example_rvd_seqs = [
    "HD HD NN NG HD NI NG HD NN NN HD NG NG HD NN HD NG",
    "HD NI HD NI HD NI HD NI HD HD HD NN NG HD HD NN",
    "NN HD HD HD NI NG NN HD NG NN NI NN NI NN NN",
    "NN HD NG NN NI NN NI NN NN HD NI NN NI HD HD",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD NN",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD NN NN NG",
    "NN HD NG NN NI NN NI NN NN HD NI NN NI HD HD",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD NN",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD NN NN NG",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD NN NN NG HD NI",
    "NN NI NN NI NN NN HD NI NN NI HD HD NN NN HD NN NN NG HD NI",
    "HD NI HD NI HD NI HD NI HD NI HD NN HD NI HD",
    "HD NI HD NI HD NI HD NI HD NI HD NN HD NI HD NI",
    "HD NI HD NI HD NI HD NI HD NI HD NN HD NI HD NI HD NN",
    "HD NI HD NI HD NI HD NI HD NI HD NN HD NI HD NI HD NN HD NI",
    "NN NG HD HD NG HD HD NN NG HD NI NN HD NN NN",
    "HD HD NG HD HD NN NG HD NI NN HD NN NN HD NG",
    "NN NG HD HD NG HD HD NN NG HD NI NN HD NN NN",
    "HD HD HD NI HD NI HD NI HD NI HD NI HD NG NN NN NG", 
    "HD HD HD NI NN NI HD HD NN NN NI NI HD NN NG NN NN HD", 
    "HD HD NG HD HD NI HD HD HD NI NI HD NN NN NG NN NG HD NG", 
    "HD HD NG HD HD NI NG NN NG NG NN NI NI HD NI NG", 
    "HD HD NG HD NI NG HD HD NI NN NG NN NI HD NG HD NG", 
    "HD HD NG HD NI NN NN NG NG NN NG HD NI NI NG NG", 
    "HD HD NG NN NN HD NN NN HD HD NN NN NG HD NG", 
    "HD HD NI HD HD NI NI NG HD HD NG HD NI HD HD", 
    "HD HD NI NG NG NG NG NN NI HD NG NG NG NG NG", 
    "HD HD NI NG NN NG NN NN HD NG NG HD NG HD NG NG NN HD NG", 
    "HD HD NI NG NN NI NI HD NI NG NN HD NI NI HD HD NG NN NI NI", 
    "HD HD NN NG HD NI NG HD NN NN HD NG NG HD NN HD NG", 
    "HD HD NN NI HD HD NI HD HD NN HD HD HD NG NN NG NN NG", 
    "HD NG HD NG NG NG NG NG NG NG NG NG NG NI NI NG HD HD HD", 
    "HD NG HD NG NN HD NG NN HD HD HD NG NN NG HD NN HD NI NN NG", 
    "HD NG NG HD NI NI HD NG NI NG NN NN NN HD NN HD NI NG", 
    "HD NG NG NG HD NG NG NN NI NG HD NG HD NG NN", 
    "HD NG NG NI NG HD NG NN NG HD HD NI NI NI HD NI NI NG", 
    "HD NG NI NG HD NG NG NI HD NI NN NI NI NN NI NI NN HD NG NG NG", 
    "HD NG NN NG NI HD HD NI NI HD HD NN NG NG NG HD HD NG", 
    "HD NG NN NI NG NG NN NN NG NG NG NI NN NG NG HD NI", 
    "HD NG NN NN NI NI HD NG NG NG NG NI NI HD NI HD HD NI HD NG", 
    "HD NI HD NI HD NI HD NI HD HD HD NN NG HD HD NN", 
    "HD NI NG HD NN NI NN NI NI HD HD HD NI NG NG", 
    "HD NI NG NN HD NG NN NG NN NG NN NI HD NG NN", 
    "HD NI NG NN NN NI HD NI HD HD NG NN NG NN NG HD NN HD HD NI", 
    "HD NI NG NN NN NN HD NI HD HD NG NG HD NI NI",
    "HD NN NG NN NI NI HD NI NG NN HD HD NI NI NN NN NG HD NG NG", 
    "HD NN NN NI NN NN HD NN NN NI NI HD HD NI NI NI NG NG HD",
]
print(TargetFinderCountTask("/opt/boglab/genome_data/" + organism + ".fasta", "NA", 0, 3.0, example_rvd_seqs))
