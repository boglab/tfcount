#!/usr/bin/python2
from btfcount import TargetFinderCountTask

off_target_pairs = [
    "HD HD HD NI HD NI HD NI HD NI HD NI HD NG NN NN NG",
    "NI HD NG NN HD NI NN HD NG HD HD NG HD NG NN HD NG",
    "HD HD HD NI NN NI HD HD NN NN NI NI HD NN NG NN NN HD",
    "HD NG NN NN HD HD HD NG NN HD NI NG NN NG NG HD HD NI",
    "HD HD NG HD HD NI HD HD HD NI NI HD NN NN NG NN NG HD NG",
    "HD HD NI NN NN NI HD NG NN NG NG NG NN HD NG NG",
    "HD HD NG HD HD NI NG NN NG NG NN NI NI HD NI NG",
    "HD NN NG NG HD NI NI NI NI HD NN HD NG NN NI NG",
    "HD HD NG HD NI NG HD HD NI NN NG NN NI HD NG HD NG",
    "NI NN HD NI HD NI HD NN HD HD HD NI NG NG NG",
    "HD HD NG HD NI NN NN NG NG NN NG HD NI NI NG NG",
    "HD HD NI NG NN NG HD HD NI NG NI NG HD NI NG",
    "HD HD NG NN NN HD NN NN HD HD NN NN NG HD NG",
    "NN NN HD HD NI NN NN HD HD NI NN NI NI NG NG",
    "HD HD NI HD HD NI NI NG HD HD NG HD NI HD HD",
    "NI NG NG NG HD HD HD NI NN HD NI NN NN NN HD NN",
    "HD HD NI HD HD NI NI NG HD HD NG HD NI HD HD",
    "NN NN NI NI NG NI NG NG NG HD HD HD NI NN HD NI NN",
    "HD HD NI NG NG NG NG NN NI HD NG NG NG NG NG",
    "HD HD NI NI HD NI NN NG NG NG HD NG NN HD NI NI NI HD",
    "HD HD NI NG NN NG NN NN HD NG NG HD NG HD NG NG NN HD NG",
    "HD NG HD HD NG NG HD NN NG NG NG NN NG NG NG HD HD NI NG",
    "HD HD NI NG NN NI NI HD NI NG NN HD NI NI HD HD NG NN NI NI",
    "NG NN NI NN NG NI NI NG NI NG HD NI NN NG NG NG HD NI HD",
    "HD HD NN NG HD NI NG HD NN NN HD NG NG HD NN HD NG",
    "NI NG NN NN NN NI NI HD HD NG HD HD NI NN NI",
    "HD HD NN NI HD HD NI HD HD NN HD HD HD NG NN NG NN NG",
    "HD NN HD HD HD HD NN NN HD HD NG HD NI HD NG NG NG HD",
    "HD NG HD NG NG NG NG NG NG NG NG NG NG NI NI NG HD HD HD",
    "NI HD NG NG HD HD NI NG HD NG NN NN HD NG NN HD NI NN HD",
    "HD NG HD NG NN HD NG NN HD HD HD NG NN NG HD NN HD NI NN NG",
    "NN NG HD HD NI NN HD NI NI HD NG NN NG HD HD NG",
    "HD NG NG HD NI NI HD NG NI NG NN NN NN HD NN HD NI NG",
    "HD NN HD NG HD NG NN HD NI NI NI NG NN NN HD NG",
    "HD NG NG NG HD NG NG NN NI NG HD NG HD NG NN",
    "NI NG NG NN NI NN NN NI HD HD NG NG NI NG NN",
    "HD NG NG NI NG HD NG NN NG HD HD NI NI NI HD NI NI NG",
    "NN NG HD NI NG HD NG NN NG NI NN NG NG HD NN HD NG",
    "HD NG NI NG HD NG NG NI HD NI NN NI NI NN NI NI NN HD NG NG NG",
    "NI NN NI NG NN NG NG HD NG HD NG NI NG NN NI HD NN NG",
    "HD NG NN NG NI HD HD NI NI HD HD NN NG NG NG HD HD NG",
    "HD HD HD NN NI NI HD NI NN NG HD NN NG HD HD NN NG",
    "HD NG NN NI NG NG NN NN NG NG NG NI NN NG NG HD NI",
    "NG HD NN HD NG NN HD NI NN HD HD NI NI HD NN NN",
    "HD NG NN NN NI NI HD NG NG NG NG NI NI HD NI HD HD NI HD NG",
    "NI NG HD NI NN NG HD HD NI NG NG NN HD NG NG",
    "HD NI HD NI HD NI HD NI HD HD HD NN NG HD HD NN",
    "HD NN NI NG HD NG NG HD NG NG NN HD HD HD HD NI NN NN",
    "HD NI NG HD NN NI NN NI NI HD HD HD NI NG NG",
    "NN NN HD HD HD HD NI NG HD HD NI NN HD HD NN NN NI",
    "HD NI NG NN HD NG NN NG NN NG NN NI HD NG NN",
    "NI HD NN HD NN HD NI HD NN NG NN NG NN HD NN",
    "HD NI NG NN NN NI HD NI HD HD NG NN NG NN NG HD NN HD HD NI",
    "NN HD NI NG HD NG NG NG HD NN NN NN NI HD HD NG NN NN NN HD",
    "HD NI NG NN NN NN HD NI HD HD NG NG HD NI NI",
    "NN NN HD HD HD NI NG HD NG HD NI HD HD NI NN",
    "HD NN NG NN NI NI HD NI NG NN HD HD NI NI NN NN NG HD NG NG",
    "NN NN HD NG NN NI HD HD NG NN HD NG HD HD NI NG",
    "HD NN NN NI NN NN HD NN NN NI NI HD HD NI NI NI NG NG HD",
    "HD NN NG NI NN NN NG NG NG NG HD NG NN NI NI NI NG HD NG"
]

print(TargetFinderCountTask("/opt/boglab/genome_data/arabidopsis_thaliana.fasta", "NA", 0, 3.0, off_target_pairs))
#print(TargetFinderCountTask("/opt/boglab/genome_data/danio_rerio.fasta", "NA", 0, 3.0, off_target_pairs))
