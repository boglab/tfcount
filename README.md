# License

All source code is available under an ISC license.

Copyright (c) 2012-2015, Nick Booher <njbooher@gmail.com>.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# Description

tfcount is a library for quickly counting the number of predicted binding sites in a FASTA file for TALEN pairs generated by TALEN Targeter.

# Dependencies

To use tfcount you will need an NVIDIA GPU of compute capability 3.0+ ([see this list of NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus)) and CUDA Toolkit 7.5+, [libbcutils](https://github.com/boglab/cutils), Python 2.6+, and [cython](http://pypi.python.org/pypi/Cython).

# Compilation Directions

The makefile assumes CUDA is installed in `/opt/cuda`. You will need to change the paths in the makefile if this is not the case.

Run from the directory containing this file:
```
make
make install
cd cython_wrapper
python setup.py build_ext
python setup.py install
```

# Usage Directions

You can run the program with default options like this:
```
python findTALAddOffTargets.py --offtargetseq PATH_TO_FASTA_FILE --outpath PATH_TO_TALEN_TARGETER_RESULT_FILE
```
For information on available options:
```
python findTALAddOffTargets.py --help
```
