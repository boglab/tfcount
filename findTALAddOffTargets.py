#!/usr/bin/python

from datetime import datetime
from optparse import OptionParser
from btfcount import TargetFinderCountTask

class TaskError(ValueError):
	pass

def create_logger(logFilepath):
    
    if logFilepath != 'NA':
        def logger(message):
            with open(logFilepath, "a") as logFile:
                logFile.write("[%s] %s\n" % (datetime.now().ctime(), message))
        
    else:
        def logger(message):
            print "[%s] %s" % (datetime.now().ctime(), message)
    
    return logger

def validateOptions(options):
    
    if options.cupstream not in [0, 1, 2]:
        raise TaskError("Invalid cupstream value provided")
    
    if options.min < 10 or options.min > 35:
        raise TaskError("Minimum spacer length must be between 10 and 35")
    
    if options.max < 10 or options.max > 35:
        raise TaskError("Maximum spacer length must be between 10 and 35")
    
    if options.max < options.min:
        raise TaskError("Maximum spacer length cannot be less than the minimum spacer length")
    
    if options.cutoff not in [3, 3.5, 4]:
        raise TaskError("Invalid cutoff value provided")

def FindTALAddOffTargets(options):
    
    logger = create_logger(options.logFilepath)
    
    logger("Beginning")
    
    results = []
    
    with open(options.outpath, "r") as results_file:
        
        results_line = results_file.readline()
        
        while results_line:
            results.append(results_line.rstrip())
            results_line = results_file.readline()
    
    header_row_idx = 0
    
    while results[header_row_idx][:13] != 'Sequence Name':
        header_row_idx = header_row_idx + 1
    
    if results[header_row_idx][-19] != "\tOff-Target Counts":
        results[header_row_idx] = results[header_row_idx] + "\tOff-Target Counts"
    
    header_count = len(results[header_row_idx].split('\t'))
    
    result_start_idx = header_row_idx + 1
    
    talen_pairs = []
    
    for i in range(header_row_idx + 1, len(results)):
        
        row_fields = results[i].split('\t')
        
        talen_pairs.append([row_fields[8], row_fields[9]])
    
    off_target_counts = TargetFinderCountTask(options.offtargetseq, options.logFilepath, options.cupstream, options.cutoff, options.min, options.max, talen_pairs)

    j = 0
    
    for i in range(header_row_idx + 1, len(results)):
        
        row_fields = results[i].split('\t')
        
        count_string = ' '.join(str(off_target_counts[j][x]) for x in range(5))
        
        if len(row_fields) == header_count:
            row_fields[-1] = count_string
        else:
            row_fields.append(count_string)
        
        results[i] = '\t'.join(row_fields)
        
        j = j + 1
    
    with open(options.outpath, "w") as out:
        
        for results_line in results:
            out.write(results_line + '\n')
    
    logger('Finished')

if __name__ == '__main__':
    
    # import arguments and options
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)
    # Offtarget Options
    parser.add_option('-m', '--min', dest='min', type='int', default=15, help='the minimum spacer size to try; default is 15')
    parser.add_option('-x', '--max', dest='max', type='int', default=30, help='the maximum spacer size to try; default is 30')
    parser.add_option('-u', '--cupstream', dest='cupstream', type='int', default = 0, help='0 to look for T upstream, 1 to look for C, 2 to look for either; default is 0')
    parser.add_option('-t', '--cutoff', dest='cutoff', type='float', default = 3.0, help='The threshold score that off-targets must meet; default is 3.0')
    parser.add_option('-l', '--logpath', dest='logFilepath', type='string', default = 'NA', help='Optional file path to log progress to; default is stdout')
    parser.add_option('-s', '--offtargetseq', dest='offtargetseq', type = 'string', default='NA', help='Path to FASTA file to count off-targets in')
    parser.add_option('-p', '--outpath', dest='outpath', type='string', default = 'NA', help='TALEN Targeter result file path')

    (options, args) = parser.parse_args()
    
    validateOptions(options)
    
    FindTALAddOffTargets(options)
	