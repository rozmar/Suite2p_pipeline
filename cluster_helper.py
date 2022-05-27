import sys
from utils import utils_imaging, utils_io
import numpy as np
print(sys.argv)
arguments = sys.argv[2:]
command = sys.argv[1]
if command in ['utils_imaging.register_trial',
               'utils_imaging.find_ROIs',
               'utils_imaging.registration_metrics',
               'utils_io.concatenate_suite2p_files',
               'utils_io.copy_tiff_files_in_order']: # all is string
    arguments_real = list()
    for argument in arguments:
        arguments_real.append('"'+argument+'"')
    arguments=arguments_real

if '[' in arguments_real[-1]:
    last_arguments = arguments_real[-1].strip('[]').split(',')
    for last_argument in last_arguments:
        arguments = ','.join(np.asarray(arguments)[:-1]) + ',' + last_argument
        print(command)
        print(arguments)
        eval('{}({})'.format(command,arguments))
        
else:
    if type(arguments)== list and len(arguments)>1:
        arguments = ','.join(arguments)
    else:
        arguments = arguments[0]
    print(command)
    print(arguments)
    eval('{}({})'.format(command,arguments))
