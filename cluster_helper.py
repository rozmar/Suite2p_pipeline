import sys
from utils import utils_imaging, utils_io
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
if type(arguments)== list and len(arguments)>1:
    arguments = ','.join(arguments)
else:
    arguments = arguments[0]
print(command)
print(arguments)
eval('{}({})'.format(command,arguments))
