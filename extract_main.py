

import os
import numpy as np
from suite2p.extraction.extract import extract_traces_from_masks
import sys
try:
    local_temp_dir = sys.argv[1]#'/mnt/HDDS/Fast_disk_0/temp/'
    metadata_dir = sys.argv[2]#'/mnt/Data/BCI_metadata/'
    raw_scanimage_dir_base = sys.argv[3]#'/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
    suite2p_dir_base = sys.argv[4]#'/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
    try:
        subject = sys.argv[5]#'BCI_29'
    except:
        subject = None
    try:
        setup = sys.argv[6]#'Bergamo-2P-Photostim'
    except:
        setup = None
    try:
        fov =sys.argv[7]#'Bergamo-2P-Photostim'
    except:
        fov = None

except:
    local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/'
    metadata_dir = '/mnt/Data/BCI_metadata/'
    raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
    suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
    subject = 'BCI_29'
    setup = 'Bergamo-2P-Photostim'
    fov = 'FOV_03'
    
FOV_dir = os.path.join(suite2p_dir_base,setup,subject,fov)
temp_FOV_dir = os.path.join(local_temp_dir,'{}_{}_{}'.format(setup,subject,fov))    
sessions=os.listdir(FOV_dir)  
cell_masks = np.load(os.path.join(FOV_dir, 'cell_masks.npy'))
neuropil_masks = np.load(os.path.join(FOV_dir, 'neuropil_masks.npy'))
for session in sessions:
    if 'z-stack' in session.lower() or '.' in session:
        continue
    ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
    ops['batch_size']=250
    ops['nframes'] = sum(ops['nframes_list'])
    ops['reg_file'] = os.path.join(FOV_dir,session,'data.bin')
    print('extracting traces from {}'.format(session))
    F, Fneu, F_chan2, Fneu_chan2, ops = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
    np.save(os.path.join(FOV_dir,session,'F.npy'), F)
    np.save(os.path.join(FOV_dir,session,'Fneu.npy'), Fneu)