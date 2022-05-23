

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
    try:
        overwrite ='true' in sys.argv[8].lower() #'Bergamo-2P-Photostim'
    except:
        overwrite = False

except:
    local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/'
    metadata_dir = '/mnt/Data/BCI_metadata/'
    raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
    suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
    subject = 'BCI_29'
    setup = 'Bergamo-2P-Photostim'
    fov = 'FOV_03'
    overwrite = True
    
def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    
    if window >len(y):
        window = len(y)-1
    y = np.concatenate([y[window::-1],y,y[:-1*window:-1]])
    ys = list()
    for idx in range(window):    
        ys.append(np.roll(y,idx-round(window/2)))
    if func =='mean':
        out = np.nanmean(ys,0)[window:-window]
    elif func == 'min':
        out = np.nanmin(ys,0)[window:-window]
    elif func == 'max':
        out = np.nanmax(ys,0)[window:-window]
    elif func == 'std':
        out = np.nanstd(ys,0)[window:-window]
    elif func == 'median':
        out = np.nanmedian(ys,0)[window:-window]
    else:
        print('undefinied funcion in rollinfun')
    return out    
    
    
FOV_dir = os.path.join(suite2p_dir_base,setup,subject,fov)
temp_FOV_dir = os.path.join(local_temp_dir,'{}_{}_{}'.format(setup,subject,fov))    
sessions=os.listdir(FOV_dir)  
cell_masks = np.load(os.path.join(FOV_dir, 'cell_masks.npy'), allow_pickle = True).tolist()
neuropil_masks = np.load(os.path.join(FOV_dir, 'neuropil_masks.npy'), allow_pickle = True).tolist()
for session in sessions:
    if 'z-stack' in session.lower() or '.' in session:
        continue
    if 'F.npy' not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
        
        ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
        ops['batch_size']=250
        ops['nframes'] = sum(ops['nframes_list'])
        ops['reg_file'] = os.path.join(FOV_dir,session,'data.bin')
        print('extracting traces from {}'.format(session))
        F, Fneu, F_chan2, Fneu_chan2, ops = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
        np.save(os.path.join(FOV_dir,session,'F.npy'), F)
        np.save(os.path.join(FOV_dir,session,'Fneu.npy'), Fneu)
    else:
        F = np.load(os.path.join(FOV_dir,session,'F.npy'))
    if 'F0.npy' not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
        F0 = np.zeros_like(F)
        Fvar = np.zeros_like(F)
        print('calculating f0 for {}'.format(session))
        for cell_idx in range(F.shape[0]):
            #cell_idx =445
            f = F[cell_idx,:]
            sample_rate = 20
            window_t = 1 #s
            window = int(sample_rate*window_t)
            step=int(window/2)
            starts = np.arange(0,len(f)-window,step)
            stds = list()
            means = list()
            for start in starts:
                stds.append(np.var(f[start:start+window]))
                means.append(np.mean(f[start:start+window]))
            stds_roll = rollingfun(stds,100,'min')
            stds_roll = rollingfun(stds_roll,500,'median')
            
            means_roll = rollingfun(means,100,'min')
            means_roll = rollingfun(means_roll,500,'median')
            
            #%
            f_scaled = np.copy(f)
            f0 = np.ones(len(f))
            fvar = np.ones(len(f))
            for start,var,fzero in zip(starts,stds_roll,means_roll):
                f0[start:start+window]=fzero
                fvar[start:start+window]=var
            f0[start:]=fzero
            fvar[start:]=var
            F0[cell_idx,:] = f0
            Fvar[cell_idx,:] = fvar
        np.save(os.path.join(FOV_dir,session,'F0.npy'), F0)
        np.save(os.path.join(FOV_dir,session,'Fvar.npy'), Fvar)