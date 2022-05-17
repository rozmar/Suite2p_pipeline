#%% generate binned movies
import os, json
import numpy as np
from pathlib import Path
import  matplotlib.pyplot as plt
from suite2p.detection.detect import detect
local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/'
metadata_dir = '/mnt/Data/BCI_metadata/'
raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
subject = 'BCI_29'
setup = 'Bergamo-2P-Photostim'
fov = 'FOV_01'

minimum_contrast = 5
acceptable_z_range = 1


FOV_dir = os.path.join(suite2p_dir_base,setup,subject,fov)
temp_FOV_dir = os.path.join(local_temp_dir,'{}_{}_{}'.format(setup,subject,fov))
Path(temp_FOV_dir).mkdir(parents = True, exist_ok =True)
sessions=os.listdir(FOV_dir)
binned_movie_concatenated = []
zcorr_list_concatenated = []
xoff_mean_list_concatenated = []
yoff_mean_list_concatenated = []
xoff_std_list_concatenated = []
yoff_std_list_concatenated = []
trial_i = 0
new_session_idx =[]
median_z_values = []
for session in sessions:
    if 'z-stack' in session.lower() or '.' in session:
        continue
    new_session_idx.append(trial_i)
    with open(os.path.join(FOV_dir,session,'filelist.json')) as f:
        filelist_dict = json.load(f)
# =============================================================================
#     ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
#     zcorr_list_concatenated.append(filelist_dict['zoff_list'])
# =============================================================================
    
    concatenated_movie_filelist_json = os.path.join(FOV_dir,session,'filelist.json')
    with open(concatenated_movie_filelist_json, "r") as read_file:
        filelist_dict = json.load(read_file)
    xoff_mean_list_concatenated.append(filelist_dict['xoff_mean_list'])
    yoff_mean_list_concatenated.append(filelist_dict['yoff_mean_list'])
    xoff_std_list_concatenated.append(filelist_dict['yoff_std_list'])
    yoff_std_list_concatenated.append(filelist_dict['xoff_std_list'])
    zcorr_list_concatenated.append(filelist_dict['zoff_list'])
    trial_i += len(filelist_dict['xoff_mean_list'])
    median_z_values.append(np.median(np.argmax(filelist_dict['zoff_list'],2).squeeze()))
xoff_mean_list_concatenated = np.concatenate(xoff_mean_list_concatenated)        
yoff_mean_list_concatenated = np.concatenate(yoff_mean_list_concatenated)        
xoff_std_list_concatenated = np.concatenate(xoff_std_list_concatenated)        
yoff_std_list_concatenated = np.concatenate(yoff_std_list_concatenated)        
zcorr_list_concatenated = np.concatenate(zcorr_list_concatenated).squeeze()
max_zcorr_vals = np.max(zcorr_list_concatenated,1)
min_zcorr_vals = np.min(zcorr_list_concatenated,1)
contrast = max_zcorr_vals/min_zcorr_vals

zcorr_list_concatenated_norm = (zcorr_list_concatenated - min_zcorr_vals[:,np.newaxis])/(max_zcorr_vals-min_zcorr_vals)[:,np.newaxis]
hw = zcorr_list_concatenated_norm.shape[1]-np.argmax(zcorr_list_concatenated_norm[:,::-1]>.5,1) - np.argmax(zcorr_list_concatenated_norm>.5,1)
z_with_hw = (zcorr_list_concatenated_norm.shape[1]-np.argmax(zcorr_list_concatenated_norm[:,::-1]>.5,1) + np.argmax(zcorr_list_concatenated_norm>.5,1))/2

#%% quality check
median_z_value = np.median(median_z_values)

fig = plt.figure()
ax_z = fig.add_subplot(3,1,1)
ax_xy = fig.add_subplot(3,1,2,sharex = ax_z)
ax_zz = ax_xy.twinx()
ax_contrast = fig.add_subplot(3,1,3,sharex = ax_z)
ax_hw = ax_contrast.twinx()
img = zcorr_list_concatenated.T
ax_z.imshow(img ,aspect='auto', alpha = 1,origin='lower',cmap = 'magma')

x = np.arange(len(xoff_mean_list_concatenated))
ax_xy.errorbar(x, xoff_mean_list_concatenated,yerr = xoff_std_list_concatenated,fmt = '-',label = 'X offset')
ax_xy.errorbar(x, yoff_mean_list_concatenated,yerr = yoff_std_list_concatenated,fmt = '-',label = 'Y offset')

ax_zz.plot(x, np.argmax(zcorr_list_concatenated.squeeze(),1),'r-',label = 'Z offset')
ax_zz.plot(x, z_with_hw,'y-',label = 'Z offset with halfwidth')
ax_zz.plot([x[0],x[-1]],[median_z_value-acceptable_z_range]*2,'r--')
ax_zz.plot([x[0],x[-1]],[median_z_value+acceptable_z_range]*2,'r--')


ax_contrast.plot(contrast,label = 'Contrast in Z location')
ax_contrast.plot([x[0],x[-1]],[minimum_contrast]*2,'r--')
for idx in new_session_idx:
    ax_z.axvline(idx,color ='red')
   
ax_xy.legend()
ax_contrast.legend()


#%%
binned_movie_concatenated = []
ops_loaded=  False
for session in sessions:
    if 'z-stack' in session.lower() or '.' in session:
        continue
    if not ops_loaded:
        ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
        ops_loaded = True
    new_session_idx.append(trial_i)
    with open(os.path.join(FOV_dir,session,'filelist.json')) as f:
        filelist_dict = json.load(f)
    zcorr = np.asarray(filelist_dict['zoff_list']).squeeze()
    max_zcorr_vals = np.max(zcorr,1)
    min_zcorr_vals = np.min(zcorr,1)
    contrast = max_zcorr_vals/min_zcorr_vals
    median_z_session = np.median(np.argmax(filelist_dict['zoff_list'],2).squeeze())
    if np.percentile(contrast,10)<minimum_contrast:
        continue
    if median_z_session<median_z_value-acceptable_z_range or median_z_session>median_z_value+acceptable_z_range :
        continue
    print('loading the binned movie of {}'.format(session))
    mov = np.load(os.path.join(FOV_dir,session,'binned_movie.npy'))
    binned_movie_concatenated.append(mov)
binned_movie_concatenated = np.concatenate(binned_movie_concatenated)    
#%%

ops, stat = detect(ops, classfile=None, mov = binned_movie_concatenated)