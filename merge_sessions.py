import os
import numpy as np
from suite2p.extraction.extract import extract_traces_from_masks
import sys
import scipy
import matplotlib.pyplot as plt
import json
from utils import utils_io
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
sessions_dict = {}
sessions=os.listdir(FOV_dir)  
cell_masks = np.load(os.path.join(FOV_dir, 'cell_masks.npy'), allow_pickle = True).tolist()
neuropil_masks = np.load(os.path.join(FOV_dir, 'neuropil_masks.npy'), allow_pickle = True).tolist()
for session in sessions:
    if 'z-stack' in session.lower() or '.' in session:
        continue
    if 'F.npy' not in os.listdir(os.path.join(FOV_dir,session)):
        print('traces not yet exported from {}'.format(session))
        continue
    F = np.load(os.path.join(FOV_dir,session,'F.npy'))
    F_filter = scipy.ndimage.gaussian_filter(F, [1,10])
    F0 = np.min(F_filter,1)
    Fneu = np.load(os.path.join(FOV_dir,session,'Fneu.npy'))
    sessions_dict[session] = {}
    sessions_dict[session]['F'] = F
    sessions_dict[session]['F0'] = F0
    sessions_dict[session]['Fneu'] = Fneu
    
# =============================================================================
#     raw_imaging_dir = os.path.join(raw_scanimage_dir_base,setup,subject,session)
#     background_values = []
#     background_subtracted_values = []
#     with open(os.path.join(FOV_dir,session,'filelist.json')) as f:
#         filelist_dict = json.load(f)
#     background_to_subtract = []
#     basename_prev = ''
#     for file_name,frame_num in zip(filelist_dict['file_name_list'],filelist_dict['frame_num_list']):
#         basename = file_name[:-1*file_name[::-1].find('_')-1]
#         if basename != basename_prev:
#             metadata = utils_io.extract_scanimage_metadata(os.path.join(raw_imaging_dir,file_name))
#             offsets = np.asarray(metadata['metadata']['hScan2D']['channelOffsets'].strip('[]').split(' '),int)
#             subtract_offset = np.asarray(metadata['metadata']['hScan2D']['channelsSubtractOffsets'].strip('[]').split(' '))=='true'
#             asas
#             if  not subtract_offset[0]:
#                 offset_value = 0
#             else:
#                 offset_value = offsets[0]
#             basename_prev = basename
#             #print(file_name)  
#         background_to_subtract.append(np.ones(frame_num)*offset_value)
#         background_values.append(offsets[0])
#         background_subtracted_values.append(subtract_offset[0])
#     background_to_subtract = np.concatenate(background_to_subtract)
# =============================================================================
#%% define F0 as the shot noise
for session in sessions_dict.keys():
    print(session)
    #F_filter = scipy.ndimage.gaussian_filter(sessions_dict[session]['F'], [1,10])
    F0 = np.zeros_like(sessions_dict[session]['F'])
    for cell_idx in range(sessions_dict[session]['F'].shape[0]):
        #cell_idx =445
        f = sessions_dict[session]['F'][cell_idx,:]
        sample_rate = 20
        window_t = .5 #s
        window = int(sample_rate*window_t)
        step=int(window/2)
        starts = np.arange(0,len(f)-window,step)
        stds = list()
        for start in starts:
            stds.append(np.var(f[start:start+window]))
        stds_roll = rollingfun(stds,100,'min')
        stds_roll = rollingfun(stds_roll,500,'median')
        #%
        f_scaled = np.copy(f)
        f0 = np.ones(len(f))
        for start,std in zip(starts,stds_roll):
            f_scaled[start:start+window]=(f[start:start+window]-std)/std
            f0[start:start+window]=std
        f_scaled[start:]=f[start:]/std
        f0[start:]=std
        F0[cell_idx,:] = f0
        #plt.plot(f_scaled)
    
    sessions_dict[session]['F0'] = F0
#%%
for session in sessions_dict.keys():
    F_filter = scipy.ndimage.gaussian_filter(sessions_dict[session]['F'], [1,20])
    #sessions_dict[session]['F0'] = F0
    #asdas

#%%
roi_idx = 10

fig = plt.figure()
ax_f = fig.add_subplot(3,1,1)
ax_f0 = fig.add_subplot(3,1,2,sharex = ax_f)
ax_dff = fig.add_subplot(3,1,3,sharex = ax_f)
dFF = []
F = []
Fneu = []
F0 = []
for session in sessions_dict.keys():
    F.append(sessions_dict[session]['F'][roi_idx,:])
    Fneu.append(sessions_dict[session]['Fneu'][roi_idx,:])
    dFF.append((sessions_dict[session]['F'][roi_idx,:]-sessions_dict[session]['F0'][roi_idx,:])/sessions_dict[session]['F0'][roi_idx,:])
    F0.append(sessions_dict[session]['F0'][roi_idx,:])
F = np.concatenate(F)
Fneu = np.concatenate(Fneu)
dFF = np.concatenate(dFF)
F0 = np.concatenate(F0)
ax_f.plot(F,'g-')
ax_f.plot(F0,'b-')
#ax_f.plot(Fneu,'r-')
ax_f0.plot(F0,'b-')

ax_dff.plot(dFF,'k-')
ax_dff.set_ylim(np.percentile(dFF,[1,99.5]))
#%% F0 changes

F_merged = []
dFF_merged = []
F0_merged = []
F0_scaled = []
session_frame_nums = []
reference_F0 = None
for session in sessions_dict.keys():
    F_merged.append(sessions_dict[session]['F'])
    #Fneu.append(sessions_dict[session]['Fneu'][roi_idx,:])
    dFF_merged.append((sessions_dict[session]['F']-sessions_dict[session]['F0'])/sessions_dict[session]['F0'])
    F0_merged.append(sessions_dict[session]['F0'])
    session_frame_nums.append(sessions_dict[session]['F'].shape[1])
    if reference_F0 is None:
        F0_scaled.append(sessions_dict[session]['F0'])
        reference_F0 = np.mean(sessions_dict[session]['F0'])
    else:
        multiplier = reference_F0/np.mean(sessions_dict[session]['F0'])
        F0_scaled.append(sessions_dict[session]['F0']*multiplier)
        
        
F_merged=np.concatenate(F_merged,1)
dFF_merged=np.concatenate(dFF_merged,1)
F0_merged=np.concatenate(F0_merged,1)
F0_scaled = np.concatenate(F0_scaled,1)
fig = plt.figure()
ax_f0 = fig.add_subplot(211)
ax_f0_corrected = fig.add_subplot(212)
ax_f0.plot(F0_merged[:50,:].T)
ax_f0.plot(np.mean(F0_merged,0),'k-',linewidth = 3)
for idx in np.cumsum(session_frame_nums):
    ax_f0.axvline(idx,color ='red')

ax_f0_corrected.plot(F0_scaled[:50,:].T)
ax_f0_corrected.plot(np.mean(F0_scaled,0),'k-',linewidth = 3)
for idx in np.cumsum(session_frame_nums):
    ax_f0_corrected.axvline(idx,color ='red')
#%%
dFF_merged_filter = scipy.ndimage.gaussian_filter(dFF_merged, [1,10])    
dFF_merged_filter_zscore = scipy.stats.zscore(dFF_merged_filter,0)
#%%
fig=plt.figure()
plt.imshow(dFF_merged_filter_zscore, aspect='auto')
for idx in np.cumsum(session_frame_nums):
    plt.axvline(idx,color ='red')
#%%tsne
from sklearn.manifold import TSNE  
#import umap
parameters_dict = {}
# ----------------------------- PARAMETERS -----------------------------
parameters_dict['pca_parameters'] = {'perplexity': 200,#3
                                     'learning_rate' : 10}#14#10#5#14
parameters_dict['umap_parameters'] = {'learning_rate': .4,#4
                                     'n_neigbors' : 30,#14#10#5#14
                                     'min_dist' : .01,
                                     'n_epochs':500}
# ----------------------------- PARAMETERS -----------------------------

tsne = TSNE(n_components=2, verbose=1, perplexity=parameters_dict['pca_parameters']['perplexity'], n_iter=1000,learning_rate=parameters_dict['pca_parameters']['learning_rate'])
tsne = tsne.fit_transform(dFF_merged_filter_zscore[:,::10].T)
#%% # tsne with multiple perplexities
tsne_list = []
for perplexity in [10,50,100,150,200,250,300,350]:
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=5000,learning_rate=parameters_dict['pca_parameters']['learning_rate'])
    tsne = tsne.fit_transform(dFF_merged_filter_zscore[:,::10].T)
    tsne_list.append({'perplexity':perplexity,
                      'tsne':tsne})
#%% plot tsne results
fig = plt.figure()
for perplexity_i,tsne_dict in enumerate(tsne_list):
    tsne = tsne_dict['tsne']
    ax_tsne = fig.add_subplot(2,4,perplexity_i+1)
    frame_num_so_far = 0
    for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
        session_frame_num = np.int(session_frame_num)
        ax_tsne.plot(tsne[frame_num_so_far:frame_num_so_far+session_frame_num,0],tsne[frame_num_so_far:frame_num_so_far+session_frame_num,1],'.',label = 'session {}'.format(session_i),alpha = .1)
        frame_num_so_far += session_frame_num
    ax_tsne.set_title('perplexity = {}'.format(tsne_dict['perplexity']))
    if perplexity_i == 0:
        ax_tsne.legend()
#%%
fig = plt.figure()
ax_tsne = fig.add_subplot(111)
frame_num_so_far = 0
for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
    session_frame_num = np.int(session_frame_num)
    ax_tsne.plot(tsne[frame_num_so_far:frame_num_so_far+session_frame_num,0],tsne[frame_num_so_far:frame_num_so_far+session_frame_num,1],'.',label = 'session {}'.format(session_i))
    frame_num_so_far += session_frame_num
    
ax_tsne.legend()



#%% umap
import umap

parameters_dict['umap_parameters'] = {'learning_rate': .4,#4
                                     'n_neigbors' : 5000,#14#10#5#14
                                     'min_dist' : .01,
                                     'n_epochs':5000}
reducer = umap.UMAP()
umap_list = []
for neighbors in [10,50,100,200,500,800,1500,5000,10000]:
    reducer.verbose=True
    reducer.learning_rate = parameters_dict['umap_parameters']['learning_rate']
    reducer.n_neigbors = neighbors#parameters_dict['umap_parameters']['n_neigbors']
    reducer.min_dist = parameters_dict['umap_parameters']['min_dist']
    reducer.n_epochs=parameters_dict['umap_parameters']['n_epochs']
    umap_data = reducer.fit_transform(dFF_merged_filter_zscore[:,::10].T)
    umap_list.append({'neighbors':neighbors,
                      'umap':umap_data})


#%%
#%%
fig = plt.figure()
ax_umap = fig.add_subplot(111)
frame_num_so_far = 0
for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
    session_frame_num = np.int(session_frame_num)
    ax_umap.plot(umap_data[frame_num_so_far:frame_num_so_far+session_frame_num,0],umap_data[frame_num_so_far:frame_num_so_far+session_frame_num,1],'.',label = 'session {}'.format(session_i))
    frame_num_so_far += session_frame_num
    
ax_tsne.legend()