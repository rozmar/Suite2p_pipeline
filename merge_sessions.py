import os
import numpy as np
from suite2p.extraction.extract import extract_traces_from_masks
import sys
import scipy
import matplotlib.pyplot as plt
import json
from utils import utils_io
%matplotlib qt
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
    subject = 'BCI_26'
    setup = 'Bergamo-2P-Photostim'
    fov = 'FOV_04'
FOV_dir = os.path.join(suite2p_dir_base,setup,subject,fov)
sessions_dict = {}
sessions=os.listdir(FOV_dir)  
cell_masks = np.load(os.path.join(FOV_dir, 'cell_masks.npy'), allow_pickle = True).tolist()
neuropil_masks = np.load(os.path.join(FOV_dir, 'neuropil_masks.npy'), allow_pickle = True).tolist()
stat = np.load(os.path.join(FOV_dir,'stat.npy'), allow_pickle = True).tolist()
mean_image = np.load(os.path.join(FOV_dir,'mean_image.npy'))
max_image = np.load(os.path.join(FOV_dir,'max_image.npy'))
for session in sessions:
    if 'z-stack' in session.lower() or '.' in session:
        continue
    if 'F.npy' not in os.listdir(os.path.join(FOV_dir,session)) or 'F0.npy' not in os.listdir(os.path.join(FOV_dir,session)):
        print('traces not yet exported from {}'.format(session))
        continue
    F = np.load(os.path.join(FOV_dir,session,'F.npy'))
    F0 = np.load(os.path.join(FOV_dir,session,'F0.npy'))
    Fneu = np.load(os.path.join(FOV_dir,session,'Fneu.npy'))
    Fvar = np.load(os.path.join(FOV_dir,session,'Fvar.npy'))
    sessions_dict[session] = {}
    sessions_dict[session]['F'] = F
    sessions_dict[session]['F0'] = F0
    sessions_dict[session]['Fvar'] = Fvar
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
#%%

# =============================================================================
# #%%
# for session in sessions_dict.keys():
#     F_filter = scipy.ndimage.gaussian_filter(sessions_dict[session]['F'], [1,20])
#     #sessions_dict[session]['F0'] = F0
#     #asdas
# 
# #%%
# =============================================================================
roi_idx = 1

fig = plt.figure()
ax_f = fig.add_subplot(5,1,1)
ax_f0 = fig.add_subplot(5,1,2,sharex = ax_f)
ax_dff = fig.add_subplot(5,1,3,sharex = ax_f)
ax_neuropil_corrected = fig.add_subplot(5,1,4,sharex = ax_f)
ax_mean_vals = fig.add_subplot(5,1,5,sharex = ax_f)
dFF = []
F = []
Fvar = []
Fneu = []
F0 = []
F_noise_scaled = []
mean_F = []
mean_Fneu = []
new_session_idx = [0]
for session in sessions_dict.keys():
    F.append(sessions_dict[session]['F'][roi_idx,:])
    Fvar.append(sessions_dict[session]['Fvar'][roi_idx,:])
    Fneu.append(sessions_dict[session]['Fneu'][roi_idx,:])
    dFF.append((sessions_dict[session]['F'][roi_idx,:]-sessions_dict[session]['F0'][roi_idx,:])/sessions_dict[session]['F0'][roi_idx,:])
    F_noise_scaled.append((sessions_dict[session]['F'][roi_idx,:]-sessions_dict[session]['F0'][roi_idx,:])/np.sqrt(sessions_dict[session]['F0'][roi_idx,:]))
    F0.append(sessions_dict[session]['F0'][roi_idx,:])
    mean_F.append(np.mean(sessions_dict[session]['F'],0))
    mean_Fneu.append(np.mean(sessions_dict[session]['Fneu'],0))
    new_session_idx.append(sessions_dict[session]['F'].shape[1])
new_session_idx=np.cumsum(new_session_idx)
for idx_start in new_session_idx:
    ax_f.axvline(idx_start,color ='red')
    
mean_F = np.concatenate(mean_F)
mean_Fneu = np.concatenate(mean_Fneu)
F = np.concatenate(F)
Fvar = np.concatenate(Fvar)
Fneu = np.concatenate(Fneu)
dFF = np.concatenate(dFF)
F0 = np.concatenate(F0)
F_noise_scaled = np.concatenate(F_noise_scaled)
ax_f.plot(F,'g-')

ax_f.plot(F0,'b-')
for idx_start in new_session_idx:
    ax_f.axvline(idx_start,color ='red')
ax_f0.set_ylabel('F')
ax_f0.plot(Fneu,'r-',alpha = .5)
ax_f0.plot(F0,'b-')
ax_f0.plot(Fvar,'k-')

ax_f0.set_ylabel('F0 & F_neu')
ax_dff.plot(dFF,'k-')
ax_dff.set_xlabel('Frames')
ax_dff.set_ylabel('dF/F')
#ax_dff.set_ylim(np.percentile(dFF,[1,99.5]))
ax_neuropil_corrected.plot(F-.7*Fneu,'k-')
ax_neuropil_corrected.set_ylabel('F neuropil corrected')

ax_mean_vals.plot(mean_F,'k-')
ax_mean_vals.plot(mean_Fneu,'g-')
ax_mean_vals.set_ylabel('mean F and mean F0 over all ROIs')
#%% F0 vs Fvar - pixel intensity per photon and offset
plot_stuff = True
session = list(sessions_dict.keys())[8]
for session in sessions_dict.keys():
    print(session)
    #session = list(sessions_dict.keys())[1]
    photon_counts_dict = {}
    bpod_file = os.path.join('/home/rozmar/Network/GoogleServices/BCI_data/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/{}/{}-bpod_zaber.npy'.format(subject,session))
    bpod_data=np.load(bpod_file,allow_pickle=True).tolist()
    tiff_idx = 0 #np.argmax((np.asarray(bpod_data['scanimage_file_names'])=='no movie for this trial')==False)
    while bpod_data['scanimage_file_names'][tiff_idx] =='no movie for this trial':
        tiff_idx +=1
    tiff_idx +=1
    tiff_header = bpod_data['scanimage_tiff_headers'][tiff_idx][0]
    mask = np.asarray(tiff_header['metadata']['hScan2D']['mask'].strip('[]').split(';'),int)
    dwelltime = 1000000/float(tiff_header['metadata']['hScan2D']['sampleRate'])
    F0_mean = np.median(sessions_dict[session]['F0'],1)
    Fvar_mean = np.median(sessions_dict[session]['Fvar'],1)
    imaging_power = np.asarray(tiff_header['metadata']['hBeams']['powers'].strip('[]').split(' '),float)[0]
    x_pos = []
    pixel_num = []
    dwell_time = []
    samples_averaged = []
    for s in stat:
        x_pos.append(s['med'][1])
        pixel_num.append(sum(s['soma_crop'] & (s['overlap']==False)))
        samples_averaged.append(np.sum(mask[s['xpix'][s['soma_crop'] & (s['overlap']==False)]]))#*dwelltime)
        dwell_time.append(np.sum(mask[s['xpix'][s['soma_crop'] & (s['overlap']==False)]])*dwelltime)
        #break
    
    
    
    p = np.polyfit(F0_mean,Fvar_mean*samples_averaged,1)
    intensity_per_photon = p[0]
    
    
    n_photons_per_roi = np.asarray(samples_averaged)*(F0_mean)/intensity_per_photon #+p[1]/p[0] - F0_mean is not corrected with the offset
    
    
    
    n_noise_photons_per_roi = np.asarray(samples_averaged)*(p[1]/p[0])/intensity_per_photon #+p[1]/p[0] - F0_mean is not corrected with the offset
    dff_1_snr = n_photons_per_roi/(np.sqrt((n_noise_photons_per_roi*3+n_photons_per_roi*2)/2))
    photon_counts_dict['F0_photon_counts'] = n_photons_per_roi
    photon_counts_dict['noise_photon_counts'] = n_noise_photons_per_roi
    
    photon_counts_dict['dprime_1dFF'] = dff_1_snr
    sessions_dict[session]['photon_counts'] = photon_counts_dict
    if plot_stuff:
        fig = plt.figure(figsize = [15,15])
        ax_var_mean = fig.add_subplot(3,2,1)
        ax_var_mean.set_title(session)
        ax_var_mean.plot(samples_averaged,Fvar_mean/F0_mean,'ko')
        ax_var_mean.set_ylabel('variance/mean of baseline fluorescence')
        ax_var_mean.set_xlabel('samples averaged per ROI')
        
        ax_var_mean_corrected = fig.add_subplot(3,2,2)
        ax_var_mean_corrected.plot(samples_averaged,(Fvar_mean/F0_mean)*samples_averaged,'ko')
        ax_var_mean_corrected.set_ylabel('variance/mean corrected with averaging')
        ax_var_mean_corrected.set_xlabel('samples averaged per ROI')
        
        ax_var_mean_scatter = fig.add_subplot(3,2,3)
        
        ax_var_mean_scatter.set_title('{} pixel values/photon, {} offset'.format(round(p[0],2),round(p[1]/p[0],2)))
        ax_var_mean_scatter.set_xlabel('mean of F0')
        ax_var_mean_scatter.set_ylabel('corrected variance of F0')
        ax_var_mean_scatter.plot(F0_mean,Fvar_mean*samples_averaged/p[0],'ko')
        
        ax_photons_per_roi = fig.add_subplot(3,2,4)
        ax_photons_per_roi.hist(n_photons_per_roi,100)
        ax_photons_per_roi.set_xlabel('# of photons collected per ROI per frame during baseline')
        ax_photons_per_roi.set_ylabel('# of ROIs')
        
        ax_noise_photons_per_roi = fig.add_subplot(3,2,6)
        ax_noise_photons_per_roi.hist(n_noise_photons_per_roi,100)
        ax_noise_photons_per_roi.set_xlabel('# of noise photons collected per ROI per frame')
        ax_noise_photons_per_roi.set_ylabel('# of ROIs')
        ax_dprime = fig.add_subplot(3,2,5)
        ax_dprime.hist(dff_1_snr,100)
        ax_dprime.set_xlabel('d-prime for 100% dF/F change')
        ax_dprime.set_ylabel('# of ROIs')
# =============================================================================
#         fig.savefig(os.path.join(FOV_dir,'photon_count_0.pdf'), format="pdf")
#     
#         asdas
# =============================================================================
#%% simulate shot noise
framenum = 1000
photon_count = 10
intensity_per_photon = 380
trace = []
samples_to_average = 400
for frame_i in range(framenum):
    trace.append(np.mean(np.random.poisson(photon_count,samples_to_average)*intensity_per_photon))
print('mean: {}, variance: {}'.format(np.mean(trace),np.var(trace))) 
plt.plot(trace)

#%% calculate  neuropil contribution for each session
import scipy.ndimage as ndimage
for session in sessions_dict.keys():
#session = list(sessions_dict.keys())[0]
    neuropil_dict = {}
    F = sessions_dict[session]['F']
    Fneu = sessions_dict[session]['Fneu']
    F0 = sessions_dict[session]['F0']
    needed_idx = rollingfun(np.mean(F,0),20,'min')> np.median(rollingfun(np.mean(F,0),20,'min'))/2
    neuropil_dict['good_indices'] = needed_idx
    fneu_mean = np.mean(Fneu,0)
    fneu_mean = fneu_mean[needed_idx]
    slopes = []
    slopes_mean_fneu = []
    neuropil_contamination = []
    neuropil_contamination_mean_fneu = []
    for cell_idx in range(F.shape[0]): 
        #cell_idx =2
        
        f = rollingfun(F[cell_idx,needed_idx],5,'mean')
        fneu = rollingfun(Fneu[cell_idx,needed_idx],5,'mean')
        f0 = rollingfun(F0[cell_idx,needed_idx],5,'mean')
        sample_rate = 20
        window_t = 1 #s
        window = int(sample_rate*window_t)
        step=int(window/2)
        starts = np.arange(0,len(f)-window,step)
        stds = list()
        means = list()
        f0_diffs = list()
        for start in starts:
            stds.append(np.var(f[start:start+window]))
            means.append(np.mean(f[start:start+window]))
            f0_diffs.append(np.mean(f[start:start+window])-np.mean(f0[start:start+window]))
            
        
        needed_segments = np.where(np.asarray(f0_diffs)/np.mean(f0)<.05)[0]
        f_points = []
        fneu_points = []
        fneu_mean_points = []
        
        
        f_filt_ultra_low =  ndimage.minimum_filter(f, size=int(1000))
        f_filt_highpass = f-f_filt_ultra_low + f_filt_ultra_low[0]
        fneu_filt_ultra_low =  ndimage.minimum_filter(fneu, size=int(1000))
        fneu_filt_highpass = fneu-fneu_filt_ultra_low + fneu_filt_ultra_low[0]
        fneu_std_list = []
        
        for segment in needed_segments:
            start_idx = starts[segment]
            f_points.append(f_filt_highpass[start_idx:start_idx+window])
            fneu_points.append(fneu_filt_highpass[start_idx:start_idx+window])
            fneu_mean_points.append(fneu_mean[start_idx:start_idx+window])
            fneu_std_list.append(np.std(fneu_filt_highpass[start_idx:start_idx+window]))
        sd_order = np.argsort(fneu_std_list)[::-1]
        needed = sd_order[:30] # 30 seconds of data is used
        f_points = np.concatenate(np.asarray(f_points)[needed,:])
        fneu_points = np.concatenate(np.asarray(fneu_points)[needed,:])
        fneu_mean_points = np.concatenate(np.asarray(fneu_mean_points)[needed,:])
        
        p = np.polyfit(fneu_points,f_points,1)
        slopes.append(p[0])
        neuropil_contamination.append(np.mean(fneu_points)*p[0]/np.mean(f_points))
        p = np.polyfit(fneu_mean_points,f_points,1)
        slopes_mean_fneu.append(p[0])
        neuropil_contamination_mean_fneu.append(np.mean(fneu_mean_points)*p[0]/np.mean(f_points))
        #asdasd
    neuropil_dict['r_neu_local'] = np.asarray(slopes)
    neuropil_dict['r_neu_global'] = np.asarray(slopes_mean_fneu)
    neuropil_dict['neuropil_contribution_to_f0_local'] = np.asarray(neuropil_contamination)
    neuropil_dict['neuropil_contribution_to_f0_global'] = np.asarray(neuropil_contamination_mean_fneu)
    sessions_dict[session]['neuropil_dict'] = neuropil_dict
       # break
   
#asdadas
#%%
fig = plt.figure()
ax_neuropil_dist = fig.add_subplot(2,1,1)
ax_neuropil_dist.set_ylabel('neuropil contribution to F0')
ax_f0_vs_rneu = fig.add_subplot(2,1,2)
rneu_list = []
F0_list = []
good_sessions = [0,1]#[0,1,2,3,4,5,8,9]
for session_i,session in enumerate(sessions_dict.keys()):
# =============================================================================
#     if session_i not in good_sessions or session_i>3:
#         continue
# =============================================================================
    rneu_list.append(sessions_dict[session]['neuropil_dict']['neuropil_contribution_to_f0_global'])
    F0_list.append(np.mean(F0,1))
    #%
mean_rneus = np.mean(np.asarray(rneu_list),0)
std_rneus = np.std(np.asarray(rneu_list),0)
order = np.argsort(mean_rneus)

for i,idx in enumerate(order):
    rneu = np.asarray(rneu_list)[:,idx]
    ax_neuropil_dist.plot(np.zeros(len(rneu))+i,rneu,'ko')
ax_neuropil_dist.errorbar(np.arange(len(order)),mean_rneus[order],std_rneus[order],fmt='o')

ax_f0_vs_rneu.plot(np.mean(np.asarray(F0_list),0),mean_rneus,'ko')

#%%

rois = np.zeros([800,800])
for s,rneu in zip(stat,mean_rneus):
    idx = (s['soma_crop']==True) & (s['overlap']==False)
    rois[s['ypix'][idx],s['xpix'][idx]] =rneu
    #%
fig = plt.figure()
ax_roi = fig.add_subplot(221)
ax_roi.imshow(rois)
ax_meanimage = fig.add_subplot(222,sharex = ax_roi,sharey = ax_roi)
ax_meanimage.imshow(mean_image)
ax_trace = fig.add_subplot(223)
idx = np.argmax(mean_rneus)
ax_trace.plot(F[idx,:],'g-')
ax_trace.plot(Fneu[idx,:],'r-')
ax_trace.plot(np.mean(Fneu,0),'k-')

#%%

session = list(sessions_dict.keys())[1]
cell_idx = 0


F = sessions_dict[session]['F']
Fneu = sessions_dict[session]['Fneu']
F0 = sessions_dict[session]['F0']
needed_idx = sessions_dict[session]['neuropil_dict']['good_indices']
fneu_mean = np.mean(Fneu,0)
fneu_mean = fneu_mean[needed_idx]

f = rollingfun(F[cell_idx,needed_idx],5,'mean')
fneu = rollingfun(Fneu[cell_idx,needed_idx],5,'mean')
f0 = rollingfun(F0[cell_idx,needed_idx],5,'mean')
fig  = plt.figure()
ax_trace = fig.add_subplot(3,2,1)
ax_trace.plot(f,'g-')
ax_trace.plot(fneu,'r-')
ax_trace.plot(fneu_mean,'k-')
ax_std_mean =fig.add_subplot(3,2,2)
ax_std_mean.hist(sessions_dict[session]['neuropil_dict']['r_neu_local'],np.arange(0,2,.05))
ax_std_mean.set_ylabel('cells')
ax_std_mean.set_xlabel('r_neu_local')
ax_neuropil_contribution =fig.add_subplot(3,2,3)
ax_neuropil_contribution.hist(sessions_dict[session]['neuropil_dict']['neuropil_contribution_to_f0_local'],np.arange(0,2,.05))
ax_neuropil_contribution.set_xlabel('neuropil contribution local')
ax_neu_r_global =fig.add_subplot(3,2,4)
ax_neu_r_global.hist(sessions_dict[session]['neuropil_dict']['r_neu_global'],np.arange(0,2,.05))
ax_neu_r_global.set_ylabel('cells')
ax_neu_r_global.set_xlabel('r_neu_global')
ax_neuropil_contribution_global =fig.add_subplot(3,2,5)
ax_neuropil_contribution_global.hist(sessions_dict[session]['neuropil_dict']['neuropil_contribution_to_f0_global'],np.arange(0,2,.05))
ax_neuropil_contribution_global.set_ylabel('cells')
ax_neuropil_contribution_global.set_xlabel('neuropil contribution global')
ax_neuropil_contribution_vs_f0 =fig.add_subplot(3,2,6)
ax_neuropil_contribution_vs_f0.plot(np.median(F0,1),sessions_dict[session]['neuropil_dict']['neuropil_contribution_to_f0_global'],'ko')
ax_neuropil_contribution_vs_f0.set_xlabel('median F0')
ax_neuropil_contribution_vs_f0.set_ylabel('neuropil contribution global')

#%% F0 changes

F_merged = []
Fneu_merged = []
dFF_merged = []
F0_merged = []
session_frame_nums = []
neuropil_contribution = []
dprime = []
for session in sessions_dict.keys():
    F_merged.append(sessions_dict[session]['F'])
    Fneu_merged.append(sessions_dict[session]['Fneu'])
    #Fneu.append(sessions_dict[session]['Fneu'][roi_idx,:])
    dFF_merged.append((sessions_dict[session]['F']-sessions_dict[session]['F0'])/sessions_dict[session]['F0'])
    F0_merged.append(sessions_dict[session]['F0'])
    session_frame_nums.append(sessions_dict[session]['F'].shape[1])
    neuropil_contribution.append(sessions_dict[session]['neuropil_dict']['neuropil_contribution_to_f0_global'])
    dprime.append(sessions_dict[session]['photon_counts']['dprime_1dFF'])
    

        
        
F_merged=np.concatenate(F_merged,1)
Fneu_merged=np.concatenate(Fneu_merged,1)
dFF_merged=np.concatenate(dFF_merged,1)
F0_merged=np.concatenate(F0_merged,1)

#%
neuropil_contribution_merged =np.asarray(neuropil_contribution)
dprime_merged =np.asarray(dprime)
#%% do violin plots instead!!
cells_to_show = 10
fig = plt.figure()
ax_f0 = fig.add_subplot(511)
ax_f0.set_ylabel('F0')
ax_fneu = fig.add_subplot(512,sharex = ax_f0)
ax_fneu.set_ylabel('neuropil contribution')
ax_dprime = fig.add_subplot(513,sharex = ax_f0)
ax_dprime.set_ylabel('d-prime of 100%dF/F')
ax_f0.plot(F0_merged[:cells_to_show,:].T)
ax_f0.plot(np.mean(F0_merged,0),'k-',linewidth = 3)
for idx in np.cumsum(session_frame_nums):
    ax_f0.axvline(idx,color ='red')
mean_session_frames = np.mean([np.concatenate([[0],np.cumsum(session_frame_nums)])[:-1],np.cumsum(session_frame_nums)],0)
for cell_i in range(cells_to_show):
    ax_fneu.plot(mean_session_frames,neuropil_contribution_merged[:,cell_i])
ax_fneu.plot(mean_session_frames,np.mean(neuropil_contribution_merged,1),'k-',linewidth = 3)

for cell_i in range(cells_to_show):
    ax_dprime.plot(mean_session_frames,dprime_merged[:,cell_i])
ax_dprime.plot(mean_session_frames,np.mean(dprime_merged,1),'k-',linewidth = 3)
#%%
dFF_merged_filter = scipy.ndimage.gaussian_filter(dFF_merged, [1,10])    
dFF_merged_filter_zscore = scipy.stats.zscore(dFF_merged_filter,0) #1 = within cell 0=zscore among cells
#%%
F_merged_filter = scipy.ndimage.gaussian_filter(F_merged, [1,10])    
F_merged_filter_zscore = scipy.stats.zscore(F_merged_filter,1) #1 = within cell 0=zscore among cells
#%%
F_corr_merged_filter = scipy.ndimage.gaussian_filter(F_merged-Fneu_merged*.7, [1,10])    
F_corr_merged_filter_zscore = scipy.stats.zscore(F_corr_merged_filter,1) #1 = within cell 0=zscore among cells
#%%
fig=plt.figure()
im_zscore = plt.imshow(dFF_merged_filter_zscore, aspect='auto')
im_zscore.set_clim([-1,5])
for idx in np.cumsum(session_frame_nums):
    plt.axvline(idx,color ='red')
#%%
from rastermap import Rastermap
model = Rastermap(n_components=1, n_X=200, nPC=200, init='pca')
#F_corr_merged_filter_zscore[np.isnan(F_corr_merged_filter_zscore)] = 0
model.fit(F_corr_merged_filter_zscore)
#%
order =model.isort
#%%
fig=plt.figure()
im_rastermap = plt.imshow(F_corr_merged_filter_zscore[order,:], aspect='auto')
im_rastermap.set_clim([-1,3])
for idx in np.cumsum(session_frame_nums):
    plt.axvline(idx,color ='red')
#%%
fig=plt.figure()
plt.imshow(dFF_merged_filter>1, aspect='auto')
for idx in np.cumsum(session_frame_nums):
    plt.axvline(idx,color ='red')
#%%tsne base
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
#%
tsne = TSNE(n_components=2, verbose=1, perplexity=parameters_dict['pca_parameters']['perplexity'], n_iter=1000,learning_rate=parameters_dict['pca_parameters']['learning_rate'])
tsne = tsne.fit_transform(dFF_merged_filter_zscore[:,::10].T)
#%% tsne plot single

fig = plt.figure()
ax_tsne = fig.add_subplot(111)
frame_num_so_far = 0
for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
    session_frame_num = np.int(session_frame_num)
    ax_tsne.plot(tsne[frame_num_so_far:frame_num_so_far+session_frame_num,0],tsne[frame_num_so_far:frame_num_so_far+session_frame_num,1],'.',label = 'session {}'.format(session_i))
    frame_num_so_far += session_frame_num
    
ax_tsne.legend()
#%% # tsne with multiple perplexities
tsne_list = []
for perplexity in [10,50,100,150,200,250,300,350]:
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=5000,learning_rate=parameters_dict['pca_parameters']['learning_rate'])
    tsne = tsne.fit_transform(dFF_merged_filter_zscore[:,::10].T)
    tsne_list.append({'perplexity':perplexity,
                      'tsne':tsne})
#%% plot tsne results parameter sweep
fig = plt.figure()
for perplexity_i,tsne_dict in enumerate(tsne_list):
    tsne = tsne_dict['tsne']
    ax_tsne = fig.add_subplot(2,4,perplexity_i+1)
    frame_num_so_far = 0
    for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
        session_frame_num = np.int(session_frame_num)
        ax_tsne.plot(tsne[frame_num_so_far:frame_num_so_far+session_frame_num,0],
                     tsne[frame_num_so_far:frame_num_so_far+session_frame_num,1],
                     '.',
                     label = 'session {}'.format(session_i),
                     alpha = .05)
        frame_num_so_far += session_frame_num
        if session_i == 5 :
            break
    ax_tsne.set_title('perplexity = {}'.format(tsne_dict['perplexity']))
    if perplexity_i == 0:
        leg = ax_tsne.legend(bbox_to_anchor=(-.05, 1.05))
        for lh in leg.legendHandles: 
            lh.set_alpha(1)




#%% umap parameter sweep
import umap
parameters_dict = {}
parameters_dict['umap_parameters'] = {'learning_rate': .4,#4
                                     'n_neigbors' : 5000,#14#10#5#14
                                     'min_dist' : .01,
                                     'n_epochs':500}
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


#%% plot umap results
highlight_session = None
fig = plt.figure()
for perplexity_i,umap_dict in enumerate(umap_list):
    umap = umap_dict['umap']
    ax_umap = fig.add_subplot(3,3,perplexity_i+1)
    frame_num_so_far = 0
    for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
        session_frame_num = np.int(session_frame_num)
        if session_i == highlight_session:
            x = umap[frame_num_so_far:frame_num_so_far+session_frame_num,0]
            y = umap[frame_num_so_far:frame_num_so_far+session_frame_num,1]
        
        ax_umap.plot(umap[frame_num_so_far:frame_num_so_far+session_frame_num,0],
                     umap[frame_num_so_far:frame_num_so_far+session_frame_num,1],
                     '.',
                     label = 'session {}'.format(session_i),
                     alpha = .05)
        frame_num_so_far += session_frame_num
        if session_i == 5 :
            break
        
# =============================================================================
#     frame_num_so_far = 0
#     for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
#         session_frame_num = np.int(session_frame_num)
#         if session_i == highlight_session:
#             x = umap[frame_num_so_far:frame_num_so_far+session_frame_num,0]
#             y = umap[frame_num_so_far:frame_num_so_far+session_frame_num,1]
#         
#         ax_umap.plot(umap[frame_num_so_far+int(session_frame_num/10)*1:frame_num_so_far+int(session_frame_num/10)*5,0],
#                      umap[frame_num_so_far+int(session_frame_num/10)*1:frame_num_so_far+int(session_frame_num/10)*5,1],
#                      'kx',
#                      alpha = .5)
#         frame_num_so_far += session_frame_num
#         if session_i == 5 :
#             break
#             
# =============================================================================
    if highlight_session is not None:
        ax_umap.plot(x,y,'kx',label = 'session {}'.format(highlight_session),alpha = 1)
    ax_umap.set_title('n neighbors = {}'.format(umap_dict['neighbors']))
    if perplexity_i == 0:
        leg = ax_umap.legend(bbox_to_anchor=(-.05, 1.05))
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
#%% umap single run
import umap
parameters_dict = {}
parameters_dict['umap_parameters'] = {'learning_rate': .4,#4
                                     'n_neigbors' : 500,#14#10#5#14
                                     'min_dist' : .01,
                                     'n_epochs':500}
reducer = umap.UMAP()

reducer.verbose=True
reducer.learning_rate = parameters_dict['umap_parameters']['learning_rate']
reducer.n_neigbors = parameters_dict['umap_parameters']['n_neigbors']
reducer.min_dist = parameters_dict['umap_parameters']['min_dist']
reducer.n_epochs=parameters_dict['umap_parameters']['n_epochs']
umap_data = reducer.fit_transform(dFF_merged_filter_zscore[:,::10].T)
#umap_data = reducer.fit_transform(dFF_merged_filter[:,::10].T>1)

#%%
fig = plt.figure()
ax_umap = fig.add_subplot(111)
frame_num_so_far = 0
for session_i,session_frame_num in enumerate(np.asarray(session_frame_nums)/10):
    session_frame_num = np.int(session_frame_num)
    ax_umap.plot(umap_data[frame_num_so_far:frame_num_so_far+session_frame_num,0],umap_data[frame_num_so_far:frame_num_so_far+session_frame_num,1],'.',label = 'session {}'.format(session_i),alpha=.2)
    frame_num_so_far += session_frame_num
    
ax_umap.legend()