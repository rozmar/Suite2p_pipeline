import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
import numpy as np
from suite2p.extraction.extract import extract_traces_from_masks
from suite2p.registration.nonrigid import upsample_block_shifts
import math
from utils import utils_io

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        
    Source:
        https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
    

def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    window = int(window)
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

def remove_stim_artefacts(F,Fneu,frames_per_file):
    """
    removing stimulation artefacts with linear interpolation
    and nan-ing out tripped PMT traces

    Parameters
    ----------
    F : matrix of float
        Fluorescence of ROIs
    Fneu : matrix of float
        fluorescence of neuropil
    frames_per_file : list of int
        # of frames in each file (where the photostim happens)

    Returns
    -------
    F : matrix of float
        corrected fluorescence of ROIs
    Fneu : matrix of float
        corrected fluorescence of neuropil

    """
    artefact_indices = []
    fneu_mean = np.mean(Fneu,0)
    
    for stim_idx in np.concatenate([[0],np.cumsum(frames_per_file)[:-1]]):
        idx_now = []
        if stim_idx>0 and fneu_mean[stim_idx-2]*1.1<fneu_mean[stim_idx-1]:
            idx_now.append(stim_idx-1)
        idx_now.append(stim_idx)
        if stim_idx<len(fneu_mean)-2 and fneu_mean[stim_idx+2]*1.1<fneu_mean[stim_idx+1]:
            idx_now.append(stim_idx+1)
        artefact_indices.append(idx_now)
    
    f_std = np.std(F,0)
    pmt_off_indices = f_std<np.median(f_std)-3*np.std(f_std)
    pmt_off_edges = np.diff(np.concatenate([pmt_off_indices,[0]]))
    pmt_off_indices[pmt_off_edges!=0] = 1 #dilate 1
    pmt_off_edges = np.diff(np.concatenate([[0],pmt_off_indices,[0]]))
    starts = np.where(pmt_off_edges==1)[0]
    ends = np.where(pmt_off_edges==-1)[0]
    lengths = ends-starts
    for idx in np.where(lengths<=10)[0]:
        pmt_off_indices[starts[idx]:ends[idx]]=0
    
    
    F_ = F.copy()
    F_[:,np.concatenate(artefact_indices)]=np.nan
    for f in F_:
        nans, x= nan_helper(f)
        f[nans]= np.interp(x(nans), x(~nans), f[~nans])
        f[pmt_off_indices] = np.nan
    F = F_
    
    Fneu_ = Fneu.copy()
    Fneu_[:,np.concatenate(artefact_indices)]=np.nan
    for f in Fneu_:
        nans, x= nan_helper(f)
        f[nans]= np.interp(x(nans), x(~nans), f[~nans])
        f[pmt_off_indices] = np.nan
    Fneu = Fneu_
    return F, Fneu

def align_trace_to_event(F,
                         event_indices,
                         frames_before,
                         frames_after):
    """
    Creates an event-locked array of traces with given size

    Parameters
    ----------
    F : matrix of float
        sessionwise fluorescence
    event_indices : list of int
        indices of events (output of trial_times_to_session_indices )
    frames_before : int
        number of frames to keep before event
    frames_after : int
        number of frames to keep after event.

    Returns
    -------
    F_aligned : matrix of float (frames * cells * trials)
        trial-locked matrix

    """
    max_frames= frames_before+frames_after
    F_aligned = np.ones((max_frames, F.shape[0], len(event_indices)))*np.nan
    
    for i, center_idx in enumerate(event_indices):

        start_frame = center_idx - frames_before
        end_frame = center_idx + frames_after
        if end_frame > F.shape[1]:
            end_frame = F.shape[1]
        
        if start_frame<0: # taking care of edge at the beginning
            missing_frames_at_beginning = np.abs(start_frame)
            start_frame = 0
        else:
            missing_frames_at_beginning = 0
        F_aligned[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, i] = F[:, start_frame:end_frame].T
    return F_aligned

def extract_traces_core(subject,
                        FOV_dir,
                        session,
                        setup,
                        overwrite,
                        cell_masks,
                        neuropil_masks,
                        bpod_path,
                        photostim =False,
                        roi_type = '',
                        use_red_channel = False):

    if photostim:
        session = session+'/photostim'
    try:
        os.listdir(os.path.join(FOV_dir,session))
    except:
        return
    if 'F{}.npy'.format(roi_type) not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
        
        ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
        ops['batch_size']=250
        if not photostim:
            ops['nframes'] = sum(ops['nframes_list'])

        ops['reg_file'] = os.path.join(FOV_dir,session,'data.bin')
        if 'reg_file_chan2' in ops:
            ops['reg_file_chan2'] = os.path.join(FOV_dir,session,'data_chan2.bin')
        print('extracting traces from {}'.format(session))
        F, Fneu, F_chan2, Fneu_chan2, ops = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
        np.save(os.path.join(FOV_dir,session,'F{}.npy'.format(roi_type)), F)
        np.save(os.path.join(FOV_dir,session,'Fneu{}.npy'.format(roi_type)), Fneu)
        if 'reg_file_chan2' in ops:
            np.save(os.path.join(FOV_dir,session,'F_chan2{}.npy'.format(roi_type)), F_chan2)
            np.save(os.path.join(FOV_dir,session,'Fneu_chan2{}.npy'.format(roi_type)), Fneu_chan2)
            if use_red_channel:
                F = F_chan2
                Fneu = Fneu_chan2
    elif use_red_channel and 'reg_file_chan2' in ops:
        F = np.load(os.path.join(FOV_dir,session,'F_chan2{}.npy'.format(roi_type)))
        Fneu = np.load(os.path.join(FOV_dir,session,'Fneu_chan2{}.npy'.format(roi_type)))
    else:
        F = np.load(os.path.join(FOV_dir,session,'F{}.npy'.format(roi_type)))
        Fneu = np.load(os.path.join(FOV_dir,session,'Fneu{}.npy'.format(roi_type)))
    if photostim: # find stim artefacts and when PMT is off
        ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
        F,Fneu = remove_stim_artefacts(F,Fneu,ops['frames_per_file'])  
     
        #%%
    if 'F0{}.npy'.format(roi_type) not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
        #%%
        F0 = np.zeros_like(F)
        Fvar = np.zeros_like(F)
        print('calculating f0 for {}'.format(session))
        f0_offsets = []
        for cell_idx in range(F.shape[0]):
            #%
            #cell_idx =460
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
            #f_scaled = np.copy(f)
            f0 = np.ones(len(f))
            fvar = np.ones(len(f))
            for start,var,fzero in zip(starts,stds_roll,means_roll):
                f0[start:start+window]=fzero
                fvar[start:start+window]=var
            f0[start:]=fzero
            fvar[start:]=var
            F0[cell_idx,:] = f0
            Fvar[cell_idx,:] = fvar
            dff = (f-f0)/f0
            #%
            try:
                f_ = []
                f0_ = []
                for i in np.where(stds[:int(len(stds)/2)]<np.percentile(stds[:int(len(stds)/2)],20))[0]: # polish dff on the lower 20 percent of F values
                    f_.append(f[starts[i]:starts[i]+window])
                    f0_.append(f0[starts[i]:starts[i]+window])
                f_ = np.concatenate(f_)
                f0_ = np.concatenate(f0_)
                dff_ = (f_-f0_)/f0_
                
                #%
                
                c,b = np.histogram(dff_,np.arange(-2,2,.05))
                b=np.mean([b[1:],b[:-1]],0)
                needed = b>-.5
                c = c[needed]
                b=b[needed]
                #%
                f0_offsets.append(b[np.argmax(c)])
            except:
                print('f0 estimation could not be corrected for roi {}'.format(cell_idx))
                f0_offsets.append(0)
            
                
            #%%
        F0 = F0*(np.median(f0_offsets)+1)
        np.save(os.path.join(FOV_dir,session,'F0{}.npy'.format(roi_type)), F0)
        np.save(os.path.join(FOV_dir,session,'Fvar{}.npy'.format(roi_type)), Fvar)
    else:
        F0 = np.load(os.path.join(FOV_dir,session,'F0{}.npy'.format(roi_type)))
        Fvar = np.load(os.path.join(FOV_dir,session,'Fvar{}.npy'.format(roi_type)))
    
# =============================================================================
#         #%%
#         f0_offsets = []
#         for cell_idx in range(F.shape[0]):
#             
#             f = F[cell_idx,:]
#             f0 = F0[cell_idx,:]
#             dff = (f-f0)/f0
#             c,b = np.histogram(dff,np.arange(-1,2,.05))
#             f0_offsets.append(b[np.argmax(c)])
# =============================================================================
    #%%
    if 'channel_offset{}.npy'.format(roi_type) not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
        #%%
        # this script estimates the error in scanimage baselne measurement
        # due to inaccurate values, the dF/F amplitudes of low F0 cells
        # are getting huge. This script finds the correct offset by
        # minimizing the variance of maximum dF/F amplitudes across
        # all cells
        f0_scalar = np.mean(F0,1)
        dff = (F-F0)/F0
        f0_corrections = np.arange(np.min([0,-np.min(f0_scalar)]),100,.1)
        #max_amplitudes = np.asarray(data_dict[session]['max_event_amplitude'])
        max_amplitudes = np.nanmax(dff,1)#np.percentile(dff,95,1)#
        amplitudes = max_amplitudes
        amplitudes = amplitudes[np.isnan(amplitudes)==False]
        percentiles = [1,99]
        f0_correction = 55
        peak_is_pronounced = False
        while (f0_correction>50 or not peak_is_pronounced) and np.diff(percentiles)>50:
            var_list = []
            percentiles[0]+=1
            #percentiles[1]-=1
            needed = (f0_scalar>np.percentile(f0_scalar,percentiles[0])) & \
                (f0_scalar<np.percentile(f0_scalar,percentiles[1])) &  \
                (max_amplitudes>np.percentile(amplitudes,percentiles[0])) & \
                (max_amplitudes<np.percentile(amplitudes,percentiles[1])) &  \
                (np.isnan(max_amplitudes)==False) & \
                (f0_scalar >0) &\
                    (max_amplitudes>0)
                    
            for f0_correction in f0_corrections:
                f0_modified = f0_scalar+f0_correction
                dff_modified = max_amplitudes*f0_scalar/f0_modified
                var_list.append(np.std(dff_modified[needed])/np.mean(dff_modified[needed]))
           
            peak_is_pronounced = (var_list[0]-np.min(var_list))/(var_list[-1]-np.min(var_list))<2
            f0_correction  = f0_corrections[np.argmin(var_list)]


        fig = plt.figure()
        ax_original_f0 = fig.add_subplot(3,1,1)
        ax_original_f0.set_title(session)
        ax_original_f0.set_xlabel('f0')
        ax_original_f0.set_ylabel('max df/f')
        ax_hacked_f0 = fig.add_subplot(3,1,2,sharex = ax_original_f0)
        ax_hacked_f0.set_xlabel('f0 corrected with {} pixel values'.format(f0_correction))
        ax_hacked_f0.set_ylabel('max df/f')
        ax_original_f0.semilogy(f0_scalar,max_amplitudes,'ko')
        f0_modified = f0_scalar+f0_correction
        dff_modified = max_amplitudes*f0_scalar/f0_modified
        ax_hacked_f0.semilogy(f0_modified,dff_modified,'ko')

        ax_hacked_var = fig.add_subplot(3,1,3)

        ax_hacked_var.plot(f0_corrections,var_list)
        ax_hacked_var.set_xlabel('offset')
        ax_hacked_var.set_ylabel('std/mean')
        f0_correction_dict = {'channel_offset':f0_correction}
        #%%
        fig.savefig(os.path.join(FOV_dir,session,'F0_offset{}.pdf'.format(roi_type)), format="pdf")
        plt.close()
        np.save(os.path.join(FOV_dir,session,'channel_offset{}.npy'.format(roi_type)), f0_correction_dict,allow_pickle=True)
    else:
        f0_correction_dict = np.load(os.path.join(FOV_dir,session,'channel_offset{}.npy'.format(roi_type)),allow_pickle=True).tolist()
    #%%
# =============================================================================
#         F+=f0_correction_dict['channel_offset']
#         F0+=f0_correction_dict['channel_offset']
#         Fneu+=f0_correction_dict['channel_offset']
# =============================================================================
    if ('neuropil_contribution{}.npy'.format(roi_type) not in os.listdir(os.path.join(FOV_dir,session)) or overwrite) and not photostim:
        neuropil_dict = {}
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
            
            needed_segments = []
            df_max = .05
            while len(needed_segments)<30:
                needed_segments = np.where(np.asarray(f0_diffs)/np.mean(f0)<df_max)[0]
                df_max+=.05
                if df_max > 10:
                    break
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
        np.save(os.path.join(FOV_dir,session,'neuropil_contribution{}.npy'.format(roi_type)), neuropil_dict,allow_pickle=True)
# =============================================================================
#     else:
#         neuropil_dict = np.load(os.path.join(FOV_dir,session,'neuropil_contribution{}.npy'.format(roi_type)),allow_pickle=True).tolist()
# =============================================================================
      
    
    if ('photon_counts.npy' not in os.listdir(os.path.join(FOV_dir,session)) or overwrite) and roi_type == '' and not photostim: # photon counts only for the big ROIs
        #%%
        plot_stuff = True
        stat = np.load(os.path.join(FOV_dir,'stat.npy'), allow_pickle = True).tolist()
        photon_counts_dict = {}
        if photostim:
            bpod_file = os.path.join(bpod_path,setup,'{}/{}-bpod_zaber.npy'.format(subject,session[:session.find('/')]))
            
        else:
            bpod_file = os.path.join(bpod_path,setup,'{}/{}-bpod_zaber.npy'.format(subject,session))
        bpod_data=np.load(bpod_file,allow_pickle=True).tolist()
        tiff_idx = 0 #np.argmax((np.asarray(bpod_data['scanimage_file_names'])=='no movie for this trial')==False)
        while str(bpod_data['scanimage_file_names'][tiff_idx]) =='no movie for this trial':
            tiff_idx +=1
        
        tiff_header = bpod_data['scanimage_tiff_headers'][tiff_idx][0]
        mask = np.asarray(tiff_header['metadata']['hScan2D']['mask'].strip('[]').split(';'),int)
        dwelltime = 1000000/float(tiff_header['metadata']['hScan2D']['sampleRate'])
        F0_mean = np.median(F0,1)
        Fvar_mean = np.median(Fvar,1)
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
        dff_1_snr = n_photons_per_roi/(np.sqrt(np.abs((n_noise_photons_per_roi*2+n_photons_per_roi*3)/2)))
        photon_counts_dict['F0_photon_counts'] = n_photons_per_roi
        photon_counts_dict['noise_photon_counts'] = n_noise_photons_per_roi
        
        photon_counts_dict['dprime_1dFF'] = dff_1_snr
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
            #%%
            fig.savefig(os.path.join(FOV_dir,session,'photon_counts.pdf'), format="pdf")
            plt.close()
        np.save(os.path.join(FOV_dir,session,'photon_counts.npy'), photon_counts_dict,allow_pickle=True)
        
# =============================================================================
#     else:
#         photon_counts_dict = np.load(os.path.join(FOV_dir,session,'photon_counts.npy'),allow_pickle=True).tolist()
#     del photon_counts_dict, neuropil_dict, F, F0, Fneu
# =============================================================================
# =============================================================================
#     if 'dFF_noise.npy' not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
#         dfFnoise = np.zeros_like(F)
#         print('calculating dff noise for {}'.format(session))
#         for cell_idx in range(dfFnoise.shape[0]):
#             #cell_idx =445
#             dff = (F[cell_idx,:]-F0[cell_idx,:])/F0[cell_idx,:]
#             sample_rate = 20
#             window_t = 1 #s
#             window = int(sample_rate*window_t)
#             step=int(window/2)
#             starts = np.arange(0,len(dff)-window,step)
#             stds = list()
#             for start in starts:
#                 stds.append(np.std(dff[start:start+window]))
#             stds_roll = rollingfun(stds,100,'min')
#             stds_roll = rollingfun(stds_roll,500,'median')
#           
#             fvar = np.ones(len(dff))
#             for start,var in zip(starts,stds_roll):
#                 fvar[start:start+window]=var
#             fvar[start:]=var
#             dfFnoise[cell_idx,:] = fvar
#         np.save(os.path.join(FOV_dir,session,'dFF_noise.npy'), dfFnoise)
# =============================================================================
    

def extract_traces(local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/',
                   metadata_dir = '/mnt/Data/BCI_metadata/',
                   raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/',
                   suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/',
                   bpod_path = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Behavior/BCI_exported/',
                   subject = 'BCI_26',
                   setup = 'Bergamo-2P-Photostim',
                   fov = 'FOV_06',
                   overwrite = True,
                   roi_types = [''],
                   photostim = False,
                  use_red_channel = False):
    FOV_dir = os.path.join(suite2p_dir_base,setup,subject,fov)
    sessions=os.listdir(FOV_dir)  
    for roi_type in roi_types:
        cell_masks = np.load(os.path.join(FOV_dir, 'cell_masks{}.npy'.format(roi_type)), allow_pickle = True).tolist()
        neuropil_masks = np.load(os.path.join(FOV_dir, 'neuropil_masks{}.npy'.format(roi_type)), allow_pickle = True).tolist()
        for session in sessions:
            if 'z-stack' in session.lower() or '.' in session:
                continue
            extract_traces_core(subject,
                                FOV_dir,
                                session,
                                setup,
                                overwrite,
                                cell_masks,
                                neuropil_masks,
                                bpod_path,
                                photostim,
                                roi_type,
                               use_red_channel)
def extract_photostim_groups(subject,
                             FOV,
                             setup,
                             raw_movie_basedir,
                             suite2p_basedir,
                            overwrite = False):
    FOV_dir = os.path.join(suite2p_basedir,setup,subject,FOV)
    sessions=os.listdir(FOV_dir)  
    print('extractin photostim groups for {} - {}'.format(subject, FOV))
    for session in sessions:
        if 'z-stack' in session.lower() or '.' in session:
            continue
        if 'photostim' not in os.listdir(os.path.join(suite2p_basedir,setup,subject,FOV,session)):
            print('no photostim found for {}'.format(session))
            continue
        extract_photostim_groups_core(subject, 
                                    FOV,
                                    session,
                                    setup,
                                    raw_movie_basedir,
                                    suite2p_basedir,
                                     overwrite)
    
def extract_photostim_groups_core(subject, #TODO write more explanation and make this script nicer
                             FOV,
                             session,
                             setup,
                             raw_movie_basedir,
                             suite2p_basedir,
                             overwrite=False):
    """
    Extracts photostim coordinates from scanimage header.
    Rotates it and does rigid and non-rigid registration on them to match mean image.
    Finds where the photostim groups happen.
    Calculates null-distribution of photostimmed amplitudes.
    Creates a plot and saves data in a dictionary.

    Parameters
    ----------
    subject : TYPE
        DESCRIPTION.
    FOV : TYPE
        DESCRIPTION.
    session : TYPE
        DESCRIPTION.
    setup : TYPE
        DESCRIPTION.
    raw_movie_basedir : TYPE
        DESCRIPTION.
    suite2p_basedir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

# =============================================================================
#     raw_movie_basedir = '/mnt/Data/Calcium_imaging/raw/KayvonScope/'
#     suite2p_basedir = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p'
#     subject = 'BCI_35'
#     FOV = 'FOV_06'
#     session = '080422'
#     setup = 'Bergamo-2P-Photostim'   
# =============================================================================
    FOV_dir = os.path.join(suite2p_basedir,setup,subject,FOV)
    
    
    if 'photostim_groups.npy' in os.listdir(os.path.join(FOV_dir,session,'photostim')) and not overwrite:
        print('{} already exported'.format(session))
        return
        
    use_all_ROIs = False
    step_back = 10
    step_forward = 20
    baseline_length = step_back# from -step_back
    peak_length = 10 #from peak_offset
    peak_offset = 3##from trial start - gap
    
    use_overlap = False
    min_overlap = .1
    max_direct_distance = 30
    part_num= 5
    false_positive_rate = 1 #per cent
    
    
    
    ops =  np.load(os.path.join(FOV_dir,session,'photostim','ops.npy'),allow_pickle = True).tolist()
    meanimg_dict = np.load(os.path.join(FOV_dir,session,'mean_image.npy'),allow_pickle = True).tolist()
    if 'rotation_matrix' in meanimg_dict.keys():
        angle = -1*np.mean(np.asarray([np.arccos(meanimg_dict['rotation_matrix'][0,0]),
                                    np.arcsin(meanimg_dict['rotation_matrix'][1,0]),
                                    -1*np.arcsin(meanimg_dict['rotation_matrix'][0,1]),
                                    np.arccos(meanimg_dict['rotation_matrix'][1,1])]))
    else:
        angle = 0

    
    
    F = np.load(os.path.join(FOV_dir,session,'photostim','F.npy'))
    Fneu = np.load(os.path.join(FOV_dir,session,'photostim','Fneu.npy'))
    F0 = np.load(os.path.join(FOV_dir,session,'photostim','F0.npy'))
    
    
    

    
    stat =  np.load(os.path.join(FOV_dir,'stat.npy'),allow_pickle = True).tolist()
    stat_rest =  np.load(os.path.join(FOV_dir,'stat_rest.npy'),allow_pickle = True).tolist()
    
    if use_all_ROIs:
        F_rest = np.load(os.path.join(FOV_dir,session,'photostim','F_rest.npy'))
        Fneu_rest = np.load(os.path.join(FOV_dir,session,'photostim','Fneu_rest.npy'))
        F0_rest = np.load(os.path.join(FOV_dir,session,'photostim','F0_rest.npy'))
   
        
        F = np.concatenate([F,F_rest],0)
        Fneu = np.concatenate([Fneu,Fneu_rest],0)
        F0 = np.concatenate([F0,F0_rest],0)
        stat = np.concatenate([stat,stat_rest])
    
    F,Fneu = remove_stim_artefacts(F,Fneu,ops['frames_per_file'])
    photostim_indices = np.concatenate([[0],np.cumsum(ops['frames_per_file'])])[:-1]
    #%
    F0_scalar = np.nanmedian(F0,1)
    DFF = (F-F0_scalar[:,np.newaxis])/F0_scalar[:,np.newaxis]
    DFF_aligned = align_trace_to_event(DFF,
                             photostim_indices,
                             step_back,
                             step_forward)
    mask = np.zeros_like(ops['meanImg'])
    for i,roi_stat in enumerate(stat):
        stat[i]['med'] = [np.median(stat[i]['ypix'][stat[i]['soma_crop']]),
                          np.median(stat[i]['xpix'][stat[i]['soma_crop']])]
        mask[roi_stat['ypix'],roi_stat['xpix']] = 1
        
    for i,roi_stat in enumerate(stat_rest):
        stat_rest[i]['med'] = [np.median(stat_rest[i]['ypix'][stat_rest[i]['soma_crop']]),
                               np.median(stat_rest[i]['xpix'][stat_rest[i]['soma_crop']])]
        mask[roi_stat['ypix'],roi_stat['xpix']] = .5
       #%
        
    x_offset = np.median(ops['xoff'])
    y_offset  =np.median(ops['yoff'])
    raw_movie_directory = os.path.join(raw_movie_basedir,setup,subject,session)
    photostim_files_dict = utils_io.organize_photostim_files(raw_movie_directory)
    # we are assuming that the photostim group IDs stay the same even if there are multiple basenames
    
    photostim_dict = {}
    for base_name,metadata in zip(photostim_files_dict['base_names'],photostim_files_dict['base_metadata']):
        photostim_groups = metadata['metadata']['json']['RoiGroups']['photostimRoiGroups']
        photostim_order = np.asarray(metadata['metadata']['hPhotostim']['sequenceSelectedStimuli'].strip('[]').split(' ')*10,int) - 1 # python indexing # repeating the stuff 10 times
        fovdeg = list()
        for s in metadata['metadata']['hRoiManager']['imagingFovDeg'].strip('[]').split(' '): fovdeg.extend(s.split(';'))
        fovdeg = np.asarray(fovdeg,float)
        fovdeg = [np.min(fovdeg),np.max(fovdeg)]
        
        Lx = int(metadata['metadata']['hRoiManager']['pixelsPerLine'])
        Ly = int(metadata['metadata']['hRoiManager']['linesPerFrame'])
        yup,xup = upsample_block_shifts(Lx, Ly, ops['nblocks'], ops['xblock'], ops['yblock'], np.median(ops['yoff1'][:5000,:],0)[np.newaxis,:], np.median(ops['xoff1'][:5000,:],0)[np.newaxis,:])
        xup=xup.squeeze()+x_offset 
        yup=yup.squeeze()+y_offset 
        group_list = []
        for group_i,photostim_group in enumerate(photostim_groups):
            power = photostim_group['rois'][1]['scanfields']['powers']
            coordinates = photostim_group['rois'][1]['scanfields']['slmPattern']
            xy = np.asarray(coordinates)[:,:2] + photostim_group['rois'][1]['scanfields']['centerXY']
            centerXY_list = []
            sizeXY_list = []
            revolutions_list = []
            
            # calculate coordinates for galvo first
            xy_now = photostim_group['rois'][1]['scanfields']['centerXY']
            px = xy_now[0]
            py = xy_now[1]
            ox = oy = 0
            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            coordinates_now = (np.asarray([qx,qy])-fovdeg[0])/np.diff(fovdeg)

            coordinates_now = coordinates_now[::-1] # go to yx
            coordinates_now[0] = coordinates_now[0]*Ly
            coordinates_now[1] = coordinates_now[1]*Lx

            yoff_now = yup[int(coordinates_now[0]),int(coordinates_now[1])]
            xoff_now = xup[int(coordinates_now[0]),int(coordinates_now[1])]

            #lt.plot(coordinates_now[1],coordinates_now[0],'ro')        
            coordinates_now[0]-=yoff_now
            coordinates_now[1]-=xoff_now
            #plt.plot(coordinates_now[1],coordinates_now[0],'yo')

            galvo_xy = coordinates_now[::-1] # go back to xy

            
            
            for xy_now in xy:
                px = xy_now[0]
                py = xy_now[1]
                ox = oy = 0
                qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
                qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
                coordinates_now = (np.asarray([qx,qy])-fovdeg[0])/np.diff(fovdeg)
                
                coordinates_now = coordinates_now[::-1] # go to yx
                coordinates_now[0] = coordinates_now[0]*Ly
                coordinates_now[1] = coordinates_now[1]*Lx

                yoff_now = yup[int(coordinates_now[0]),int(coordinates_now[1])]
                xoff_now = xup[int(coordinates_now[0]),int(coordinates_now[1])]
                
                #lt.plot(coordinates_now[1],coordinates_now[0],'ro')        
                coordinates_now[0]-=yoff_now
                coordinates_now[1]-=xoff_now
                #plt.plot(coordinates_now[1],coordinates_now[0],'yo')

                centerXY_list.append(coordinates_now[::-1]) # go back to xy
                sizeXY_list.append((np.asarray(photostim_group['rois'][1]['scanfields']['sizeXY']))/np.diff(fovdeg)*np.asarray([Lx,Ly]))
                revolutions_list.append(photostim_group['rois'][1]['scanfields']['stimparams'][1])
                group_metadata = {'centerXY':np.asarray(centerXY_list),
                                  'sizeXY':np.asarray(sizeXY_list),
                                  'revolution':np.asarray(revolutions_list),
                                  'power':power,
                                  'galvo_centerXY':galvo_xy,
                                  'power_slm':np.asarray(coordinates)[:,3],
                                  'z':np.asarray(coordinates)[:,2]}
            group_list.append(group_metadata)
        photostim_dict[base_name] = {'photostim_order':photostim_order,
                                      'group_list':group_list,
                                      'base_name':base_name}


    offset_list = np.arange(-15,5)
    direct_amplitude_list = []    
    print('calculating offset')
    for offset in offset_list:
        
        for key in photostim_dict.keys():
            photostim_dict[key]['counter'] = offset
        photostim_group_list = []
        for file in ops['tiff_list']:
            file = file.split('/')[-1]
            
            prefix = photostim_files_dict['basename_order'][np.where(photostim_files_dict['file_order']==file)[0][0]]
            
            photostim_dict[prefix]['counter']+=1
            #print([prefix,file,photostim_dict[prefix]['counter']])
            photostim_group = photostim_dict[prefix]['photostim_order'][[np.max([0,photostim_dict[prefix]['counter']])]][0]
            photostim_group_list.append(photostim_group)
        
        amplitudes_list = []
        for group_idx in range(len(group_list)):
            DFF_now = DFF_aligned[:,:,np.asarray(photostim_group_list)==group_idx]
            DFF_averaged = np.nanmean(DFF_now,2)
            
            DFF_averaged_normalized = DFF_averaged-np.nanmean(DFF_averaged[:baseline_length,:],0)
            amplitude = np.nanmean(DFF_averaged_normalized[step_back+peak_offset:step_back+peak_offset+peak_length,:],0)

            amplitude[np.isnan(amplitude)]=0
            amplitude_order = np.argsort(amplitude)[::-1]
            distances=  []
            for s,a in zip(stat,amplitude):
                d_now = np.sqrt((group_list[group_idx]['centerXY'][:,0]-s['med'][1])**2 + (group_list[group_idx]['centerXY'][:,1] -s['med'][0])**2)
                distances.append(np.min(d_now))
            amplitudes_list.extend(amplitude[np.asarray(distances)<20])
        direct_amplitude_list.append(np.nanmean(amplitudes_list))
    
    
    #%
    offset = offset_list[np.argmax(direct_amplitude_list)]
    for key in photostim_dict.keys():
        photostim_dict[key]['counter'] = offset
    photostim_group_list = []
    for file in ops['tiff_list']:
        file = file.split('/')[-1]
        prefix = photostim_files_dict['basename_order'][np.where(photostim_files_dict['file_order']==file)[0][0]]
        photostim_dict[prefix]['counter']+=1
        photostim_group = photostim_dict[prefix]['photostim_order'][[np.max([0,photostim_dict[prefix]['counter']])]][0]
        photostim_group_list.append(photostim_group)
        

#%
    #%% calculate noise level for all the cells
    
    photostim_repeats = []
    for group_idx in range(len(group_list)):
        photostim_repeats.append(sum(np.asarray(photostim_group_list) == group_idx))
    unique_photostim_repeats = np.unique(photostim_repeats)
    print('generating null distributions')
    for cell_index in range(len(stat)):
        
        dff = DFF[cell_index,:].copy()
        s = stat[cell_index]
        distances = []
        for group_idx in range(len(group_list)):
            d_now = np.sqrt((group_list[group_idx]['centerXY'][:,0]-s['med'][1])**2 + (group_list[group_idx]['centerXY'][:,1] -s['med'][0])**2)
            distances.append(np.min(d_now))
        involving_groups = np.where(np.asarray(distances)<max_direct_distance)[0]
        direct_stim_indices = []
        for g in involving_groups:
            direct_stim_indices.extend(photostim_indices[np.asarray(photostim_group_list)==g])
        direct_stim_indices = np.unique(direct_stim_indices)   
        
        for i in direct_stim_indices:
            dff[np.max([0,i-1]):i+10] = np.nan

        for trial_i,trial_num in enumerate(unique_photostim_repeats):
            #print(trial_num)
            amplitude_all = []
            for repeat in range(100):
                #stim_indices = np.asarray(np.random.uniform(step_back,DFF.shape[1]-step_forward-peak_offset,trial_num),int)
                amplitudes = []
                while len(amplitudes)<trial_num:
                    idx = int(np.random.uniform(step_back,DFF.shape[1]-step_forward-peak_offset))
                    amplitude = np.mean(dff[idx+peak_offset:idx+peak_offset+step_forward])-np.mean(dff[idx-step_back:idx])
                    if np.isnan(amplitude) == False:
                        amplitudes.append(amplitude)
                amplitude_all.append(np.nanmean(amplitudes))
                
            for group_idx in np.where(np.asarray(photostim_repeats) == trial_num)[0]:
                if cell_index == 0:
                    group_list[group_idx]['cell_response_distribution'] = [np.sort(amplitude_all)]
                else:
                    group_list[group_idx]['cell_response_distribution'].append(np.sort(amplitude_all))
        
        
    photostim_dict_out = {'group_order': photostim_group_list,
                      'groups':group_list,
                      'group_repeats':photostim_repeats,
                      'photostim_indices':np.concatenate([[0],np.cumsum(ops['frames_per_file'])])[:-1]}
    
    
    
    
    #%%
    
    photostim_group_list = photostim_dict_out['group_order'].copy()
    group_list = photostim_dict_out['groups'].copy()
    significant_distances_all = []
    significant_amplitudes_all = []
    nonsignificant_distances_all = []
    nonsignificant_amplitudes_all = []
    significant_cells_per_groups = []
    nonsignificant_cells_per_group = []
    for group_idx in range(len(group_list)): 
        DFF_now = DFF_aligned[:,:,np.asarray(photostim_group_list)==group_idx]
        DFF_averaged = np.nanmean(DFF_now,2)

        DFF_averaged_normalized = DFF_averaged-np.nanmean(DFF_averaged[:baseline_length,:],0)
        amplitude = np.nanmean(DFF_averaged_normalized[step_back+peak_offset:step_back+peak_offset+peak_length,:],0)
        distances=  []
        for s,a in zip(stat,amplitude):
            d_now = np.sqrt((group_list[group_idx]['centerXY'][:,0]-s['med'][1])**2 + (group_list[group_idx]['centerXY'][:,1] -s['med'][0])**2)
            distances.append(np.min(d_now))
        direct_cell_indices = np.asarray(distances)<max_direct_distance
        significant_direct_cells = []
        nonsignificant_direct_cells = []

        for idx in np.where(direct_cell_indices)[0]:
            null_distribution = photostim_dict_out['groups'][group_idx]['cell_response_distribution'][idx]
            if amplitude[idx]>np.percentile(null_distribution,100-false_positive_rate):
                significant_direct_cells.append(idx)
            else:
                nonsignificant_direct_cells.append(idx)
        group_list[group_idx]['photostimmed_cells'] = significant_direct_cells
        significant_distances_all.append(np.asarray(distances)[significant_direct_cells])
        nonsignificant_distances_all.append(np.asarray(distances)[nonsignificant_direct_cells])
        significant_amplitudes_all.append(np.asarray(amplitude)[significant_direct_cells])
        nonsignificant_amplitudes_all.append(np.asarray(amplitude)[nonsignificant_direct_cells])
        significant_cells_per_groups.append(len(significant_direct_cells))
        nonsignificant_cells_per_group.append(len(nonsignificant_direct_cells))
            

    
    #overlaps
    if use_overlap:
        from scipy.ndimage import gaussian_filter

        photostim_mask_list = []
        grid = np.asarray(np.meshgrid(np.arange(ops['meanImg'].shape[0]),np.arange(ops['meanImg'].shape[1])))
        for group_idx in range(len(group_list)):
            photostim_mask = np.zeros(ops['meanImg'].shape)
            for center,size in zip(group_list[group_idx]['centerXY'],group_list[group_idx]['sizeXY']):
                photostim_mask[np.sqrt((grid[1,:,:]-center[1])**2 + (grid[0,:,:]-center[0])**2)<size[0]/2] =1
            photostim_mask_list.append(photostim_mask)
        overlaps = np.zeros([len(stat),len(group_list)])*np.nan
        for cell_i,s in enumerate(stat):
            print(cell_i/len(stat))
            cell_mask = np.zeros(ops['meanImg'].shape)
            cell_mask[s['ypix'],s['xpix']]=1
            cell_mask = gaussian_filter(cell_mask, 2)
            for group_idx in range(len(group_list)):
                photostim_mask = photostim_mask_list[group_idx].copy()
                photostim_mask+=cell_mask
                #photostim_overlap.append()
                overlaps[cell_i,group_idx]=np.sum(photostim_mask>1)/(size[0]*size[1])
           # asdasd
    start_indices = np.asarray(np.arange(part_num)*len(photostim_group_list)/part_num,int)
    end_indices = np.asarray((np.arange(part_num)+1)*len(photostim_group_list)/part_num,int)
    direct_amplitude_list_stability = []
    direct_amplitude_list_stability_per_group = []
    direct_amplitude_list_stability_per_cell = np.nan*np.ones([len(stat),len(group_list),part_num])## neurons X photostim groups * parts
    for part_i,(s,e) in enumerate(zip(start_indices,end_indices)):
        photostim_group_list_ = np.asarray(photostim_group_list)[s:e]
        DFF_aligned_ = DFF_aligned[:,:,s:e]
        amplitudes_list = []
        amplitudes_list_per_group = []
        for group_idx in range(len(group_list)):
            DFF_now = DFF_aligned_[:,:,np.asarray(photostim_group_list_)==group_idx]
            DFF_averaged = np.nanmean(DFF_now,2)

            DFF_averaged_normalized = DFF_averaged-np.nanmean(DFF_averaged[:baseline_length,:],0)
            amplitude = np.nanmean(DFF_averaged_normalized[step_back+peak_offset:step_back+peak_offset+peak_length,:],0)
            
            amplitude[np.isnan(amplitude)]=0
            #amplitude_order = np.argsort(amplitude)[::-1]
# =============================================================================
#             distances=  []
#             for s,a in zip(stat,amplitude):
#                 d_now = np.sqrt((group_list[group_idx]['centerXY'][:,0]-s['med'][1])**2 + (group_list[group_idx]['centerXY'][:,1] -s['med'][0])**2)
#                 distances.append(np.min(d_now))
#             if use_overlap:
#                 direct_cell_indices = (overlaps[:,group_idx]).squeeze()>min_overlap
#             else:
#                 direct_cell_indices = np.asarray(distances)<min_direct_distance
# =============================================================================
            direct_cell_indices = group_list[group_idx]['photostimmed_cells']
            
            direct_amplitude_list_stability_per_cell[direct_cell_indices,group_idx,part_i]=amplitude[direct_cell_indices]
            amplitudes_list.extend(amplitude[direct_cell_indices])
            amplitudes_list_per_group.append(np.nanmean(amplitude[direct_cell_indices]))
        direct_amplitude_list_stability.append(np.nanmean(amplitudes_list))
        direct_amplitude_list_stability_per_group.append(amplitudes_list_per_group)
      
    
    #%%
    fig = plt.figure(figsize = [15,10])
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,4)
    ax3 = fig.add_subplot(2,3,5)
    ax4 = fig.add_subplot(2,3,6)
    
    ax5 = fig.add_subplot(2,3,2)
    ax6 = fig.add_subplot(2,3,3)
    
    
    ax1.plot(offset_list,direct_amplitude_list)
    ax1.set_xlabel('offset in order')
    ax1.set_ylabel('average directly stimulated amplitude')
    
    
    ax2.plot(np.asarray(direct_amplitude_list_stability_per_group))
    ax2.plot(direct_amplitude_list_stability,'k-',linewidth = 4)
    ax2.set_xlabel('photostim progress')
    ax2.set_ylabel('direct photostimamplitude per group')
    ax3.plot(np.nanmean(direct_amplitude_list_stability_per_cell,1).squeeze().T)
    ax3.plot(np.nanmean(np.nanmean(direct_amplitude_list_stability_per_cell,1),0).squeeze().T,'k-',linewidth = 4)
    ax3.set_xlabel('photostim progress')
    ax3.set_ylabel('direct photostimamplitude per cell')
    a = np.nanmean(direct_amplitude_list_stability_per_cell,1).squeeze()
    #ax4.plot(a[:,0],a[:,-1],'ko')
    ax4.plot(direct_amplitude_list_stability_per_cell[:,:,0].flatten(),direct_amplitude_list_stability_per_cell[:,:,-1].flatten(),'k.',alpha = .2)
    needed = (np.isnan(a[:,0]) == False) &(a[:,0]>.5)
    p = np.polyfit(a[needed,0],a[needed,-1],1)
    ax4.plot([0,2],np.polyval(p,[0,2]),'r-',label = 'y={:.2f}*x+{:.2f}'.format(p[0],p[1]))
    ax4.set_xlabel('direct amplitude 1st part')
    ax4.set_ylabel('direct amplitude last part')
    ax4.legend()
    ax5.plot(np.concatenate(nonsignificant_distances_all),np.concatenate(nonsignificant_amplitudes_all),'k.',alpha = .5)
    ax5.plot(np.concatenate(significant_distances_all),np.concatenate(significant_amplitudes_all),'r.',alpha = .5)
    
    ax5.set_xlabel('distance from closest photostim')
    ax5.set_ylabel('mean ampitude (dF/F)')
    
    ax6.hist(nonsignificant_cells_per_group,10,color = 'black',alpha = .5,label = 'no significant excitation')
    ax6.hist(significant_cells_per_groups,10,color = 'red',alpha = .5,label = 'significant excitation')
    ax6.set_xlabel('# of directly photostimmed cells in group')
    ax6.set_ylabel('# of groups')
    ax6.legend()
    #ax6.plot(nonsignificant_cells_per_group,significant_cells_per_groups,'ko')
    np.save(os.path.join(FOV_dir,session,'photostim','photostim_groups.npy'),photostim_dict_out,allow_pickle = True)
    
    fig.savefig(os.path.join(FOV_dir,session,'photostim','direct_photostim.pdf'), format="pdf")
    plt.close()