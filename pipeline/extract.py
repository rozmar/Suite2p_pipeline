import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
import numpy as np
from suite2p.extraction.extract import extract_traces_from_masks

    
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
    
def extract_traces(local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/',
                   metadata_dir = '/mnt/Data/BCI_metadata/',
                   raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/',
                   suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/',
                   bpod_path = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Behavior/BCI_exported/',
                   subject = 'BCI_26',
                   setup = 'Bergamo-2P-Photostim',
                   fov = 'FOV_06',
                   overwrite = True):
    FOV_dir = os.path.join(suite2p_dir_base,setup,subject,fov)
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
            Fneu = np.load(os.path.join(FOV_dir,session,'Fneu.npy'))
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
        else:
            F0 = np.load(os.path.join(FOV_dir,session,'F0.npy'))
            Fvar = np.load(os.path.join(FOV_dir,session,'Fvar.npy'))
        
        
        if 'neuropil_contribution.npy' not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
            
    
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
            np.save(os.path.join(FOV_dir,session,'neuropil_contribution.npy'), neuropil_dict,allow_pickle=True)
        else:
            neuropil_dict = np.load(os.path.join(FOV_dir,session,'neuropil_contribution.npy'),allow_pickle=True).tolist()
          
        
        if 'photon_counts.npy' not in os.listdir(os.path.join(FOV_dir,session)) or overwrite:
            plot_stuff = True
            stat = np.load(os.path.join(FOV_dir,'stat.npy'), allow_pickle = True).tolist()
            photon_counts_dict = {}
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
                fig.savefig(os.path.join(FOV_dir,session,'photon_counts.pdf'), format="pdf")
            np.save(os.path.join(FOV_dir,session,'photon_counts.npy'), photon_counts_dict,allow_pickle=True)
        else:
            photon_counts_dict = np.load(os.path.join(FOV_dir,session,'photon_counts.npy'),allow_pickle=True).tolist()
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
