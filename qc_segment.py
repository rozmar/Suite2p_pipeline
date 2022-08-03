#%% generate binned movies
import os, json
import scipy
import numpy as np
from pathlib import Path
import  matplotlib.pyplot as plt
from suite2p.detection.detect import detect
from suite2p.extraction.masks import create_masks
#from suite2p.extraction.extract import extract_traces_from_masks
from suite2p import registration
import cv2
import tifffile
version = '1.0'

def correlate_z_stacks(FOV_dir):
#%%
    try:
        binned_movies_dict = np.load(os.path.join(FOV_dir,'session_mean_images.npy'),allow_pickle = True).tolist()
    except:
        binned_movies_dict = {}
    try:
        z_stack_corr_dict = np.load(os.path.join(FOV_dir,'z_stack_correlations.npy'),allow_pickle = True).tolist()
    except:
        z_stack_corr_dict = {}
    
    sessions_ = os.listdir(FOV_dir)
    sessions = []
    for session in sessions_:
        if 'z-stack' in session.lower() or '.' in session:
            continue
        sessions.append(session)
    
    for session in sessions:
        if session not in binned_movies_dict.keys():
            mov = np.load(os.path.join(FOV_dir,session,'binned_movie.npy'))
            #%
            binned_movies_dict[session] = {'meanImg':np.mean(mov,0),
                                           'maxImg':np.max(mov,0),
                                           'stdImg':np.std(mov,0)}
    np.save(os.path.join(FOV_dir,'session_mean_images.npy'),binned_movies_dict)
    ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()     
    
    
    
    #%
    zstack_names = os.listdir(os.path.join(FOV_dir,'Z-stacks'))
    stacks_dict = {}
    for zstack_name in np.sort(zstack_names):
        if '.tif' in zstack_name:
            stack = tifffile.imread(os.path.join(FOV_dir,'Z-stacks',zstack_name))
            stacks_dict[zstack_name[:-4]]= stack    
    
    #%
    ops['maxregshift'] = .3
    session_zcorrs = []
    stack_zcorrs = []
    for stack in stacks_dict.keys():
        if stack not in z_stack_corr_dict.keys():
            #%
            z_stack_corr_dict[stack] = {'stacks':{},
                                        'sessions':{}}
            #%
        zcorr_list = []
        #%
        for session in sessions:
            
            if session not in z_stack_corr_dict[stack]['sessions'].keys():
                ops_orig, zcorr_orig = registration.zalign.compute_zpos_single_frame(stacks_dict[stack], binned_movies_dict[session]['meanImg'][np.newaxis,:,:], ops)
                zcorr_list.append(zcorr_orig)
                z_stack_corr_dict[stack]['sessions'][session] = zcorr_orig
            else:
                zcorr_list.append(z_stack_corr_dict[stack]['sessions'][session])
#%
        session_zcorrs.append(np.asarray(zcorr_list).squeeze())
        stack_zcorr_now = []
        for stack_ in stacks_dict.keys():
            if stack_ not in z_stack_corr_dict[stack]['stacks'].keys():
                ops_orig, zcorr_orig = registration.zalign.compute_zpos_single_frame(stacks_dict[stack], stacks_dict[stack_], ops)
                z_stack_corr_dict[stack]['stacks'][stack_] = zcorr_orig
            else:
                zcorr_orig =z_stack_corr_dict[stack]['stacks'][stack_]
            stack_zcorr_now.append(zcorr_orig)
        stack_zcorrs.append(stack_zcorr_now)
    #% calculate offsets

    #stack_shift_list = []
    for zstack_1 in z_stack_corr_dict.keys():
        print(zstack_1)
        if 'stack_offsets' not in z_stack_corr_dict[zstack_1].keys():
            z_stack_corr_dict[zstack_1]['stacks_offsets'] = {}
            z_stack_corr_dict[zstack_1]['stacks_loss'] = {}
            z_stack_corr_dict[zstack_1]['stacks_sigma'] = {}
        for zstack_2 in z_stack_corr_dict.keys():
            if zstack_2 in z_stack_corr_dict[zstack_1]['stacks_offsets'].keys():
                continue
            matrix = z_stack_corr_dict[zstack_1]['stacks'][zstack_2]
            max_zcorr_vals = np.max(matrix,0)
            min_zcorr_vals = np.min(matrix,0)
            matrix = (matrix - min_zcorr_vals[np.newaxis,:])/(max_zcorr_vals-min_zcorr_vals)[np.newaxis,:]
            f = scipy.interpolate.interp2d(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]), matrix, kind='linear')
            matrix = f(np.arange(0,matrix.shape[1],.1), np.arange(0,matrix.shape[0],.1))
            matrix = matrix/np.sum(matrix)
           # asasda
           #%%
            offset_list = []
            loss_list = []
            sigma_list = [0,10,20,30,50,100,150,200,300]
            for sigma in sigma_list:
                
                reference = np.eye(matrix.shape[0])
                max_zcorr_vals = np.max(reference,0)
                min_zcorr_vals = np.min(reference,0)
                reference = (reference - min_zcorr_vals[np.newaxis,:])/(max_zcorr_vals-min_zcorr_vals)[np.newaxis,:]
                reference =  reference/np.sum(reference)
                reference = scipy.ndimage.gaussian_filter(reference, sigma)
                shift_list = []
                loss = []
                for shift in range(int(-1*matrix.shape[0]/2),int(matrix.shape[0]/2)):
                    shift_list.append(shift/10)
                    d = np.roll(matrix,shift) - reference
                    d[d<0]=0
                    loss.append(np.sum(np.abs(d)))
                    #asdas
                offset_list.append(shift_list[np.argmin(loss)])
                loss_list.append(np.min(loss))
            idx = np.argmin(loss_list)
            z_stack_corr_dict[zstack_1]['stacks_offsets'][zstack_2]=offset_list[idx]
            z_stack_corr_dict[zstack_1]['stacks_loss'][zstack_2]=loss_list[idx]
            z_stack_corr_dict[zstack_1]['stacks_sigma'][zstack_2]=sigma_list[idx]/10
            #%%
    np.save(os.path.join(FOV_dir,'z_stack_correlations.npy'),z_stack_corr_dict)

        #%
    import matplotlib
    font = { 'size'   : 6}
        
    matplotlib.rc('font', **font)
    sessions_real = []       
    for i,session in enumerate(sessions):
        if 'z-stack' in session.lower() or '.' in session:
            continue
        sessions_real.append(session)
    fig = plt.figure(figsize = [20,20])
    for i,(stack,zcorr,stack_zcorr) in enumerate(zip(stacks_dict.keys(),session_zcorrs,stack_zcorrs)): 
        zcorr = zcorr.T
        ax_now = fig.add_subplot(len(session_zcorrs),len(stack_zcorr)+1,(i)*(len(stack_zcorr)+1)+1)
        max_zcorr_vals = np.max(zcorr,0)
        min_zcorr_vals = np.min(zcorr,0)
        try:
            zcorr_norm = (zcorr - min_zcorr_vals[np.newaxis,:])/(max_zcorr_vals-min_zcorr_vals)[np.newaxis,:]
        except: #single session
            zcorr_norm = ((zcorr - min_zcorr_vals)/(max_zcorr_vals-min_zcorr_vals))[np.newaxis,:].T
        ax_now.imshow(zcorr_norm,aspect = 'auto')
        ax_now.set_ylabel(stack)
        
        #
        ax_now.set_xticks(np.arange(zcorr_norm.shape[1]))
        ax_now.set_xticklabels(sessions_real)
        for i_zstack,(stack_image,stack_2_name) in enumerate(zip(stack_zcorr,stacks_dict.keys())):
            ax_now = fig.add_subplot(len(session_zcorrs),len(stack_zcorr)+1,(i)*(len(stack_zcorr)+1)+i_zstack+2)
            max_zcorr_vals = np.max(stack_image,0)
            min_zcorr_vals = np.min(stack_image,0)
            stack_image_norm = (stack_image - min_zcorr_vals[np.newaxis,:])/(max_zcorr_vals-min_zcorr_vals)[np.newaxis,:]
            ax_now.imshow(stack_image_norm,aspect = 'auto')
            ax_now.set_xlabel(stack_2_name)
            ax_now.set_ylabel(stack)
            ax_now.axis('off')
            ax_now.set_title('{} offset {:.2f} loss {} sigma'.format(z_stack_corr_dict[stack]['stacks_offsets'][stack_2_name],
                                                              round(z_stack_corr_dict[stack]['stacks_loss'][stack_2_name],1),
                                                              z_stack_corr_dict[stack]['stacks_sigma'][stack_2_name]))
            #%%
    fig.savefig(os.path.join(FOV_dir,'Z-positions.pdf'), format="pdf")

def qc_segment(local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/',
               metadata_dir = '/mnt/Data/BCI_metadata/',
               raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/',
               suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/',
               subject = 'BCI_29',
               setup = 'Bergamo-2P-Photostim',
               fov = 'FOV_03',
               minimum_contrast = 3,
               acceptable_z_range = 3,
               segment_cells = False,
               correlte_z_stacks = False):
    # TODO these variables are hard-coded
   
    use_cellpose = False
    photostim_name_list = ['slm','stim','file','photo']
    blacklist_for_binned_movie = {'BCI_26':['060622']}
    # TODO these variables are hard-coded
    
    FOV_dir = os.path.join(suite2p_dir_base,setup,subject,fov)
    temp_FOV_dir = os.path.join(local_temp_dir,'{}_{}_{}'.format(setup,subject,fov))
    Path(temp_FOV_dir).mkdir(parents = True, exist_ok =True)
    sessions=os.listdir(FOV_dir)
    binned_movie_concatenated = []
    zcorr_list_concatenated = []
    session_zoffset_list_concatenated = []
    xoff_mean_list_concatenated = []
    yoff_mean_list_concatenated = []
    xoff_std_list_concatenated = []
    yoff_std_list_concatenated = []
    mean_intensity_list = []
    trial_i = 0
    new_session_idx =[]
    new_session_names = []
    median_z_values = []
    xoff_list = []
    yoff_list = []
    session_data_dict = {}
    reference_image_dict = {}
    
    correlate_z_stacks(FOV_dir) # calculate Z-stack correlations
    z_stack_corr_dict = np.load(os.path.join(FOV_dir,'z_stack_correlations.npy'),allow_pickle = True).tolist()
    z_offset = 0
    reference_z_stack_name= None
    for session in sessions:
        if 'z-stack' in session.lower() or '.' in session:
            continue
        print(session)
        
        with open(os.path.join(FOV_dir,session,'s2p_params.json')) as f:
            s2p_params = json.load(f)
        
        z_stack_name = s2p_params['z_stack_name']
        if reference_z_stack_name is not None:
            z_offset -= z_stack_corr_dict[z_stack_name[:-4]]['stacks_offsets'][reference_z_stack_name[:-4]]
        reference_z_stack_name = z_stack_name
            
        new_session_idx.append(trial_i)
        new_session_names.append(session)
        with open(os.path.join(FOV_dir,session,'filelist.json')) as f:
            filelist_dict = json.load(f)
        ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
        
        
        concatenated_movie_filelist_json = os.path.join(FOV_dir,session,'filelist.json')
        with open(concatenated_movie_filelist_json, "r") as read_file:
            filelist_dict = json.load(read_file)
        meanimg_dict = np.load(os.path.join(FOV_dir,session,'mean_image.npy'),allow_pickle = True).tolist()
        reference_image_dict[session] = meanimg_dict
        xoff_mean_now = []
        yoff_mean_now = []
        xoff_std_now = []
        yoff_std_now = []
        xoff_now = []
        yoff_now = []
        zoff_now = []
        framenum_so_far = 0
        mean_int_now = []
        if 'mean_instensity' not in filelist_dict.keys():
            filelist_dict['mean_instensity'] = filelist_dict['zoff_list'].copy()
        for framenum, filename,z,intensity in zip(filelist_dict['frame_num_list'],filelist_dict['file_name_list'],filelist_dict['zoff_list'],filelist_dict['mean_instensity']):
            bad_trial = False
            for photo_name in photostim_name_list:
                if photo_name in filename.lower():
                    bad_trial=True
            if not bad_trial:
                xoff = ops['xoff_list'][framenum_so_far:framenum_so_far+framenum]
                yoff = ops['yoff_list'][framenum_so_far:framenum_so_far+framenum]
                
                xoff_mean_now.append(np.mean(xoff))
                yoff_mean_now.append(np.mean(yoff))
                xoff_std_now.append(np.std(xoff))
                yoff_std_now.append(np.std(yoff))
                zoff_now.append(z)
                
                xoff_list.append(xoff)
                yoff_list.append(yoff)
                
                xoff_now.append(xoff)
                yoff_now.append(yoff)
                
                mean_int_now.append(intensity)
                
                if np.std(xoff)>5:
                    print(filename)
            framenum_so_far += framenum
        session_data_dict[session] = {'xoff':xoff_now,
                                      'yoff':yoff_now,
                                      'zoff':zoff_now,
                                      'session_z_offfset':z_offset}
        
        xoff_mean_list_concatenated.extend(xoff_mean_now)
        yoff_mean_list_concatenated.extend(yoff_mean_now)
        xoff_std_list_concatenated.extend(xoff_std_now)
        yoff_std_list_concatenated.extend(yoff_std_now)
        zcorr_list_concatenated.extend(zoff_now)
        session_zoffset_list_concatenated.extend((np.zeros(len(zoff_now))+z_offset).tolist())
        mean_intensity_list.extend(mean_int_now)
        trial_i += len(xoff_mean_now)
        median_z_values.append(np.median(np.argmax(zoff_now,2).squeeze())+z_offset)
    new_session_idx.append(trial_i) # end of all trials
    xoff_mean_list_concatenated = np.asarray(xoff_mean_list_concatenated)        
    yoff_mean_list_concatenated = np.asarray(yoff_mean_list_concatenated)        
    xoff_std_list_concatenated = np.asarray(xoff_std_list_concatenated)        
    yoff_std_list_concatenated = np.asarray(yoff_std_list_concatenated)        
    session_zoffset_list_concatenated = np.asarray(session_zoffset_list_concatenated)    
    mean_intensity_list = np.asarray(mean_intensity_list)      
    try:# this is where the offsets can go btw
        zcorr_list_concatenated = np.concatenate(zcorr_list_concatenated).squeeze()
    except: # the z-stacks have different number of planes - 
        z_sizes = []
        for zcorr_now in zcorr_list_concatenated:
            #print(np.asarray(zcorr_now).shape)
            z_sizes.append(np.asarray(zcorr_now).shape[1])
        z_size_needed = np.min(z_sizes)
        #print(z_size_needed)
        zcorr_list_new = []
        for zcorr_now in zcorr_list_concatenated:
            if np.asarray(zcorr_now).shape[1]>z_size_needed:
                diff = int((np.asarray(zcorr_now).shape[1]-z_size_needed)/2)
                #print(diff)
                zcorr_now = np.asarray(zcorr_now)[:,diff:-diff]
            zcorr_list_new.append(zcorr_now)
        zcorr_list_concatenated = np.concatenate(zcorr_list_new).squeeze()
        
            
            
    max_zcorr_vals = np.max(zcorr_list_concatenated,1)
    min_zcorr_vals = np.min(zcorr_list_concatenated,1)
    contrast = max_zcorr_vals/min_zcorr_vals
    
    zcorr_list_concatenated_norm = (zcorr_list_concatenated - min_zcorr_vals[:,np.newaxis])/(max_zcorr_vals-min_zcorr_vals)[:,np.newaxis]
    hw = zcorr_list_concatenated_norm.shape[1]-np.argmax(zcorr_list_concatenated_norm[:,::-1]>.5,1) - np.argmax(zcorr_list_concatenated_norm>.5,1)
    z_with_hw = (zcorr_list_concatenated_norm.shape[1]-np.argmax(zcorr_list_concatenated_norm[:,::-1]>.5,1) + np.argmax(zcorr_list_concatenated_norm>.5,1))/2 + session_zoffset_list_concatenated
    
    if minimum_contrast is None:
        minimum_contrast = np.percentile(contrast,90)/2
    #%%
    
    imgs_all = []
    for session in reference_image_dict.keys():
        
        texted_image =cv2.putText(img=np.copy(reference_image_dict[session]['refImg']), text="{}".format(session), org=(20,40),fontFace=3, fontScale=1, color=(255,255,255), thickness=2)
        imgs_all.append(texted_image)
    tifffile.imsave(os.path.join(FOV_dir,'session_refImages.tiff'),imgs_all)
    
    #%% quality check -Z position
    median_z_value = np.median(median_z_values)
    
    fig = plt.figure(figsize = [20,10])
    ax_z = fig.add_subplot(4,1,1)
    ax_z.set_title('{} --- {}'.format(subject,fov))
    ax_xy = fig.add_subplot(4,1,3,sharex = ax_z)
    ax_zz = ax_xy.twinx()
    ax_contrast = fig.add_subplot(4,1,2,sharex = ax_z)
    ax_hw = ax_contrast.twinx()
    ax_mean_intensity = fig.add_subplot(4,1,4,sharex = ax_z)
    ax_mean_intensity.set_ylabel('Trial mean intensity (pixel value)')
    img = zcorr_list_concatenated_norm.T
    ax_z.imshow(img ,aspect='auto', alpha = 1,origin='lower',cmap = 'magma')
    
    x = np.arange(len(xoff_mean_list_concatenated))
    ax_xy.errorbar(x, xoff_mean_list_concatenated,yerr = xoff_std_list_concatenated,fmt = '-',label = 'X offset')
    ax_xy.errorbar(x, yoff_mean_list_concatenated,yerr = yoff_std_list_concatenated,fmt = '-',label = 'Y offset')
    
    ax_mean_intensity.plot(x,mean_intensity_list,'g-')
    
    ax_zz.plot(x, np.argmax(zcorr_list_concatenated.squeeze(),1)+session_zoffset_list_concatenated,'r-',label = 'Z offset')
    ax_zz.plot(x, z_with_hw,'y-',label = 'Z offset with halfwidth')
    ax_zz.plot([x[0],x[-1]],[median_z_value-acceptable_z_range]*2,'r--')
    ax_zz.plot([x[0],x[-1]],[median_z_value+acceptable_z_range]*2,'r--',label = 'Acceptable Z offsets for segmentation')
    
    
    ax_contrast.plot(contrast,'k-',label = 'Contrast in Z location')
    
    ax_contrast.plot([x[0],x[-1]],[minimum_contrast]*2,'r--')
    for idx_start,idx_end,session in zip(new_session_idx[:-1],new_session_idx[1:],new_session_names):
        ax_z.axvline(idx_start,color ='red')
        #ax_z.axvline(idx_end,color ='red')
        ax_z.text(np.mean([idx_start,idx_end]),ax_z.get_ylim()[0]+np.diff(ax_z.get_ylim())[0]/5*4,session,color = 'white',ha='center', va='center')
       
    #ax_xy.legend()#bbox_to_anchor = (1.1,1))
    ax_mean_intensity.set_xlabel('Total trial number')
    ax_xy.set_ylabel('XY offsets (pixels)')
    ax_xy.set_ylim([ax_xy.get_ylim()[0],np.diff(ax_xy.get_ylim())+ax_xy.get_ylim()[1]])
    ax_zz.set_ylim([median_z_value-acceptable_z_range*3,median_z_value+acceptable_z_range*3])
    ax_zz.set_ylim([ax_zz.get_ylim()[0]-np.diff(ax_zz.get_ylim()),ax_zz.get_ylim()[1]])
    ax_zz.set_ylabel('Z offset (plane)')
    #ax_zz.legend()#(bbox_to_anchor = (1.1,1))
    ax_contrast.legend()
    fig.savefig(os.path.join(FOV_dir,'XYZ_motion.pdf'), format="pdf")
    
    
    if segment_cells:
        #%% concatenate binned movie
        max_binned_frame_num = 7000#after this it fills up the 120GB RAM
        binned_movie_concatenated = []
        ops_loaded=  False
        total_binned_frame_num = 0
        for session in sessions:
            if 'z-stack' in session.lower() or '.' in session:
                continue
            if not ops_loaded:
                ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
                ops_loaded = True
            new_session_idx.append(trial_i)
            if subject in blacklist_for_binned_movie.keys():
                if session in blacklist_for_binned_movie[subject]:
                    print('session {} blacklisted, skipping from binned movie'.format(session))
                    continue
        # =============================================================================
        #     with open(os.path.join(FOV_dir,session,'filelist.json')) as f:
        #         filelist_dict = json.load(f)
        #     zcorr = np.asarray(filelist_dict['zoff_list']).squeeze()
        # =============================================================================
            zcorr = np.asarray(session_data_dict[session]['zoff']).squeeze()
            max_zcorr_vals = np.max(zcorr,1)
            min_zcorr_vals = np.min(zcorr,1)
            contrast = max_zcorr_vals/min_zcorr_vals
            median_z_session = np.median(np.argmax(zcorr,1)) +session_data_dict[session]['session_z_offfset']
            
            
            if np.percentile(contrast,10)<minimum_contrast:
                print('{} skipped due to low Z contrast'.format(session))
                continue
            if median_z_session<median_z_value-acceptable_z_range or median_z_session>median_z_value+acceptable_z_range :
                print('{} skipped due to wrong Z position'.format(session))
                continue
            print('loading the binned movie of {}'.format(session))
            mov = np.load(os.path.join(FOV_dir,session,'binned_movie.npy'))
            binned_movie_concatenated.append(mov)
            total_binned_frame_num += mov.shape[0]
            
        binned_movie_concatenated = np.concatenate(binned_movie_concatenated)   
        #%
        t_step_size = int(np.ceil(binned_movie_concatenated.shape[0]/max_binned_frame_num))
        binned_movie_concatenated = binned_movie_concatenated[::t_step_size,:,:]
        #%% segment ROIs
        if use_cellpose:
            ops['anatomical_only'] = True
            ops['diameter'] = [10,10]
        else:
            ops['anatomical_only'] = False
        ops['xrange'] = [0, ops['Lx']]
        ops['yrange'] = [0, ops['Ly']]
        try:
            del ops['meanImg_chan2']
        except:
            pass
        ops, stat = detect(ops, classfile=None, mov = binned_movie_concatenated)
        stat_original = stat.copy()
        #%% cut off pixels that drift out of the FOV 
        if np.percentile(np.concatenate(yoff_list),5)<0:
            ops['yrange'][0] = int(abs(np.percentile(np.concatenate(yoff_list),5)))
        if np.percentile(np.concatenate(xoff_list),5)<0:
            ops['xrange'][0] = int(abs(np.percentile(np.concatenate(xoff_list),5)))
            
        if np.percentile(np.concatenate(yoff_list),95)>0:
            ops['yrange'][1] = ops['Ly']-int(abs(np.percentile(np.concatenate(yoff_list),95)))
        if np.percentile(np.concatenate(xoff_list),95)>0:
            ops['xrange'][1] = ops['Lx']-int(abs(np.percentile(np.concatenate(xoff_list),95)))
        
        stat = []
        for s in stat_original:
            needed_idx = (s['ypix']>ops['yrange'][0]) &(s['ypix']<ops['yrange'][1]) &(s['xpix']>ops['xrange'][0])  &(s['xpix']<ops['xrange'][1])
            for k in s.keys():
                try:
                    if len(s[k]) == len(needed_idx):
                        s[k] = s[k][needed_idx]
                except:
                    pass
            if sum(needed_idx)>1:
                stat.append(s)
            #break
        #%% calculate neuroil and cell masks
        cell_masks_, neuropil_masks_ = create_masks(ops, stat)
        #%% select good rois
        cutoff_pixel_num = [50, 300]
        minimum_pixel_num = 10
        cell_masks = []
        neuropil_masks = []
        rois = np.zeros_like(ops['meanImg'])
        rois_small = np.zeros_like(ops['meanImg'])
        rois_good = np.zeros_like(ops['meanImg'])
        npix_list = []
        npix_list_somacrop = []
        npix_list_somacrop_nooverlap = []
        stat_good = []
        cell_masks_rest = []
        neuropil_masks_rest = []
        stat_rest = []
        for i,(s,cell_mask,neuropil_mask) in enumerate(zip(stat,cell_masks_,neuropil_masks_)):
            #neurpil_coord = np.unravel_index(s['neuropil_mask'],rois.shape)
            rois[s['ypix'][s['soma_crop']==True],s['xpix'][s['soma_crop']==True]] += s['npix']#s['lam']/np.sum(s['lam'])
            npix_list.append(s['npix'])
            npix_list_somacrop.append(s['npix_soma'])
            npix_list_somacrop_nooverlap.append(sum((s['overlap'] == False) & s['soma_crop']))
            #rois[neurpil_coord[0],neurpil_coord[1]] = .5#cell['lam']/np.sum(cell['lam'])
            idx = (s['soma_crop']==True) & (s['overlap']==False)
            pixel_num = sum( s['soma_crop']) #(s['overlap'] == False) &
            useful_pixel_num = sum((s['overlap'] == False) & sum( s['soma_crop'])) #
            if pixel_num>=cutoff_pixel_num[0]  and pixel_num<=cutoff_pixel_num[1] and useful_pixel_num>minimum_pixel_num:
                rois_good[s['ypix'][idx],s['xpix'][idx]] =s['lam'][idx]/np.sum(s['lam'][idx])*sum(idx)
                cell_masks.append(cell_mask)
                neuropil_masks.append(neuropil_mask)
                stat_good.append(s)
            elif pixel_num<cutoff_pixel_num[0] and useful_pixel_num>minimum_pixel_num:
                rois_small[s['ypix'][idx],s['xpix'][idx]] =s['lam'][idx]/np.sum(s['lam'][idx])*sum(idx)
                cell_masks_rest.append(cell_mask)
                neuropil_masks_rest.append(neuropil_mask)
                stat_rest.append(s)
        #%
        np.save(os.path.join(FOV_dir, 'stat.npy'), stat_good)
        scipy.io.savemat(
            file_name=os.path.join(FOV_dir, 'stat.mat'),
            mdict={
                'stat': stat_good
            }
        )
        np.save(os.path.join(FOV_dir, 'cell_masks.npy'), cell_masks)
        np.save(os.path.join(FOV_dir, 'neuropil_masks.npy'), neuropil_masks)
         
        np.save(os.path.join(FOV_dir, 'stat_rest.npy'), stat_rest)
        np.save(os.path.join(FOV_dir, 'cell_masks_rest.npy'), cell_masks_rest)
        np.save(os.path.join(FOV_dir, 'neuropil_masks_rest.npy'), neuropil_masks_rest)
               
        #%
        mean_image = np.mean(binned_movie_concatenated,0)
        max_image = np.max(binned_movie_concatenated,0)
        std_image = np.std(binned_movie_concatenated,0)
        #%
        np.save(os.path.join(FOV_dir, 'mean_image.npy'), mean_image)
        np.save(os.path.join(FOV_dir, 'max_image.npy'), max_image)
        np.save(os.path.join(FOV_dir, 'std_image.npy'), std_image)
                
        #%%
        fig = plt.figure(figsize = [20,20])
        ax_rois = fig.add_subplot(2,3,4)
        ax_rois.set_title('all ROIs (n = {})'.format(len(stat)))
        im = ax_rois.imshow(rois)
        im.set_clim([0,200])
        ax_rois_small = fig.add_subplot(2,3,5,sharex =ax_rois,sharey = ax_rois )
        ax_rois_small.set_title('small ROIs (n = {})'.format(len(stat_rest)))
        im2 = ax_rois_small.imshow(rois_small)
        im2.set_clim([0,np.percentile(rois_small[rois_small>0],95)])
        ax_rois_good = fig.add_subplot(2,3,6,sharex =ax_rois,sharey = ax_rois)
        ax_rois_good.set_title('cellular ROIs (n = {})'.format(len(stat_good)))
        im3 = ax_rois_good.imshow(rois_good)
        im3.set_clim([0,np.percentile(rois_good[rois_good>0],95)])
        
        fig_roisize = plt.figure(figsize = [20,20])
        ax_roisize = fig_roisize.add_subplot(3,1,1)
        ax_roisize.set_title('ROI sizes')
        ax_roisize_soma = fig_roisize.add_subplot(3,1,2,sharex = ax_roisize)
        ax_roisize_soma.set_title('somatic ROI sizes')
        ax_roisize_soma_nooverlap = fig_roisize.add_subplot(3,1,3,sharex = ax_roisize)
        ax_roisize_soma_nooverlap.set_title('somatic ROI sizes without overlaps')
        ax_roisize.hist(npix_list,np.arange(0,300,5))
        ax_roisize_soma.hist(npix_list_somacrop,np.arange(0,300,5))
        ax_roisize_soma_nooverlap.hist(npix_list_somacrop_nooverlap,np.arange(0,300,5))
        ax_roisize_soma.axvline(cutoff_pixel_num[0],color ='red')
        ax_roisize_soma.axvline(cutoff_pixel_num[1],color ='red', label = 'pixel size cutoff')
        ax_roisize_soma_nooverlap.set_xlabel('Pixel count')
        
        ax_meanimage = fig.add_subplot(2,3,1,sharex =ax_rois,sharey = ax_rois )
        ax_meanimage.set_title('mean image')
        im = ax_meanimage.imshow(mean_image)
        im.set_clim(np.percentile(mean_image.flatten(),[0,95]))
        ax_maximage = fig.add_subplot(2,3,2,sharex =ax_rois,sharey = ax_rois )
        ax_maximage.set_title('max projection')
        im = ax_maximage.imshow(max_image)
        im = ax_meanimage.imshow(mean_image)
        im.set_clim(np.percentile(max_image.flatten(),[0,95]))
        
        
        ax_stdimage = fig.add_subplot(2,3,3,sharex =ax_rois,sharey = ax_rois )
        ax_stdimage.set_title('std of image')
        im = ax_stdimage.imshow(std_image)
        im.set_clim(np.percentile(std_image.flatten(),[0,95]))
        if use_cellpose:
            fig.savefig(os.path.join(FOV_dir,'ROIs_cellpose.pdf'), format="pdf")
            fig_roisize.savefig(os.path.join(FOV_dir,'ROI_sizes_cellpose.pdf'), format="pdf")
        else:
            fig.savefig(os.path.join(FOV_dir,'ROIs.pdf'), format="pdf")
            fig_roisize.savefig(os.path.join(FOV_dir,'ROI_sizes.pdf'), format="pdf")
    
    #%% generate MEGATIFF
    
    
    imgs_all = []
    for i,session in enumerate(sessions):
        if 'z-stack' in session.lower() or '.' in session:
            continue
    
        imgs_ = np.load(os.path.join(FOV_dir,session,'meanImg.npy'))
        imgs = []
        for ti,img in enumerate(imgs_):
            texted_image =cv2.putText(img=np.copy(img), text="{}_T{}".format(session,ti), org=(20,40),fontFace=3, fontScale=1, color=(255,255,255), thickness=2)
            imgs.append(texted_image)
            #print([i,ti])
        imgs = np.asarray(imgs,dtype = np.int32)
        if len(imgs_all) == 0:
            imgs_all = imgs
        else:
            imgs_all = np.concatenate([imgs_all,imgs])
        i+=1
        #%%
    tifffile.imsave(os.path.join(FOV_dir,'meanimages.tiff'),imgs_all)
    meanImages = imgs_all.copy()
    
    #%% generate Session mean images
    imgs_all = []
    meanImages = imgs_all.copy()
    
    for i,session in enumerate(sessions):
        if 'z-stack' in session.lower() or '.' in session:
            continue
        reference_movie_directory = os.path.join(FOV_dir,session)
        with open(os.path.join(reference_movie_directory,'filelist.json')) as f:
            filelist_dict = json.load(f)
        print(session)
        #ops = np.load(os.path.join(reference_movie_directory,'ops.npy'),allow_pickle=True).tolist()
        #z_plane_indices = np.argmax(ops['zcorr_list'],1)
        z_plane_indices = filelist_dict['zoff_mean_list'] ##HOTFIX - ops and filelist doesn't match ??
        non_photostim = []
        for framenum, filename,z in zip(filelist_dict['frame_num_list'],filelist_dict['file_name_list'],filelist_dict['zoff_list']):
            bad_trial = False
            for photo_name in photostim_name_list:
                if photo_name in filename.lower():
                    bad_trial=True
            if bad_trial:
                non_photostim.append(False)
            else:
                non_photostim.append(True)
        #print([len(z_plane_indices),len(z_plane_indices_2)])
        needed_trials = (z_plane_indices == np.median(z_plane_indices))& np.asarray(non_photostim) #)
        meanimage_all = np.load(os.path.join(reference_movie_directory,'meanImg.npy'))
        mean_img = np.mean(meanimage_all[needed_trials,:,:],0)
        texted_image =cv2.putText(img=np.copy(mean_img), text="{}".format(session), org=(20,40),fontFace=3, fontScale=1, color=(255,255,255), thickness=2)
        imgs_all.append(texted_image)
    tifffile.imsave(os.path.join(FOV_dir,'session_meanimages.tiff'),imgs_all)
    #%% register each session to every z-stack
   