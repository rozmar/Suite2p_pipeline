#%% generate binned movies
import os, json
import scipy
import numpy as np
from pathlib import Path
import  matplotlib.pyplot as plt
from suite2p.detection.detect import detect
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces_from_masks
import sys
%matplotlib qt
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
        minimum_contrast = int(sys.argv[8])#'Bergamo-2P-Photostim'
    except:
        minimum_contrast = None
    try:
        acceptable_z_range = int(sys.argv[9])#'Bergamo-2P-Photostim'
    except:
        acceptable_z_range = None   
except:
    local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/'
    metadata_dir = '/mnt/Data/BCI_metadata/'
    raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
    suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
    subject = 'BCI_26'
    setup = 'Bergamo-2P-Photostim'
    fov = 'FOV_06'#'_reference_is_1st_session'#'FOV_03'
    
    minimum_contrast = None
    acceptable_z_range = 1
    
use_cellpose = False
photostim_name_list = ['slm','stim','file','photo']
#photostim_name_list = []

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
mean_intensity_list = []
trial_i = 0
new_session_idx =[]
new_session_names = []
median_z_values = []
xoff_list = []
yoff_list = []
session_data_dict = {}
for session in sessions:
    if 'z-stack' in session.lower() or '.' in session:
        continue
    print(session)
    new_session_idx.append(trial_i)
    new_session_names.append(session)
    with open(os.path.join(FOV_dir,session,'filelist.json')) as f:
        filelist_dict = json.load(f)
    ops = np.load(os.path.join(FOV_dir,session,'ops.npy'),allow_pickle = True).tolist()
    
    
    concatenated_movie_filelist_json = os.path.join(FOV_dir,session,'filelist.json')
    with open(concatenated_movie_filelist_json, "r") as read_file:
        filelist_dict = json.load(read_file)
    
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
                                  'zoff':zoff_now}
    
    xoff_mean_list_concatenated.extend(xoff_mean_now)
    yoff_mean_list_concatenated.extend(yoff_mean_now)
    xoff_std_list_concatenated.extend(xoff_std_now)
    yoff_std_list_concatenated.extend(yoff_std_now)
    zcorr_list_concatenated.extend(zoff_now)
    mean_intensity_list.extend(mean_int_now)
    trial_i += len(xoff_mean_now)
    median_z_values.append(np.median(np.argmax(zoff_now,2).squeeze()))
new_session_idx.append(trial_i) # end of all trials
xoff_mean_list_concatenated = np.asarray(xoff_mean_list_concatenated)        
yoff_mean_list_concatenated = np.asarray(yoff_mean_list_concatenated)        
xoff_std_list_concatenated = np.asarray(xoff_std_list_concatenated)        
yoff_std_list_concatenated = np.asarray(yoff_std_list_concatenated)        

mean_intensity_list = np.asarray(mean_intensity_list)      

zcorr_list_concatenated = np.concatenate(zcorr_list_concatenated).squeeze()
max_zcorr_vals = np.max(zcorr_list_concatenated,1)
min_zcorr_vals = np.min(zcorr_list_concatenated,1)
contrast = max_zcorr_vals/min_zcorr_vals

zcorr_list_concatenated_norm = (zcorr_list_concatenated - min_zcorr_vals[:,np.newaxis])/(max_zcorr_vals-min_zcorr_vals)[:,np.newaxis]
hw = zcorr_list_concatenated_norm.shape[1]-np.argmax(zcorr_list_concatenated_norm[:,::-1]>.5,1) - np.argmax(zcorr_list_concatenated_norm>.5,1)
z_with_hw = (zcorr_list_concatenated_norm.shape[1]-np.argmax(zcorr_list_concatenated_norm[:,::-1]>.5,1) + np.argmax(zcorr_list_concatenated_norm>.5,1))/2

if minimum_contrast is None:
    minimum_contrast = np.percentile(contrast,90)/2

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
img = zcorr_list_concatenated.T
ax_z.imshow(img ,aspect='auto', alpha = 1,origin='lower',cmap = 'magma')

x = np.arange(len(xoff_mean_list_concatenated))
ax_xy.errorbar(x, xoff_mean_list_concatenated,yerr = xoff_std_list_concatenated,fmt = '-',label = 'X offset')
ax_xy.errorbar(x, yoff_mean_list_concatenated,yerr = yoff_std_list_concatenated,fmt = '-',label = 'Y offset')

ax_mean_intensity.plot(x,mean_intensity_list,'g-')

ax_zz.plot(x, np.argmax(zcorr_list_concatenated.squeeze(),1),'r-',label = 'Z offset')
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
ax_zz.set_ylim([ax_zz.get_ylim()[0]-np.diff(ax_zz.get_ylim()),ax_zz.get_ylim()[1]])
ax_zz.set_ylabel('Z offset (plane)')
#ax_zz.legend()#(bbox_to_anchor = (1.1,1))
ax_contrast.legend()
fig.savefig(os.path.join(FOV_dir,'XYZ_motion.pdf'), format="pdf")



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
    
# =============================================================================
#     with open(os.path.join(FOV_dir,session,'filelist.json')) as f:
#         filelist_dict = json.load(f)
#     zcorr = np.asarray(filelist_dict['zoff_list']).squeeze()
# =============================================================================
    zcorr = np.asarray(session_data_dict[session]['zoff']).squeeze()
    max_zcorr_vals = np.max(zcorr,1)
    min_zcorr_vals = np.min(zcorr,1)
    contrast = max_zcorr_vals/min_zcorr_vals
    median_z_session = np.median(np.argmax(zcorr,1))
    
    
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
    pixel_num = sum((s['overlap'] == False) & s['soma_crop'])
    if pixel_num>=cutoff_pixel_num[0]  and pixel_num<=cutoff_pixel_num[1]:
        rois_good[s['ypix'][idx],s['xpix'][idx]] =s['lam'][idx]/np.sum(s['lam'][idx])*sum(idx)
        cell_masks.append(cell_mask)
        neuropil_masks.append(neuropil_mask)
        stat_good.append(s)
    else:
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
ax_roisize_soma_nooverlap.axvline(cutoff_pixel_num[0],color ='red')
ax_roisize_soma_nooverlap.axvline(cutoff_pixel_num[1],color ='red', label = 'pixel size cutoff')
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

import cv2
import tifffile
imgs_all = []
for i,session in enumerate(sessions):
    if 'z-stack' in session.lower() or '.' in session:
        continue

    imgs_ = np.load(os.path.join(FOV_dir,session,'meanImg.npy'))
    imgs = []
    for ti,img in enumerate(imgs_):
        texted_image =cv2.putText(img=np.copy(img), text="{}_T{}".format(session,ti), org=(20,40),fontFace=3, fontScale=1, color=(255,255,255), thickness=2)
        imgs.append(texted_image)
        print([i,ti])
    imgs = np.asarray(imgs,dtype = np.int32)
    if len(imgs_all) == 0:
        imgs_all = imgs
    else:
        imgs_all = np.concatenate([imgs_all,imgs])
    i+=1
tifffile.imsave(os.path.join(FOV_dir,'meanimages.tiff'),imgs_all)


#% generate Session mean images
imgs_all = []
import cv2
import tifffile
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

