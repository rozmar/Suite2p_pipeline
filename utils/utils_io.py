import os
import shutil
from pathlib import Path
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
import json
import datetime
import time

def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
    """Patches shutil method to hugely improve copy speed"""
    #from stackoverflow for faster copy - bigger buffer size
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
shutil.copyfileobj = _copyfileobj_patched

#%%
def extract_scanimage_metadata(file): # this is a duplicate function - also in utils_imaging
    #%
    image = ScanImageTiffReader(file)
    metadata_raw = image.metadata()
    description_first_image = image.description(0)
    description_first_image_dict = dict(item.split(' = ') for item in description_first_image.rstrip(r'\n ').rstrip('\n').split('\n'))
    metadata_str = metadata_raw.split('\n\n')[0]
    metadata_json = metadata_raw.split('\n\n')[1]
    metadata_dict = dict(item.split('=') for item in metadata_str.split('\n') if 'SI.' in item)
    metadata = {k.strip().replace('SI.','') : v.strip() for k, v in metadata_dict.items()}
    for k in list(metadata.keys()):
        if '.' in k:
            ks = k.split('.')
            # TODO just recursively create dict from .-containing values
            if k.count('.') == 1:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') == 2:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                    metadata[ks[0]][ks[1]] = {}
                elif not ks[1] in metadata[ks[0]].keys():
                    metadata[ks[0]][ks[1]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') > 2:
                print('skipped metadata key ' + k + ' to minimize recursion in dict')
            metadata.pop(k)
    metadata['json'] = json.loads(metadata_json)
    frame_rate = metadata['hRoiManager']['scanVolumeRate']
    try:
        z_collection = metadata['hFastZ']['userZs']
        num_planes = len(z_collection)
    except: # new scanimage version
        if metadata['hFastZ']['enable'] == 'true':
            print('multiple planes not handled in metadata collection.. HANDLE ME!!!')
            #time.sleep(1000)
            num_planes = 1
        else:
            num_planes = 1
    
    roi_metadata = metadata['json']['RoiGroups']['imagingRoiGroup']['rois']
    
    
    if type(roi_metadata) == dict:
        roi_metadata = [roi_metadata]
    num_rois = len(roi_metadata)
    roi = {}
    w_px = []
    h_px = []
    cXY = []
    szXY = []
    for r in range(num_rois):
        roi[r] = {}
        roi[r]['w_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][0]
        w_px.append(roi[r]['w_px'])
        roi[r]['h_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][1]
        h_px.append(roi[r]['h_px'])
        roi[r]['center'] = roi_metadata[r]['scanfields']['centerXY']
        cXY.append(roi[r]['center'])
        roi[r]['size'] = roi_metadata[r]['scanfields']['sizeXY']
        szXY.append(roi[r]['size'])
        #print('{} {} {}'.format(roi[r]['w_px'], roi[r]['h_px'], roi[r]['size']))
    
    w_px = np.asarray(w_px)
    h_px = np.asarray(h_px)
    szXY = np.asarray(szXY)
    cXY = np.asarray(cXY)
    cXY = cXY - szXY / 2
    cXY = cXY - np.amin(cXY, axis=0)
    mu = np.median(np.transpose(np.asarray([w_px, h_px])) / szXY, axis=0)
    imin = cXY * mu
    
    n_rows_sum = np.sum(h_px)
    n_flyback = (image.shape()[1] - n_rows_sum) / np.max([1, num_rois - 1])
    
    irow = np.insert(np.cumsum(np.transpose(h_px) + n_flyback), 0, 0)
    irow = np.delete(irow, -1)
    irow = np.vstack((irow, irow + np.transpose(h_px)))
    
    data = {}
    data['fs'] = frame_rate
    data['nplanes'] = num_planes
    data['nrois'] = num_rois #or irow.shape[1]?
    if data['nrois'] == 1:
        data['mesoscan'] = 0
    else:
        data['mesoscan'] = 1
    
    if data['mesoscan']:
        #data['nrois'] = num_rois #or irow.shape[1]?
        data['dx'] = []
        data['dy'] = []
        data['lines'] = []
        for i in range(num_rois):
            data['dx'] = np.hstack((data['dx'], imin[i,1]))
            data['dy'] = np.hstack((data['dy'], imin[i,0]))
            data['lines'] = list(range(irow[0,i].astype('int32'), irow[1,i].astype('int32') - 1)) ### TODO NOT QUITE RIGHT YET
        data['dx'] = data['dx'].astype('int32')
        data['dy'] = data['dy'].astype('int32')
        print(data['dx'])
        print(data['dy'])
        print(data['lines'])
            #data['lines']{i} = 
            #data.dx(i) = int32(imin(i,2));
            #data.dy(i) = int32(imin(i,1));
            #data.lines{i} = irow(1,i):(irow(2,i)-1)
    movie_start_time = datetime.datetime.strptime(description_first_image_dict['epoch'].rstrip(']').lstrip('['), '%Y %m %d %H %M %S.%f')
    out = {'metadata':metadata,
           'roidata':data,
           'roi_metadata':roi_metadata,
           'frame_rate':frame_rate,
           'num_planes':num_planes,
           'shape':image.shape(),
           'description_first_frame':description_first_image_dict,
           'movie_start_time': movie_start_time}
    #%%
    return out
def extract_files_from_dir(basedir):
    #%
    files = os.listdir(basedir)
    exts = list()
    basenames = list()
    fileindexs = list()
    dirs = list()
    for file in files:
        if Path(os.path.join(basedir,file)).is_dir():
            exts.append('')
            basenames.append('')
            fileindexs.append(np.nan)
            dirs.append(file)
        else:
            if '_' in file:# and ('cell' in file.lower() or 'stim' in file.lower()):
                basenames.append(file[:-1*file[::-1].find('_')-1])
                try:
                    fileindexs.append(int(file[-1*file[::-1].find('_'):file.find('.')]))
                except:
                    print('weird file index: {}'.format(file))
                    fileindexs.append(-1)
            else:
                basenames.append(file[:file.find('.')])
                fileindexs.append(-1)
            exts.append(file[file.find('.'):])
    tokeep = np.asarray(exts) != ''
    files = np.asarray(files)[tokeep]
    exts = np.asarray(exts)[tokeep]
    basenames = np.asarray(basenames)[tokeep]
    fileindexs = np.asarray(fileindexs)[tokeep]
    out = {'dir':basedir,
           'filenames':files,
           'exts':exts,
           'basenames':basenames,
           'fileindices':fileindexs,
           'dirs':dirs
           }
    #%
    return out

def copy_tiff_files_in_loop(source_movie_directory,target_movie_directory):
    while True:
        copy_tiff_files_in_order(source_movie_directory,target_movie_directory)
        time.sleep(5)
        

def copy_tiff_files_in_order(source_movie_directory,target_movie_directory):
    source_movie_directory = source_movie_directory.strip("'").strip('"')
    target_movie_directory = target_movie_directory.strip("'").strip('"')
    Path(target_movie_directory).mkdir(parents = True,exist_ok = True)
    dirs_in_target_dir = os.listdir(target_movie_directory)

    files_dict = extract_files_from_dir(source_movie_directory)
    file_idxs = (files_dict['exts']=='.tif')
    fnames = files_dict['filenames'][file_idxs]
    file_indices = files_dict['fileindices'][file_idxs]
    basenames = files_dict['basenames'][file_idxs]
    order  = np.argsort(file_indices)
    fnames = fnames[order]
    basenames = basenames[order]
    file_indices = file_indices[order]
    uniquebasenames = np.unique(basenames)
    stacknames = list()
    for basename in uniquebasenames:
        if 'stack' in basename:
            stacks = basenames == basename
            stacknames.extend(fnames[stacks])
            needed = basenames != basename
            basenames = basenames[needed]
            fnames = fnames[needed]
            file_indices = file_indices[needed]
        
    uniquebasenames = np.unique(basenames)
    start_times = list()
    tokeep = np.ones(len(uniquebasenames))
    for i,basename in enumerate(uniquebasenames):
        fname = fnames[np.where(basename==basenames)[0][0]]
        try:
            metadata = extract_scanimage_metadata(os.path.join(files_dict['dir'],fname))
            start_times.append(metadata['movie_start_time'])
        except:
            print('error in {}'.format(basename))
            start_times.append(np.nan)
            tokeep[i]=0
    start_times = np.asarray(start_times)[tokeep==1]
    uniquebasenames = np.asarray(uniquebasenames)[tokeep==1]    
    order = np.argsort(start_times)  
    fnames_new = list()
    for idx in order:
        fnames_new.append(fnames[basenames==uniquebasenames[idx]])
    fnames = np.concatenate(fnames_new)
    
    #%
    try:
        file_dict = np.load(os.path.join(target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
    except:
        file_dict = {'copied_files':list(),
                     'h5_files':list()}
    if 'copied_stacks' not in file_dict.keys():
        file_dict['copied_stacks'] = list()
    if 'copy_finished' not in file_dict.keys():
        file_dict['copy_finished'] = False
# =============================================================================
#     for stackname in stacknames:
#         if stackname not in file_dict['copied_stacks']:
#             target_dir = os.path.join(target_movie_directory,stackname[:-4])
#             Path(target_dir).mkdir(parents = True,exist_ok = True)
#             sourcefile = os.path.join(source_movie_directory,stackname)
#             destfile = os.path.join(os.path.join(target_movie_directory,stackname[:-4]),stackname)
#             shutil.copyfile(sourcefile,destfile+'_tmp')
#             os.rename(destfile+'_tmp',destfile)
#             file_dict['copied_stacks'].append(stackname)
#             np.save(os.path.join(target_movie_directory,'copy_data.npy'),file_dict)
#             
# =============================================================================
    print('starting to copy {} files'.format(len(fnames)))
    for fname in fnames:
        if fname not in file_dict['copied_files']:#dirs_in_target_dir: 
            #metadata = extract_scanimage_metadata(os.path.join(files_dict['dir'],fname))
            target_dir = os.path.join(target_movie_directory,fname[:-4])
            Path(target_dir).mkdir(parents = True,exist_ok = True)
            sourcefile = os.path.join(source_movie_directory,fname)
            destfile = os.path.join(os.path.join(target_movie_directory,fname[:-4]),fname)
            shutil.copyfile(sourcefile,destfile+'_tmp')
            os.rename(destfile+'_tmp',destfile)
            file_dict['copied_files'].append(fname)
            np.save(os.path.join(target_movie_directory,'copy_data.npy'),file_dict)
            #break
    file_dict['copy_finished'] = True
    np.save(os.path.join(target_movie_directory,'copy_data.npy'),file_dict)
# =============================================================================
#     h5_file_idxs = (files_dict['exts']=='.h5')
#     h5_fnames = files_dict['filenames'][h5_file_idxs]
#     for fname in h5_fnames:
#         if fname not in file_dict['h5_files'] and 'slm' in fname.lower():
#             target_dir = os.path.join(target_movie_directory,'_ws_files')
#             Path(target_dir).mkdir(parents = True,exist_ok = True)
#             sourcefile = os.path.join(source_movie_directory,fname)
#             destfile = os.path.join(os.path.join(target_movie_directory,'_ws_files'),fname)
#             shutil.copyfile(sourcefile,destfile+'_tmp')
#             os.rename(destfile+'_tmp',destfile)
#             try:
#                 file_dict['h5_files'].append(fname)
#             except:
#                 file_dict['h5_files'] = [fname]
#             np.save(os.path.join(target_movie_directory,'copy_data.npy'),file_dict)
# =============================================================================


def concatenate_suite2p_files(target_movie_directory):
    #%%
    concatenated_ops_loaded = False
    concatenated_movie_dir = os.path.join(target_movie_directory,'_concatenated_movie')
    Path(concatenated_movie_dir).mkdir(parents = True,exist_ok = True)
    concatenated_movie_file = os.path.join(target_movie_directory,'_concatenated_movie','data.bin')
    concatenated_movie_file_chan2 = os.path.join(target_movie_directory,'_concatenated_movie','data_chan2.bin')
    concatenated_movie_ops = os.path.join(target_movie_directory,'_concatenated_movie','ops.npy')
    meanimg_file = os.path.join(target_movie_directory,'_concatenated_movie','meanImg.npy')
    concatenated_movie_filelist_json = os.path.join(target_movie_directory,'_concatenated_movie','filelist.json')
    try:
        with open(concatenated_movie_filelist_json, "r") as read_file:
            filelist_dict = json.load(read_file)
    except:
        filelist_dict = {'file_name_list' : [],
                         'frame_num_list' :[],
                         'file_added_time':[],
                         'xoff_mean_list':[],
                         'yoff_mean_list':[],
                         'xoff_std_list':[],
                         'yoff_std_list':[],
                         'zoff_mean_list':[],
                         'zoff_std_list':[],
                         'mean_instensity':[],
                         'concatenation_underway':True}
    file_dict = np.load(os.path.join(target_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
    for file_idx,file in enumerate(file_dict['copied_files']):
# =============================================================================
#         if file == 'openLoop_00004.tif':
#             break
# =============================================================================
        #try:
            #%
        print(file)
        dir_now = os.path.join(target_movie_directory,file[:-4])
        try:
            os.listdir(dir_now)
        except:
            print('no movie dir found')
            break
        if 'reg_progress.json' not in os.listdir(dir_now):
            print('no json file for {}'.format(file))
            break
        with open(os.path.join(dir_now,'reg_progress.json'), "r") as read_file:
            reg_dict = json.load(read_file)
        if 'registration_finished' not in reg_dict.keys():
            print('registration is not done, stopped at {}'.format(file))
            break
        if not reg_dict['registration_finished']:
            print('registration is not done, stopped at {}'.format(file))
            break
        if file in filelist_dict['file_name_list']: # skip files that are already added
            continue
        ops = np.load(os.path.join(dir_now,'suite2p','plane0','ops.npy'),allow_pickle = True).tolist()
        try:
            ops['zcorr']=ops['zcorr'].T
        except:
            pass
        sourcefile = os.path.join(dir_now,'suite2p','plane0','data.bin')
        sourcefile_chan2 = os.path.join(dir_now,'suite2p','plane0','data_chan2.bin')
        #%
        if file_idx == 0: #the first one is copied
            shutil.copy(sourcefile,concatenated_movie_file)
            if os.path.exists(sourcefile_chan2):
                shutil.copy(sourcefile_chan2,concatenated_movie_file_chan2)
            np.save(concatenated_movie_ops,ops)
            filelist_dict['file_name_list'].append(file)
            filelist_dict['frame_num_list'].append(ops['nframes'])
            filelist_dict['file_added_time'].append(str(datetime.datetime.now()))
            filelist_dict['concatenation_underway'] = True
            filelist_dict['xoff_mean_list'] = [np.mean(ops['xoff'])]
            filelist_dict['yoff_mean_list'] = [np.mean(ops['yoff'])]
            filelist_dict['xoff_std_list'] = [np.std(ops['xoff'])]
            filelist_dict['yoff_std_list'] = [np.std(ops['yoff'])]#'zcorr'
            filelist_dict['mean_instensity'] = [np.mean(ops['meanImg'])]
            try:
                zcorr = np.argmax(ops['zcorr'],1)
                filelist_dict['zoff_mean_list'] = [np.mean(zcorr)]
                filelist_dict['zoff_std_list'] = [np.std(zcorr)]#''
                filelist_dict['zoff_list'] = [ops['zcorr'].tolist()]
            except:
                pass # no zcorr
            meanimg_list = np.asarray([ops['meanImg']])
            np.save(meanimg_file,meanimg_list)
        else:
            
            #%
            with open(concatenated_movie_file, "ab") as myfile, open(sourcefile, "rb") as file2:
                myfile.write(file2.read())
            if os.path.exists(concatenated_movie_file_chan2):
                with open(concatenated_movie_file_chan2, "ab") as myfile, open(sourcefile_chan2, "rb") as file2:
                    myfile.write(file2.read())
            
            if not concatenated_ops_loaded:
                ops_concatenated = np.load(concatenated_movie_ops,allow_pickle = True).tolist()
                meanimg_list = np.load(meanimg_file)
                concatenated_ops_loaded = True
            
            meanimg_list = np.concatenate([meanimg_list,[ops['meanImg']]])
            
            for key in ops.keys():
                if key == 'do_regmetrics':
                    continue
                
                #%
                addlist = False
                try:
                    if ops[key]!=ops_concatenated[key] or key in ['xrange','yrange','nframes','frames_per_file','frames_per_folder','tPC','fs','bidi_corrected','bidiphase']:
                        addlist = True
                        #print(key)
                except:
                    addlist = True
                    #print('error:' + key)
                #%
                if not addlist:
                    continue
                if file_idx == 1:
                     ops_concatenated[key+'_list'] = ops_concatenated[key]
                #%
                skipit = False
                try: # ref and mean images have to be concatenated in a different way
                    if ops['Lx'] in ops[key].shape and ops['Ly'] in ops[key].shape:
                        ops_concatenated[key+'_list'] = (ops_concatenated[key+'_list']*(file_idx+1) + ops[key])/(file_idx+2)
                        skipit = True
                except:
                    pass
                if not skipit:
                    try:
                        ops_concatenated[key+'_list'] = np.concatenate([ops_concatenated[key+'_list'],ops[key]])  
                    except:
                        try:
                            ops_concatenated[key+'_list'] = np.concatenate([ops_concatenated[key+'_list'],[ops[key]]])
                        except:
                            ops_concatenated[key+'_list'] = np.concatenate([[ops_concatenated[key+'_list']],[ops[key]]])
                
                    
                    
            #%     
            
            filelist_dict['file_name_list'].append(file)
            filelist_dict['frame_num_list'].append(ops['nframes'])
            filelist_dict['file_added_time'].append(str(datetime.datetime.now()))
            filelist_dict['concatenation_underway'] = True
            filelist_dict['xoff_mean_list'].append(np.mean(ops['xoff']))
            filelist_dict['yoff_mean_list'].append(np.mean(ops['yoff']))
            filelist_dict['xoff_std_list'].append(np.std(ops['xoff']))
            filelist_dict['yoff_std_list'].append(np.std(ops['yoff']))
            filelist_dict['mean_instensity'].append(np.mean(ops['meanImg']))
            
            try:
                zcorr = np.argmax(ops['zcorr'],1)
                filelist_dict['zoff_mean_list'].append(np.mean(zcorr))
                filelist_dict['zoff_std_list'].append(np.std(zcorr))
                filelist_dict['zoff_list'].append(ops['zcorr'].tolist())
            except:
                pass # no zcorr
            
            with open(concatenated_movie_filelist_json, "w") as data_file:
                json.dump(filelist_dict, data_file, indent=2)
            #np.save(concatenated_movie_ops,ops_concatenated)        
                
                #break
# =============================================================================
#         except:
#             print('error occured, progress saved nonetheless')
# =============================================================================
    filelist_dict['concatenation_underway'] = False
    with open(concatenated_movie_filelist_json, "w") as data_file:
        json.dump(filelist_dict, data_file, indent=2)
    np.save(concatenated_movie_ops,ops_concatenated)   
    np.save(meanimg_file,meanimg_list)     




def concatenate_suite2p_sessions(source_movie_directory,target_movie_directory): #TODO this is just a copy of the function above. pls finish ASAP
    #%%
    concatenated_ops_loaded = False
    concatenated_movie_dir = target_movie_directory
    Path(concatenated_movie_dir).mkdir(parents = True,exist_ok = True)
    concatenated_movie_file = os.path.join(target_movie_directory,'data.bin')
    concatenated_movie_file_chan2 = os.path.join(target_movie_directory,'data_chan2.bin')
    concatenated_movie_ops = os.path.join(target_movie_directory,'ops.npy')
    meanimg_file = os.path.join(target_movie_directory,'meanImg.npy')
    concatenated_movie_filelist_json = os.path.join(target_movie_directory,'filelist.json')
    try:
        with open(concatenated_movie_filelist_json, "r") as read_file:
            filelist_dict = json.load(read_file)
    except:
        filelist_dict = {'file_name_list' : [],
                         'frame_num_list' :[],
                         'file_added_time':[],
                         'xoff_mean_list':[],
                         'yoff_mean_list':[],
                         'xoff_std_list':[],
                         'yoff_std_list':[],
                         'zoff_mean_list':[],
                         'zoff_std_list':[],
                         'concatenation_underway':True,
                         'session_name_list':[]}
    sessions = os.listdir(source_movie_directory)
    session_dates = []
    for session in sessions:
        try:
            session_date = datetime.datetime.strptime(session,'%m%d%y')
        except:
            try:
                session_date = datetime.datetime.strptime(session,'%Y-%m-%d')
            except:
                try:
                    session_date = datetime.datetime.strptime(session[:6],'%m%d%y')
                except:

                    print('cannot understand date for session dir: {} - should be a date'.format(session))
        session_dates.append(session_date)
    session_order = np.argsort(session_dates)
    sessions = np.asarray(sessions)[session_order]
    
    for session_idx,session in enumerate(sessions):

        print(session)
        dir_now = os.path.join(source_movie_directory,session)

        if session in filelist_dict['session_name_list']: # skip files that are already added
            continue
        with open(os.path.join(dir_now,'filelist.json'), "r") as read_file:
            filelist_dict_session = json.load(read_file)
        filelist_dict_session['session_name_list']=[session]*len(filelist_dict_session['file_name_list'])
        ops = np.load(os.path.join(dir_now,'ops.npy'),allow_pickle = True).tolist()
        sourcefile = os.path.join(dir_now,'data.bin')
        sourcefile_chan2 = os.path.join(dir_now,'data_chan2.bin')
        #%
        shutil.copy(os.path.join(dir_now,'meanImg.npy'),os.path.join(target_movie_directory,'meanImg_{}.npy'.format(session_idx)))
        if session_idx == 0: #the first one is copied
            shutil.copy(sourcefile,concatenated_movie_file)
            if os.path.exists(sourcefile_chan2):
                shutil.copy(sourcefile_chan2,concatenated_movie_file_chan2)
            np.save(concatenated_movie_ops,ops)
            filelist_dict = filelist_dict_session
            
            
        else:
            


            with open(concatenated_movie_file, "ab") as myfile, open(sourcefile, "rb") as file2:
                shutil.copyfileobj(file2, myfile)
                #myfile.write(file2.read())
            if os.path.exists(concatenated_movie_file_chan2):
                with open(concatenated_movie_file_chan2, "ab") as myfile, open(sourcefile_chan2, "rb") as file2:
                    myfile.write(file2.read())
            if not concatenated_ops_loaded:
                ops_concatenated = np.load(concatenated_movie_ops,allow_pickle = True).tolist()
                
                concatenated_ops_loaded = True
            
            for key in ops.keys():
                if '_list' in key:
                    ops_concatenated[key] = np.concatenate([ops_concatenated[key],ops[key]])  
                    
            for key in    filelist_dict.keys():
                try:
                    filelist_dict[key].extend(filelist_dict_session[key])
                except:
                    print('{} could not be extended'.format(key))
                    pass
            
            with open(concatenated_movie_filelist_json, "w") as data_file:
                json.dump(filelist_dict, data_file, indent=2)
            #np.save(concatenated_movie_ops,ops_concatenated)        
                
                #break
# =============================================================================
#         except:
#             print('error occured, progress saved nonetheless')
# =============================================================================
    filelist_dict['concatenation_underway'] = False
    with open(concatenated_movie_filelist_json, "w") as data_file:
        json.dump(filelist_dict, data_file, indent=2)
    np.save(concatenated_movie_ops,ops_concatenated)   
    