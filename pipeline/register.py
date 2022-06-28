import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
from utils import utils_imaging, utils_io
import datetime
import json
import time
from suite2p.io.binary import BinaryFile


def register_z_stacks(local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/',
                      metadata_dir = '/home/rozmar/Network/GoogleServices/BCI_data/Metadata/',
                      raw_scanimage_dir_base ='/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/',
                      suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/',
                      subject_ = None,
                      setup = None,
                      method = 'suite2p'):
    """
    

    Parameters
    ----------
    local_temp_dir : str
        location where the raw data will be copied over for processing
    metadata_dir : str
        location where the metadata resides (csv files from google drive)
    raw_scanimage_dir_base : str
        location where the raw data resides 
    suite2p_dir_base : str
        location where the suite2p outputs are stored
    subject_ : str
        name of the mouse
    setup : str
        name of the rig

    Returns
    -------
    None.

    """

    if setup is None:
        setups = os.listdir(raw_scanimage_dir_base)    
    else:
        setups = [setup]
    for setup in setups:
        if subject_ is None:
            subjects = os.listdir(os.path.join(raw_scanimage_dir_base,setup))
        else:
            subjects = [subject_]
        for subject in subjects:
            try:
                subject_metadata = pd.read_csv(os.path.join(metadata_dir,subject.replace('_','')+'.csv'))
            except:
                print('no metadata found')
                continue
            print(subject)
            
            #decode session dates here

            sessions = os.listdir(os.path.join(raw_scanimage_dir_base,setup,subject))
            session_date_dict = {}
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
                            print('cannot understand date for session dir: {}'.format(session))
                            continue
                if session_date.date() in session_date_dict.keys():
                    print('there were multiple sessions on {}'.format(session_date.date()))
                    session_date_dict[session_date.date()] = [session_date_dict[session_date.date()],session]
                else:
                    session_date_dict[session_date.date()] = session
            
            FOV_list_ = np.unique(np.asarray(subject_metadata['FOV'].values,str))
            FOV_list = []
            for FOV in FOV_list_:
                if FOV != '-' and FOV != 'nan' and len(FOV)>0:
                    FOV_list.append(FOV)
            
            for FOV in FOV_list:
                z_stacks = subject_metadata.loc[subject_metadata['FOV']==FOV,'Z-stack'].values
                sessions = subject_metadata.loc[subject_metadata['FOV']==FOV,'Date'].values
                for z_stack_,session_date in zip(z_stacks,sessions):
                    session_date = datetime.datetime.strptime(session_date,'%Y/%m/%d').date()
    # =============================================================================
    #                 if '[' in z_stack_:
    #                     z_stack_=z_stack_.strip('[]').split(',')
    #                 else:
    #                     z_stack_ = [z_stack_]
    # =============================================================================
                    if session_date not in session_date_dict.keys():
                        continue
                    session_ = session_date_dict[session_date]
                    
                    #print([session_,len(session)])
                    if type(session_)== str:
                        session_ = [session_]
                    for session in session_:
                        if type(z_stack_) is str:
                            z_stack_ = z_stack_.split(',')
                        else:
                            z_stack_ = [z_stack_]
                        for z_stack in z_stack_:
                            if type(z_stack) is not str:
                                continue
                            z_stack = z_stack.strip(' ')
                            print([FOV,session,z_stack])
                            z_stack = z_stack.strip(' ')
                            #session = datetime.datetime.strptime(session,'%Y/%m/%d').date()
                            if type(z_stack) is not str:
                                continue
                            z_stack_dir = os.path.join(suite2p_dir_base,setup,subject,FOV,'Z-stacks')
                            Path(z_stack_dir).mkdir(exist_ok = True, parents = True)
                            #z_stack_save_name = '{}_{}.tif'.format(session,z_stack)
                            new_zstack_name = '{}_{}_{}.tif'.format(subject,session,z_stack[:-4])
                            if new_zstack_name in os.listdir(z_stack_dir): 
                                continue #already done
                            try:
                                tiff_files_in_raw_folder = os.listdir(os.path.join(raw_scanimage_dir_base,setup,subject,session))
                            except:
                                continue # tiff doesn't exist in this session, it's going to be the other session today
                            if z_stack in tiff_files_in_raw_folder:
                                temp_dir = os.path.join(local_temp_dir,'{}_{}_{}'.format(subject,session,z_stack[:-4]))
                                Path(temp_dir).mkdir(exist_ok = True, parents = True)
                                if method == 'suite2p':
                                    utils_imaging.register_zstack(os.path.join(raw_scanimage_dir_base,setup,subject,session,z_stack) ,temp_dir)
                                else:
                                    utils_imaging.average_zstack(os.path.join(raw_scanimage_dir_base,setup,subject,session,z_stack) ,temp_dir)
                                
                                shutil.copyfile(os.path.join(temp_dir,new_zstack_name),os.path.join(z_stack_dir,new_zstack_name))




# this script registers trials and sessions together in a chronological order
#%% import libraries and set parameters



# =============================================================================
# local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/'
# metadata_dir = '/mnt/Data/BCI_metadata/'
# raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
# suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
# subject = 'BCI_29'
# setup = 'Bergamo-2P-Photostim'
# =============================================================================
def register_photostim(): #probably should just simply run the registration without even copying the files.. - just get all the slm files in a list and boom..
    pass



def register_session(local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/',
                     metadata_dir = '/mnt/Data/BCI_metadata/',
                     raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/',
                     suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/',
                     subject = None,
                     setup = None,
                     max_process_num = 4,
                     batch_size = 50,
                     FOV_needed = None):  
    
    ########### TODO these variables are hard-coded now
    repo_location = '/home/jupyter/Scripts/Suite2p_pipeline'#TODO this is hard-coded):
    suite2p_dir_base_gs = 'gs://aind-transfer-service-test/marton.rozsa/Data/Calcium_imaging/suite2p/'#TODO this is hard-coded):
    s2p_params = {'max_reg_shift':50, # microns
                'max_reg_shift_NR': 20, # microns
                'block_size': 200, # microns
                'smooth_sigma':0.5, # microns
                'smooth_sigma_time':0, #seconds,
                'overwrite': False,
                'batch_size':batch_size,
                #'num_workers':4,
                'z_stack_name':'',
                'reference_session':''} # folder where the suite2p output is saved
    photostim_name_list = ['slm','stim','file','photo']
    reference_is_previous_session = False
    skip_photostim_trials = True
    acceptable_z_range_for_binned_movie = 1
    minimum_contrast_for_binned_movie = None
    trial_number_for_mean_image = 10
    ########### TODO these variables are hard-coded now
    
    processes_running = 0
    
    #% metadata will be updated manually, this script will read .csv files
    subject_metadata = pd.read_csv(os.path.join(metadata_dir,subject.replace('_','')+'.csv'))
    #decode session dates here
    import datetime
    sessions = os.listdir(os.path.join(raw_scanimage_dir_base,setup,subject))
    session_date_dict = {}
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
                    print('cannot understand date for session dir: {}'.format(session))
                    continue
        if session_date.date() in session_date_dict.keys():
            print('there were multiple sessions on {}'.format(session_date.date()))
            if type(session_date_dict[session_date.date()]) == list():
                session_date_dict[session_date.date()].append(session)
            else:
                session_date_dict[session_date.date()] = [session_date_dict[session_date.date()],session]
        else:
            session_date_dict[session_date.date()] = session
    
    FOV_list_ = np.unique(np.asarray(subject_metadata['FOV'].values,str))
    FOV_list = []
    for FOV in FOV_list_:
        if FOV != '-' and FOV != 'nan' and len(FOV)>0:
            FOV_list.append(FOV)
    
    for FOV in FOV_list:
        if FOV_needed is not None:
            if FOV.lower() != FOV_needed.lower():
                continue
        first_session_of_FOV = True
        session_dates = subject_metadata.loc[subject_metadata['FOV']==FOV,'Date']
        training_types = subject_metadata.loc[subject_metadata['FOV']==FOV,'Training type']
        for session_date,training_type in zip(session_dates,training_types):
            if "bci" not in training_type.lower():
                print('no BCI training according to metadata in session {}'.format(session_date))
                continue
            session_date = datetime.datetime.strptime(session_date,'%Y/%m/%d').date()
            if session_date not in session_date_dict.keys():
                print('session not found in raw scanimage folder: {}'.format(session_date))
                continue
            session_ = session_date_dict[session_date]
            
            #print([session_,len(session)])
            if type(session_)== str:
                session_ = [session_]
            for session in session_:
                
                print(session)
                archive_movie_directory = os.path.join(suite2p_dir_base,setup,subject,FOV,session)
                archive_movie_directory_gs = os.path.join(suite2p_dir_base_gs,setup,subject,FOV,session)
                if os.path.exists(archive_movie_directory):
                    if len(os.listdir(archive_movie_directory))>1:
                        print('{} already registered and saved, skipping')
                        if first_session_of_FOV:
                            first_session_of_FOV = False
                            if not reference_is_previous_session:
                                reference_session_name = session
                        if reference_is_previous_session:
                            reference_session_name = session
                        continue
                # start copying files to local drive
                source_movie_directory = os.path.join(raw_scanimage_dir_base,setup,subject,session)
                temp_movie_directory = os.path.join(local_temp_dir,'{}_{}'.format(subject,session))
        # =============================================================================
        #         copy_thread = threading.Thread(target = utils_io.copy_tiff_files_in_order, args = (source_movie_directory,temp_movie_directory))
        #         copy_thread.start()
        # =============================================================================
        
                cluster_command_list = ['cd {}'.format(repo_location),
                                        
                                        'python cluster_helper.py {} "\'{}\'" {}'.format('utils_io.copy_tiff_files_in_order',source_movie_directory,temp_movie_directory)]
                bash_command = r" && ".join(cluster_command_list)+ r" &"
                print('copying files over - and not waiting for it')
                os.system(bash_command)
                
                #% select the first Z-stack in the FOV directory
                available_z_stacks = np.sort(os.listdir(os.path.join(suite2p_dir_base,setup,subject,FOV,'Z-stacks')))
                available_z_stacks_real = []
                z_stack_dates = []
                for zstackname in available_z_stacks:
                    if '.tif' in zstackname:
                        available_z_stacks_real.append(zstackname)
                        z_stack_dates.append(datetime.datetime.strptime(zstackname[len(subject)+1:len(subject)+1+6],'%m%d%y').date()) # assuming US date standard
                        
                        
                if len(available_z_stacks_real)>0:
                    zstack_idx = np.argmin(np.abs(np.asarray(z_stack_dates)-session_date))
                    zstackname = available_z_stacks_real[zstack_idx]
                    Path(os.path.join(temp_movie_directory,zstackname[:-4])).mkdir(parents = True,exist_ok = True)
                    shutil.copyfile(os.path.join(suite2p_dir_base,setup,subject,FOV,'Z-stacks',zstackname),
                                     os.path.join(temp_movie_directory,zstackname[:-4],zstackname))
                else:
                    zstackname = None
                s2p_params['z_stack_name'] =zstackname
                sp2_params_file = os.path.join(temp_movie_directory,'s2p_params.json')
                #%
                with open(sp2_params_file, "w") as data_file:
                    json.dump(s2p_params, data_file, indent=2)
                #% 
                if first_session_of_FOV: # first session needs a mean image to be generated
                    
                    
                    first_session_of_FOV = False
                    
                    reference_session_name = session
                else: #copy reference frame over from previous session
                    reference_movie_dir = os.path.join(temp_movie_directory,'_reference_image')
                    Path(reference_movie_dir).mkdir(parents = True,exist_ok = True)
                    reference_movie_directory = os.path.join(suite2p_dir_base,setup,subject,FOV,reference_session_name)
                
                    with open(os.path.join(reference_movie_directory,'filelist.json')) as f:
                        filelist_dict_ = json.load(f)
                    ops = np.load(os.path.join(reference_movie_directory,'ops.npy'),allow_pickle=True).tolist()
                    #z_plane_indices = np.argmax(ops['zcorr_list'],1)
                    z_plane_indices = filelist_dict_['zoff_mean_list'] ##HOTFIX - ops and filelist doesn't match ??
                    #print([len(z_plane_indices),len(z_plane_indices_2)])
                    needed_trials = z_plane_indices == np.median(z_plane_indices) #
                    meanimage_all = np.load(os.path.join(reference_movie_directory,'meanImg.npy'))
                    mean_img = np.mean(meanimage_all[needed_trials,:,:],0)
                    meanimage_dict = {'refImg':mean_img,
                                      'movies_used':np.asarray(filelist_dict_['file_name_list'])[needed_trials],
                                      'reference_session':reference_session_name}
                    np.save(os.path.join(reference_movie_dir,'mean_image.npy'),meanimage_dict) 
                    del meanimage_all, mean_img
                    
                    s2p_params['reference_session'] =reference_session_name
                    sp2_params_file = os.path.join(temp_movie_directory,'s2p_params.json')
                    #%
                    with open(sp2_params_file, "w") as data_file:
                        json.dump(s2p_params, data_file, indent=2)
                        
                reference_movie_dir = os.path.join(temp_movie_directory,'_reference_image')
                Path(reference_movie_dir).mkdir(parents = True,exist_ok = True)
                os.chmod(reference_movie_dir, 0o777)
                reference_movie_json = os.path.join(temp_movie_directory,'_reference_image','refimage_progress.json')
                refimage_dict = {'ref_image_started':True,
                                 'ref_image_finished':False,
                                 'ref_image_started_time': str(time.time())}
                with open(reference_movie_json, "w") as data_file:
                    json.dump(refimage_dict, data_file, indent=2)
                trial_num_to_use = int(trial_number_for_mean_image)
                #%
                try:
                    file_dict = np.load(os.path.join(temp_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
                except:
                    file_dict = {'copied_files':[]}
                while len(file_dict['copied_files'])<trial_num_to_use+1:
                    
                    print('waiting for {} trials to be available for generating reference frame -- already got {}'.format(trial_num_to_use,file_dict['copied_files']))
                    time.sleep(3)
                    try:
                        file_dict = np.load(os.path.join(temp_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
                    except:
                        file_dict = {'copied_files':[]}
                cluster_command_list = ['cd {}'.format(repo_location),
                                        'python cluster_helper.py {} "\'{}\'" {}'.format('utils_imaging.generate_mean_image_from_trials',temp_movie_directory,trial_num_to_use)]
                bash_command = r" && ".join(cluster_command_list)+ r" &"
                os.system(bash_command)
                print(bash_command)
                print('generating reference frame')
                #%% registering
                copy_finished = False
                all_files_registered = False
                while not copy_finished or not all_files_registered:
                    retry_loadin_file_dict = True
                    while retry_loadin_file_dict:# sometimes the other thread is still writing this file, so reading doesn't always happen
                        try:
                            file_dict = np.load(os.path.join(temp_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
                            retry_loadin_file_dict = False
                        except:
                            pass
                        
                    copy_finished = file_dict['copy_finished']
                    processes_running = 0
                    movies_registered = 0
                    files_running = []
                    for file in file_dict['copied_files']:
                        if not os.path.exists(os.path.join(temp_movie_directory,'mean_image.npy')):
                            print('no reference image!!')
                            time.sleep(3)
                            break
                        
                        dir_now = os.path.join(temp_movie_directory,file[:-4])
                        reg_json_file = os.path.join(temp_movie_directory,file[:-4],'reg_progress.json')
                        if 'reg_progress.json' in os.listdir(dir_now):
                            success = False
                            while not success:
                                try:
                                    with open(reg_json_file, "r") as read_file:
                                        reg_dict = json.load(read_file)
                                        success = True
                                except:
                                    print('json not ready, retrying..')
                                    pass # json file is not ready yet
        
                        else:
                            reg_dict = {'registration_started':False}
                        if 'registration_finished' in reg_dict.keys():
                            if reg_dict['registration_finished']:
                                movies_registered+=1
                                continue
                        if reg_dict['registration_started']:
                            processes_running+=1
                            files_running.append(file)
                            #print('{} is running'.format(file))
                            continue
        
                        cluster_command_list = ['cd {}'.format(repo_location),
                                                "python cluster_helper.py {} \"{}\" \"{}\"".format('utils_imaging.register_trial',temp_movie_directory,file)]
                        bash_command = r" && ".join(cluster_command_list)
                        if not copy_finished:
                            max_process_num_ = 2
                        else:
                            max_process_num_ = max_process_num
                        if processes_running < max_process_num_ :
                            if skip_photostim_trials:
                                skip_this_trial = False
                                for photo_name in photostim_name_list:
                                    if photo_name in file.lower():
                                        skip_this_trial = True
                                if skip_this_trial:
                                    continue
                            print('starting {}'.format(file))
                            reg_dict['registration_started'] = True
                            with open(reg_json_file, "w") as data_file:
                                json.dump(reg_dict, data_file, indent=2)
                            bash_command = bash_command + '&'
                            os.system(bash_command)
                            processes_running+=1
                            break # adding only a single thread at a time so they will do slightly different things..
                            
                        else:
                            break
                    photostim_trial_num = 0
                    for file in file_dict['copied_files']:
                        for photo_name in photostim_name_list:
                            if photo_name in file.lower():
                                photostim_trial_num+=1
                                continue
                    
                    if movies_registered == len(file_dict['copied_files']):
                        all_files_registered = True
                    elif skip_photostim_trials and movies_registered == len(file_dict['copied_files'])-photostim_trial_num:
                        all_files_registered = True
                    else:
                        all_files_registered = False
                    print('{} files registered out of {}, {} running processes : {}'.format( movies_registered,len(file_dict['copied_files']),processes_running, files_running))
                    time.sleep(3)
                #%% concatenating
                concatenated_movie_dir = os.path.join(temp_movie_directory,'_concatenated_movie')
                Path(concatenated_movie_dir).mkdir(parents = True,exist_ok = True)
                os.chmod(concatenated_movie_dir, 0o777 )
                utils_io.concatenate_suite2p_files(temp_movie_directory)
                #%%
                
                
                ops = np.load(os.path.join(concatenated_movie_dir,'ops.npy'),allow_pickle=True).tolist()
                z_plane_indices = np.argmax(ops['zcorr_list'],1)
                max_zcorr_vals = np.max(ops['zcorr_list'],1)
                min_zcorr_vals = np.min(ops['zcorr_list'],1)
                contrast = max_zcorr_vals/min_zcorr_vals
                needed_z = np.median(z_plane_indices)
                needed_trials = z_plane_indices == needed_z #
                with open(os.path.join(concatenated_movie_dir,'filelist.json')) as f:
                    filelist_dict = json.load(f)
                ops['nframes'] = sum(ops['nframes_list'])
                ops['reg_file'] = os.path.join(concatenated_movie_dir,'data.bin')
                ops['fs'] = np.mean(ops['fs_list'])
                bin_size = int(max(1, ops['nframes'] // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
                badframes = np.asarray(np.zeros(ops['nframes']),bool)
                frames_so_far = 0
                if minimum_contrast_for_binned_movie is None:
                    minimum_contrast= np.percentile(contrast,90)/2
                else:
                    minimum_contrast=minimum_contrast_for_binned_movie
                for framenum, filename,z,contrast_now in zip(filelist_dict['frame_num_list'],filelist_dict['file_name_list'],z_plane_indices,contrast):
                    bad_trial = False
                    for photo_name in photostim_name_list:
                        if photo_name in filename.lower():
                            bad_trial=True
                    if z > needed_z + acceptable_z_range_for_binned_movie or z < needed_z - acceptable_z_range_for_binned_movie or contrast_now<minimum_contrast:
                        bad_trial=True
                    if bad_trial:
                        badframes[frames_so_far:frames_so_far+framenum] = True
                    frames_so_far+=framenum
                    
                #%
                with BinaryFile(read_filename=ops['reg_file'], Ly=ops['Ly'], Lx=ops['Lx']) as f:
                    mov = f.bin_movie(
                        bin_size=bin_size,
                        bad_frames=badframes,
                        y_range=None,
                        x_range=None,
                    )
                np.save(os.path.join(concatenated_movie_dir,'binned_movie.npy'),mov)
                del mov
                #np.save(os.path.join(concatenated_movie_dir,'binned_movie_indices.npy'),np.where(badframes==False)[0])
                # archiving
                
                Path(archive_movie_directory).mkdir(parents = True,exist_ok = True)
                command_list = ['gsutil -m cp {} {}'.format(os.path.join(temp_movie_directory,'*.*'),archive_movie_directory),
                                'gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp {} {}'.format(os.path.join(temp_movie_directory,'_concatenated_movie','*.*'),archive_movie_directory_gs),
                                'gsutil -m cp {} {}'.format(os.path.join(temp_movie_directory,s2p_params['z_stack_name'][:-4],s2p_params['z_stack_name']),archive_movie_directory)]
                bash_command = r" && ".join(command_list)
                print(bash_command)
                os.system(bash_command)
                
                os.system('rm -r {}'.format(temp_movie_directory))
                
                if reference_is_previous_session:
                    reference_session_name = session
            
            
        
        

