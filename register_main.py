
# this script registers trials and sessions together in a chronological order
#%% import libraries and set parameters
import pandas as pd
import numpy as np
import threading
from pathlib import Path
import os
import json
import time
import shutil
from utils import utils_io
import sys


repo_location = '/home/jupyter/Scripts/Suite2p_pipeline'
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
    max_process_num = sys.argv[7]#'Bergamo-2P-Photostim'
except:
    max_process_num = 3
# =============================================================================
# local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/'
# metadata_dir = '/mnt/Data/BCI_metadata/'
# raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
# suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
# subject = 'BCI_29'
# setup = 'Bergamo-2P-Photostim'
# =============================================================================

s2p_params = {'max_reg_shift':50, # microns
            'max_reg_shift_NR': 20, # microns
            'block_size': 200, # microns
            'smooth_sigma':0.5, # microns
            'smooth_sigma_time':0, #seconds,
            'overwrite': False,
            #'num_workers':4,
            'z_stack_name':'',
            'reference_session':''} # folder where the suite2p output is saved

trial_number_for_mean_image = 10
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
        session_date_dict[session_date.date()] = [session_date_dict[session_date.date()],session]
    else:
        session_date_dict[session_date.date()] = session

FOV_list_ = np.unique(np.asarray(subject_metadata['FOV'].values,str))
FOV_list = []
for FOV in FOV_list_:
    if FOV != '-' and FOV != 'nan' and len(FOV)>0:
        FOV_list.append(FOV)

for FOV in FOV_list:
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
        session = session_date_dict[session_date]
        # start copying files to local drive
        source_movie_directory = os.path.join(raw_scanimage_dir_base,setup,subject,session)
        temp_movie_directory = os.path.join(local_temp_dir,'{}_{}'.format(subject,session))
        copy_thread = threading.Thread(target = utils_io.copy_tiff_files_in_order, args = (source_movie_directory,temp_movie_directory))
        copy_thread.start()
        #% select the first Z-stack in the FOV directory
        available_z_stacks = np.sort(os.listdir(os.path.join(suite2p_dir_base,setup,subject,FOV,'Z-stacks')))
        for zstackname in available_z_stacks:
            if '.tif' in zstackname:
                Path(os.path.join(temp_movie_directory,zstackname[:-4])).mkdir(parents = True,exist_ok = True)
                shutil.copyfile(os.path.join(suite2p_dir_base,setup,subject,FOV,'Z-stacks',zstackname),
                                 os.path.join(temp_movie_directory,zstackname[:-4],zstackname))
                continue
            else:
                zstackname = None
        s2p_params['z_stack_name'] =zstackname
        sp2_params_file = os.path.join(temp_movie_directory,'s2p_params.json')
        #%
        with open(sp2_params_file, "w") as data_file:
            json.dump(s2p_params, data_file, indent=2)
        #% 
        if first_session_of_FOV: # first session needs a mean image to be generated
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
            first_session_of_FOV = False

        #%% registering
        copy_finished = False
        all_files_registered = False
        while not copy_finished or not all_files_registered:
            file_dict = np.load(os.path.join(temp_movie_directory,'copy_data.npy'),allow_pickle = True).tolist()
            copy_finished = file_dict['copy_finished']
            processes_running = 0
            movies_registered = 0
            for file in file_dict['copied_files']:
                if not os.path.exists(os.path.join(temp_movie_directory,'mean_image.npy')):
                    print('no reference image!!')
                    time.sleep(3)
                    break
                
                dir_now = os.path.join(temp_movie_directory,file[:-4])
                reg_json_file = os.path.join(temp_movie_directory,file[:-4],'reg_progress.json')
                if 'reg_progress.json' in os.listdir(dir_now):
                    with open(reg_json_file, "r") as read_file:
                        reg_dict = json.load(read_file)
                else:
                    reg_dict = {'registration_started':False}
                if 'registration_finished' in reg_dict.keys():
                    if reg_dict['registration_finished']:
                        movies_registered+=1
                        continue
                if reg_dict['registration_started']:
                    processes_running+=1
                    continue

                cluster_command_list = ['cd {}'.format(repo_location),
                                        "python cluster_helper.py {} \"{}\" \"{}\"".format('utils_imaging.register_trial',temp_movie_directory,file)]
                bash_command = r" && ".join(cluster_command_list)
                if processes_running < max_process_num :
                    print('starting {}'.format(file))
                    reg_dict['registration_started'] = True
                    with open(reg_json_file, "w") as data_file:
                        json.dump(reg_dict, data_file, indent=2)
                    bash_command = bash_command + '&'
                    os.system(bash_command)
                    processes_running+=1
                    
                else:
                    break
            if movies_registered == len(file_dict['copied_files']):
                all_files_registered = True
            else:
                all_files_registered = False
            print('{} running processes, {} files registered out of {}'.format(processes_running, movies_registered,len(file_dict['copied_files'])))
            time.sleep(3)
        #%% concatenating
        concatenated_movie_dir = os.path.join(temp_movie_directory,'_concatenated_movie')
        Path(concatenated_movie_dir).mkdir(parents = True,exist_ok = True)
        os.chmod(concatenated_movie_dir, 0o777 )
        utils_io.concatenate_suite2p_files(temp_movie_directory)
        # archiving
        archive_movie_directory = os.path.join(suite2p_dir_base,setup,subject,FOV,session)
        Path(archive_movie_directory).mkdir(parents = True,exist_ok = True)
        command_list = ['cp {} {}'.format(os.path.join(temp_movie_directory,'*.*'),archive_movie_directory),
                        'cp {} {}'.format(os.path.join(temp_movie_directory,'_concatenated_movie','*.*'),archive_movie_directory),
                        'cp {} {}'.format(os.path.join(temp_movie_directory,s2p_params['z_stack_name'][:-4],s2p_params['z_stack_name']),archive_movie_directory)]
        bash_command = r" && ".join(command_list)
        print(bash_command)
        os.system(bash_command)
        
        os.system('rm -r {}'.format(temp_movie_directory))
        
        
#%% go through raw sessions in the bucket and see if they are already registered, if not, start registering on the local hdd

