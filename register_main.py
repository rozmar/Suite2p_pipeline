
# this script registers trials and sessions together in a chronological order
#%% import libraries and set parameters
import pandas as pd
import numpy as np
import threading
from pathlib import Path
import os
import shutil
from utils import utils_imaging, utils_io
import sys

# =============================================================================
# local_temp_dir = sys.argv[1]#'/mnt/HDDS/Fast_disk_0/temp/'
# metadata_dir = sys.argv[2]#'/mnt/Data/BCI_metadata/'
# raw_scanimage_dir_base = sys.argv[3]#'/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
# suite2p_dir_base = sys.argv[4]#'/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
# try:
#     subject = sys.argv[5]#'BCI_29'
# except:
#     subject = None
# try:
#     setup = sys.argv[6]#'Bergamo-2P-Photostim'
# except:
#     setup = None
# =============================================================================

local_temp_dir = '/mnt/HDDS/Fast_disk_0/temp/'
metadata_dir = '/mnt/Data/BCI_metadata/'
raw_scanimage_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/raw/'
suite2p_dir_base = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/'
subject = 'BCI_29'
setup = 'Bergamo-2P-Photostim'

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
        #utils_io.copy_tiff_files_in_order(source_movie_directory,temp_movie_directory) # this script copies all the data locally
        asd
        
        
        
        
#%% go through raw sessions in the bucket and see if they are already registered, if not, start registering on the local hdd

