
# this script registers trials and sessions together in a chronological order
#%% import libraries and set parameters
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
from utils import utils_imaging
import sys

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

#% metadata will be updated manually, this script will read .csv files
subject_metadata = pd.read_csv(os.path.join(metadata_dir,subject.replace('_','')+'.csv'))
#%% export Z-stacks
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
    z_stacks = subject_metadata.loc[subject_metadata['FOV']==FOV,'Z-stack']
    sessions = subject_metadata.loc[subject_metadata['FOV']==FOV,'Date']
    for z_stack,session in zip(z_stacks,sessions):
        session = datetime.datetime.strptime(session,'%Y/%m/%d').date()
        if type(z_stack) is not str:
            continue
        z_stack_dir = os.path.join(suite2p_dir_base,setup,subject,FOV,'Z-stacks')
        Path(z_stack_dir).mkdir(exist_ok = True, parents = True)
        z_stack_save_name = '{}_{}.tif'.format(session,z_stack)
        if z_stack_save_name in os.listdir(z_stack_dir): 
            continue #already done
        tiff_files_in_raw_folder = os.listdir(os.path.join(raw_scanimage_dir_base,setup,subject,session_date_dict[session]))
        if z_stack in tiff_files_in_raw_folder:
            temp_dir = os.path.join(local_temp_dir,'{}_{}_{}'.format(subject,session_date_dict[session],z_stack[:-4]))
            Path(temp_dir).mkdir(exist_ok = True, parents = True)
            utils_imaging.register_zstack(os.path.join(raw_scanimage_dir_base,setup,subject,session_date_dict[session],z_stack)
                                          ,temp_dir)
            new_zstack_name = '{}_{}_{}'.format(subject,session_date_dict[session],z_stack[:-4])
            shutil.copyfile(os.path.join(temp_dir,new_zstack_name),os.path.join(z_stack_dir,new_zstack_name))