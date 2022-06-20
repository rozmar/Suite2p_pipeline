import sys,os
from pipeline import register, qc_segment,extract
import BCI_analysis

subject = sys.argv[1]
fov = sys.argv[2]
resegment_cells = 'true' in sys.argv[3].lower()


# - HARD-CODED VARIABLES FOR GOOGLE CLOUD
local_temp_dir = '/home/jupyter/temp/' 
metadata_dir = '/home/jupyter/bucket/Metadata/' 
raw_scanimage_dir_base ='/home/jupyter/bucket/Data/Calcium_imaging/raw/' 
suite2p_dir_base = '/home/jupyter/bucket/Data/Calcium_imaging/suite2p/'
bpod_path = '/home/jupyter/bucket/Data/Behavior/BCI_exported/'
setup = 'Bergamo-2P-Photostim'
save_path = "/home/jupyter/bucket/Data/Calcium_imaging/sessionwise_tba"
# - HARD-CODED VARIABLES FOR GOOGLE CLOUD

register.register_z_stacks(local_temp_dir = local_temp_dir,
                          metadata_dir = metadata_dir,
                          raw_scanimage_dir_base =raw_scanimage_dir_base,
                          suite2p_dir_base = suite2p_dir_base,
                          subject_ = subject,
                          setup = setup)


register.register_session(local_temp_dir = local_temp_dir,
                          metadata_dir = metadata_dir,
                          raw_scanimage_dir_base =raw_scanimage_dir_base,
                          suite2p_dir_base = suite2p_dir_base,
                          subject_ = subject,
                          setup = setup,
                          max_process_num = 4,
                          batch_size = 50,
                          FOV_needed = fov)

qc_segment.qc_segment(local_temp_dir = local_temp_dir,
                      metadata_dir = metadata_dir,
                      raw_scanimage_dir_base =raw_scanimage_dir_base,
                      suite2p_dir_base = suite2p_dir_base,
                      subject_ = subject,
                      setup = setup,
                      fov = fov,
                      minimum_contrast = None,
                      acceptable_z_range = 1,
                      segment_cells = resegment_cells)

extract.extract_traces(local_temp_dir = local_temp_dir,
                      metadata_dir = metadata_dir,
                      raw_scanimage_dir_base =raw_scanimage_dir_base,
                      suite2p_dir_base = suite2p_dir_base,
                      bpod_path = bpod_path,
                      subject = subject,
                      setup = setup,
                      fov = fov,
                      overwrite = resegment_cells)

BCI_analysis.io_suite2p.suite2p_to_npy(os.path.join(suite2p_dir_base,setup), 
                                       os.path.join(raw_scanimage_dir_base,setup), 
                                       os.path.join(bpod_path,setup),
                                       save_path, 
                                       overwrite=resegment_cells, 
                                       mice_name = subject,
                                       fov_list = fov)