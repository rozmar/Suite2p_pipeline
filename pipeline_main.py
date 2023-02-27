import sys,os
import register, qc_segment,extract
try:
    import BCI_analysis
except:
    print('could not import BCI_analysis.. skipping')

subject = sys.argv[1]
fov = sys.argv[2]
register_z_stacks = 'true' in sys.argv[3].lower()
regiter_sessions = 'true' in sys.argv[4].lower()
resegment_cells = 'true' in sys.argv[5].lower()
correlte_z_stacks = 'true' in sys.argv[6].lower()
overwrite_export = 'true' in sys.argv[7].lower()
register_photostim = 'true' in sys.argv[8].lower()
export_photostim = 'true' in sys.argv[9].lower()
export_photostim_apical_dendrites = 'true' in sys.argv[10].lower()
try:
    extract_photostim_groups = 'true' in sys.argv[11].lower()
except:
    extract_photostim_groups = False
try:
    overwrite_photostim_groups = 'true' in sys.argv[12].lower()
except:
    overwrite_photostim_groups = False

# - HARD-CODED VARIABLES FOR GOOGLE CLOUD
local_temp_dir = '/home/jupyter/temp/' 
metadata_dir = '/home/jupyter/bucket/Metadata/' 
raw_scanimage_dir_base ='/home/jupyter/bucket/Data/Calcium_imaging/raw/' 
suite2p_dir_base = '/home/jupyter/bucket/Data/Calcium_imaging/suite2p/'
bpod_path = '/home/jupyter/bucket/Data/Behavior/BCI_exported/'
setup = 'Bergamo-2P-Photostim'#'DOM3-MMIMS'
save_path = "/home/jupyter/bucket/Data/Calcium_imaging/sessionwise_tba"
# - HARD-CODED VARIABLES FOR GOOGLE CLOUD
if register_z_stacks:
    try:
        register.register_z_stacks(local_temp_dir = local_temp_dir,
                                  metadata_dir = metadata_dir,
                                  raw_scanimage_dir_base =raw_scanimage_dir_base,
                                  suite2p_dir_base = suite2p_dir_base,
                                  subject_ = subject,
                                  setup = setup)
    except:
        print('could not register z-stack, trying averaging')
        register.register_z_stacks(local_temp_dir = local_temp_dir,
                                  metadata_dir = metadata_dir,
                                  raw_scanimage_dir_base =raw_scanimage_dir_base,
                                  suite2p_dir_base = suite2p_dir_base,
                                  subject_ = subject,
                                  setup = setup,
                                  method = 'averaging')

if regiter_sessions:
    register.register_session(local_temp_dir = local_temp_dir,
                              metadata_dir = metadata_dir,
                              raw_scanimage_dir_base =raw_scanimage_dir_base,
                              suite2p_dir_base = suite2p_dir_base,
                              subject = subject,
                              setup = setup,
                              max_process_num = 4,
                              batch_size = 50,
                              FOV_needed = fov)
if resegment_cells or correlte_z_stacks:
    qc_segment.qc_segment(local_temp_dir = local_temp_dir,
                          metadata_dir = metadata_dir,
                          raw_scanimage_dir_base =raw_scanimage_dir_base,
                          suite2p_dir_base = suite2p_dir_base,
                          subject = subject,
                          setup = setup,
                          fov = fov,
                          minimum_contrast = 3,
                          acceptable_z_range = 1,
                          segment_cells = resegment_cells,
                          correlte_z_stacks =correlte_z_stacks) 

extract.extract_traces(local_temp_dir = local_temp_dir,
                      metadata_dir = metadata_dir,
                      raw_scanimage_dir_base =raw_scanimage_dir_base,
                      suite2p_dir_base = suite2p_dir_base,
                      bpod_path = bpod_path,
                      subject = subject,
                      setup = setup,
                      fov = fov,
                      overwrite = overwrite_export)


    

BCI_analysis.io_suite2p.suite2p_to_npy(os.path.join(suite2p_dir_base,setup), 
                                       os.path.join(raw_scanimage_dir_base,setup), 
                                       os.path.join(bpod_path,setup),
                                       save_path, 
                                       overwrite=overwrite_export, 
                                       mice_name = subject,
                                       fov_list = [fov],
                                       max_frames = 500)
if register_photostim:
    register.register_photostim(local_temp_dir = local_temp_dir,
                              metadata_dir = metadata_dir,
                              raw_scanimage_dir_base =raw_scanimage_dir_base,
                              suite2p_dir_base = suite2p_dir_base,
                              subject = subject,
                              setup = setup,
                              max_process_num = 4,
                              batch_size = 50,
                              FOV_needed = fov)


if export_photostim:
    if export_photostim_apical_dendrites:
        roi_types = ['','_rest']
    else:
        roi_types= ['']
        
    extract.extract_traces(local_temp_dir = local_temp_dir,
                          metadata_dir = metadata_dir,
                          raw_scanimage_dir_base =raw_scanimage_dir_base,
                          suite2p_dir_base = suite2p_dir_base,
                          bpod_path = bpod_path,
                          subject = subject,
                          setup = setup,
                          fov = fov,
                          overwrite = overwrite_export,
                          roi_types = roi_types,
                          photostim = True)
if extract_photostim_groups:
    extract.extract_photostim_groups(subject=subject,
                                 FOV=fov,
                                 setup=setup,
                                 raw_movie_basedir=raw_scanimage_dir_base,
                                 suite2p_basedir=suite2p_dir_base,
                                     overwrite = overwrite_photostim_groups)