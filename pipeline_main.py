import sys,os
import register, qc_segment,extract
try:
    import BCI_analysis
except:
    print('could not import BCI_analysis.. skipping')

subject = sys.argv[1]
fov = sys.argv[2]

bin_red_channel = False
use_red_channel = False
refine_ROIs = False

if sys.argv[3].lower() == 'refine-rois-register-export':
    register_z_stacks = True
    nonrigid = True
    register_sessions = True
    segment_cells = False
    overwrite_segmentation = False
    correlte_z_stacks = True
    export_traces = True
    overwrite_export = False
    register_photostim = True
    export_photostim = True
    export_photostim_apical_dendrites = False
    extract_photostim_groups = True
    overwrite_photostim_groups = True
    segment_mode = 'none'
    refine_ROIs = True
if sys.argv[3].lower() == 'refine-rois-register-export-overwrite':
    register_z_stacks = True
    nonrigid = True
    register_sessions = True
    segment_cells = False
    overwrite_segmentation = False
    correlte_z_stacks = True
    export_traces = True
    overwrite_export = True
    register_photostim = True
    export_photostim = True
    export_photostim_apical_dendrites = False
    extract_photostim_groups = True
    overwrite_photostim_groups = True
    segment_mode = 'none'
    refine_ROIs = True
if sys.argv[3].lower() == 'register-export':
    register_z_stacks = True
    nonrigid = True
    register_sessions = True
    segment_cells = False
    overwrite_segmentation = False
    correlte_z_stacks = True
    export_traces = True
    overwrite_export = False
    register_photostim = True
    export_photostim = True
    export_photostim_apical_dendrites = False
    extract_photostim_groups = True
    overwrite_photostim_groups = True
    segment_mode = 'none'
elif sys.argv[3].lower() == 'register-export-red':
    register_z_stacks = True
    nonrigid = True
    register_sessions = True
    segment_cells = False
    overwrite_segmentation = False
    correlte_z_stacks = True
    export_traces = True
    overwrite_export = False
    register_photostim = True
    export_photostim = True
    export_photostim_apical_dendrites = False
    extract_photostim_groups = True
    overwrite_photostim_groups = False
    segment_mode = 'none'
    bin_red_channel = True
    use_red_channel = True
elif sys.argv[3].lower() == 'axon-register-export':
    register_z_stacks = True
    nonrigid = True
    register_sessions = True
    segment_cells = False
    overwrite_segmentation = False
    correlte_z_stacks = True
    export_traces = True
    overwrite_export = False
    register_photostim = True
    export_photostim = True
    export_photostim_apical_dendrites = False
    extract_photostim_groups = True
    overwrite_photostim_groups = False
    segment_mode = 'none'
elif sys.argv[3].lower() == 'segment-only':
    register_z_stacks = False
    register_sessions = False
    nonrigid = False
    segment_cells = True
    overwrite_segmentation = True
    correlte_z_stacks = False
    export_traces = False
    overwrite_export = False
    register_photostim = False
    export_photostim = False
    export_photostim_apical_dendrites = False
    extract_photostim_groups = False
    overwrite_photostim_groups = False
    segment_mode = 'soma'
elif sys.argv[3].lower() == 'axon-segment-only':
    register_z_stacks = False
    register_sessions = False
    nonrigid = False
    segment_cells = True
    overwrite_segmentation = True
    correlte_z_stacks = False
    export_traces = False
    overwrite_export = False
    register_photostim = False
    export_photostim = False
    export_photostim_apical_dendrites = False
    extract_photostim_groups = False
    overwrite_photostim_groups = False
    segment_mode = 'axon'
elif sys.argv[3].lower() == 'export-overwrite':
    register_z_stacks = False
    register_sessions = False
    nonrigid = True
    segment_cells = False
    overwrite_segmentation = False
    correlte_z_stacks = False
    export_traces = True
    overwrite_export = True
    register_photostim = False
    export_photostim = False
    export_photostim_apical_dendrites = False
    extract_photostim_groups = False
    overwrite_photostim_groups = False
    segment_mode = 'none'
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

if register_sessions:
    register.register_session(local_temp_dir = local_temp_dir,
                              metadata_dir = metadata_dir,
                              raw_scanimage_dir_base =raw_scanimage_dir_base,
                              suite2p_dir_base = suite2p_dir_base,
                              subject = subject,
                              setup = setup,
                              max_process_num = 4,
                              batch_size = 50,
                              FOV_needed = fov,
                              nonrigid = nonrigid,
                              nonrigid_smooth_sigma_time = 1,
                              bin_red_channel = bin_red_channel)
if segment_cells or correlte_z_stacks:
    qc_segment.qc_segment(local_temp_dir = local_temp_dir,
                          metadata_dir = metadata_dir,
                          raw_scanimage_dir_base =raw_scanimage_dir_base,
                          suite2p_dir_base = suite2p_dir_base,
                          subject = subject,
                          setup = setup,
                          fov = fov,
                          minimum_contrast = 3,
                          acceptable_z_range = 3,#1 originally
                          segment_cells = segment_cells,
                          overwrite_segment = overwrite_segmentation,
                          correlte_z_stacks =correlte_z_stacks,
                         segment_mode =segment_mode) 
if refine_ROIs:
    qc_segment.refine_ROIS(suite2p_dir_base = suite2p_dir_base,
                           subject = subject,
                           setup = setup,
                           fov = fov,
                           overwrite = overwrite_segmentation,
                           allow_overlap = True,
                           use_cellpose = False,
                           denoise_detect = False)
if export_traces:

    extract.extract_traces(local_temp_dir = local_temp_dir,
                          metadata_dir = metadata_dir,
                          raw_scanimage_dir_base =raw_scanimage_dir_base,
                          suite2p_dir_base = suite2p_dir_base,
                          bpod_path = bpod_path,
                          subject = subject,
                          setup = setup,
                          fov = fov,
                          overwrite = overwrite_export,
                          roi_types = [''],
                          photostim = False,
                          use_red_channel =use_red_channel)


    # BCI_analysis.io_suite2p.suite2p_to_npy(os.path.join(suite2p_dir_base,setup), 
    #                                        os.path.join(raw_scanimage_dir_base,setup), 
    #                                        os.path.join(bpod_path,setup),
    #                                        save_path, 
    #                                        overwrite=overwrite_export, 
    #                                        mice_name = subject,
    #                                        fov_list = [fov],
    #                                        max_frames = 500)
if register_photostim:
    register.register_photostim(local_temp_dir = local_temp_dir,
                              metadata_dir = metadata_dir,
                              raw_scanimage_dir_base =raw_scanimage_dir_base,
                              suite2p_dir_base = suite2p_dir_base,
                              subject = subject,
                              setup = setup,
                              max_process_num = 4,
                              batch_size = 50,
                              FOV_needed = fov,
                              nonrigid = nonrigid)


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