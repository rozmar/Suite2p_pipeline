# Suite2p_pipeline
suite2p wrapper for multi-trial multi-session recordings
## Mount bucket
```
mkdir bucket
fusermount -u ./bucket
gcsfuse --implicit-dirs --only-dir marton.rozsa aind-transfer-service-test ./bucket
```

## Install
```
mkdir Scripts
cd Scripts
git clone https://github.com/rozmar/suite2p.git
cd suite2p
conda env create -f environment.yml
conda activate bci_with_suite2p
pip install -e .
cd ..
git clone https://github.com/rozmar/Suite2p_pipeline.git
cd Suite2p_pipeline 

```
## Initialize an already installed instance
```
cd ~
mkdir bucket
fusermount -u ./bucket
gcsfuse --implicit-dirs --only-dir marton.rozsa aind-transfer-service-test ./bucket
cd Scripts/suite2p/
git pull origin main
cd ..
cd Suite2p_pipeline
git pull origin main
conda activate bci_with_suite2p

```
## Register Z-stacks

```
python ./register_z_stacks.py /home/jupyter/temp/ /home/jupyter/bucket/Metadata/ /home/jupyter/bucket/Data/Calcium_imaging/raw/ /home/jupyter/bucket/Data/Calcium_imaging/suite2p/ BCI_29 Bergamo-2P-Photostim 
```

## Register sessions
```
python ./register_main.py /home/jupyter/temp/ /home/jupyter/bucket/Metadata/ /home/jupyter/bucket/Data/Calcium_imaging/raw/ /home/jupyter/bucket/Data/Calcium_imaging/suite2p/ BCI_29 Bergamo-2P-Photostim 4 50 FOV_02
```

## Segment traces
```
python ./segment_main.py /home/jupyter/temp/ /home/jupyter/bucket/Metadata/ /home/jupyter/bucket/Data/Calcium_imaging/raw/ /home/jupyter/bucket/Data/Calcium_imaging/suite2p/ BCI_29 Bergamo-2P-Photostim FOV_03_reference_is_1st_session None 1 true
```

## Extract traces
```
python ./extract_main.py /home/jupyter/temp/ /home/jupyter/bucket/Metadata/ /home/jupyter/bucket/Data/Calcium_imaging/raw/ /home/jupyter/bucket/Data/Calcium_imaging/suite2p/ /home/jupyter/bucket/Data/Behavior/BCI_exported/  BCI_29 Bergamo-2P-Photostim FOV_03 true
```
