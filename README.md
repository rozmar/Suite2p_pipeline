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
mkdir scripts
cd scripts
git clone https://github.com/rozmar/suite2p.git
cd suite2p
conda env create -f environment.yml -y
conda activate bci_with_suite2p
pip install -e .
cd ..
git clone https://github.com/rozmar/Suite2p_pipeline.git
cd Suite2p_pipeline 
python ./register_z_stacks.py /home/jupyter/temp/ /home/jupyter/bucket/Metadata/ /home/jupyter/bucket/Data/Calcium_imaging/raw/ /home/jupyter/bucket/Data/Calcium_imaging/suite2p/ BCI_29 Bergamo-2P-Photostim 

```
