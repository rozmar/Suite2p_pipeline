# Suite2p_pipeline
suite2p wrapper for multi-trial multi-session recordings

## Install
```
cd ~
mkdir bucket
fusermount -u ./bucket
gcsfuse --implicit-dirs --only-dir marton.rozsa aind-transfer-service-test ./bucket
mkdir Scripts
cd Scripts
git clone https://github.com/rozmar/suite2p.git
cd suite2p
conda env create -f environment.yml
conda activate bci_with_suite2p
pip install -e .
cd ..
git clone https://github.com/kpdaie/BCI_analysis.git
cd BCI_analysis
pip install -e .
cd ..
git clone https://github.com/rozmar/Suite2p_pipeline.git
cd Suite2p_pipeline 
pip install opencv-python
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
cd BCI_analysis
git pull origin main
cd ..
cd Suite2p_pipeline
git pull origin main
conda activate bci_with_suite2p

```
## Run pipeline
### Registration, segmentation, extraction, everything with photostim
```
python ./pipeline_main.py BCI_29 FOV_03 true true true true true true
```
### Register and extract only a new session, no repeated segmentation with photostim
```
python ./pipeline_main.py BCI_29 FOV_03 true true false true false true
```
### Register photostim
```
python ./pipeline_main.py BCI_29 FOV_03 false false false false false true
```
## Add BCI_analysis
```
cd ~
cd Scripts
conda activate bci_with_suite2p
git clone https://github.com/kpdaie/BCI_analysis.git
cd BCI_analysis
pip install -e .


```
## add environment kernel for jupyter notebook
```
conda activate bci_with_suite2p
python -m ipykernel install --user --name=bci_with_suite2p
```
