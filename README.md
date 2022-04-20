# Spiral 3DREAM sequence and reconstruction

This repository contains files to reproduce results from the 2022 ISMRM abstract #219 "Spiral 3DREAM sequence for fast whole-brain B1 Mapping"

## Overview

The repository contains several files to reproduce the results of the abstract:
- The Pulseq spiral 3DREAM sequence file: "20210721_3Dream_beta5.seq".
- Raw data collected with the sequence on a 7T Siemens Magnetom scanner in the MRD format: file "rawdata_spiral3dream.h5". 
- Reconstructed images stored in the MRD image format: "images_spiral3dream.h5". 
- Images converted to Nifti are in the "nifti" folder. The folder contains STE and FID images and the resulting flip angle map in degrees times factor 10.

For running the reconstruction pipeline, a Python environment is required, which can be installed by using the provided yml file: `conda env create -f ismrmrd_client.yml`. Also, a Docker installation is required. More information on the reconstruction server can be found at https://github.com/mrphysics-bonn/python-ismrmrd-reco.

Reproducing the reconstruction:
1. To reproduce the reconstruction, the reconstruction server has to be pulled from DockerHub: `docker pull mavel101/dream_reco`.
2. Start the server by running `./start_docker` from this directory.
3. Activate the conda environment: `conda activate ismrmrd_client`.
4. Run the reconstruction with the command `./send_data_pulseq.sh rawdata_spiral3dream.h5 out.h5`. The result will be saved in the file out.h5 in MRD image format.

The reconstruction pipelines source code can be found in the submodule "python-ismrmrd-server" (file "bart_pulseq_spiral_dream.py").

## Author

Questions & feedback go to marten.veldmann@dzne.de
