# Spiral 3DREAM

This repository contains Pulseq sequence files and reconstructed images from the paper "Spiral 3DREAM sequence for fast whole-brain B1 Mapping".

The "seq" folder holds the source code of the spiral 3DREAM sequence ("write_spiral_3dream.py"). The required dependencies can be installed into a new Python environment using the supplied yml file:
```
conda env create -f dream.yml
```
Additionally, Pulseq sequence files for 2/3/4/5 mm isotropic resolution can be found in the subfolder "pulseq_files".

The "images" folder holds spiral & Cartesian 3DREAM images including FA maps, FID (filtered/not filtered) and STE images. Additionally FA maps from an AFI sequence are available.

The reconstruction pipeline can be found at https://github.com/mrphysics-bonn/python-ismrmrd-reco. Contact the author for more information on the pipeline.

## Author

marten.veldmann@dzne.de
