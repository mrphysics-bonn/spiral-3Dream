# Spiral 3DREAM

This repository contains Pulseq sequence files, scanner raw data and reconstructed images from the paper "Spiral 3DREAM sequence for fast whole-brain B1 Mapping".

The "seq" folder holds the source code of the spiral 3DREAM sequence ("write_spiral_3dream.py"). The required dependencies can be installed into a new Python environment using the supplied yml file:
```
conda env create -f dream.yml
```
Additionally, Pulseq sequence files for 5 mm isotropic resolution (phantom) and 3/5 mm isotropic resolution (in vivo) can be found in the subfolder "pulseq_files".

The "images" folder holds in vivo spiral & Cartesian 3DREAM images including FA maps, FID and STE images. Additionally FA maps from an AFI sequence are available. Additional information on the the image data can be found in the file "Image_description" in the "images" folder.

The reconstruction pipeline can be found at https://github.com/mrphysics-bonn/python-ismrmrd-reco. Contact the author for more information on the pipeline.

## Author

marten.veldmann@dzne.de
