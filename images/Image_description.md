# image information

A description of the images of the in vivo flip angle comparison.

## 3mm_inv
The in vivo 3 mm images obtained from subject 1 (see Table 1 in the paper).

The images dimensions are as follows: FOV = 216 x 216 x 216 mm³, isotropic resolution = 3 mm, matrix size = 72 x 72 x 72.

The file names are:
- AfiFa.nii.gz: Flip angle map of the AFI method
- AfiMag.nii.gz: Magnitude image of the AFI method
- Cart3Dfa_reg.nii.gz: Flip angle map of the Cartesian 3DREAM sequence, registered to the magnitude image of the AFI method
- Cart3Dfid_reg.nii.gz: Combined contrasts (STE* and FID) of the Cartesian 3DREAM sequence, registered to the magnitude image of the AFI method
- mask_3mm.nii.gz: brain mask of the 3 mm AFI magnitude image
- Spir3Dfa_reg.nii.gz: Flip angle map of the spiral 3DREAM sequence, registered to the magnitude image of the AFI method
- Spir3Dfid_reg.nii.gz: FID of the spiral 3DREAM sequence, registered to the magnitude image of the AFI method
- Spir3Dste_reg.nii.gz: STE* of the spiral 3DREAM sequence, registered to the magnitude image of the AFI method

## 5mm_inv
The in vivo 5 mm images obtained from subject 1 (see Table 1 in the paper).

The images dimensions are as follows: FOV = 200 x 200 x 200 mm³, isotropic resolution = 5 mm, matrix size = 40 x 40 x 40.

The file names correspond to the ones of the 3 mm images, but with a brain mask of the 5 mm AFI magnitude image.
