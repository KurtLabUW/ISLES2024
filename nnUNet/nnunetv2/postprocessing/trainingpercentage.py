import os
import nibabel as nib
import numpy as np
 
AFFINE = np.array([[ -1.,   0.,   0.,  -0.],
                   [  0.,  -1.,   0., 239.],
                   [  0.,   0.,   1.,   0.],
                   [  0.,   0.,   0. ,  1.]]) 

sourcefolder = '/mmfs1/gscratch/scrubbed/hitender/val/fold-4'
outfolder = '/mmfs1/gscratch/scrubbed/hitender/val_labeled/fold-4'

i=0
for filename in os.listdir(sourcefolder):
    if filename.split('.')[-1] != 'gz':
        continue

    print(i, filename)

    pathname = os.path.join(sourcefolder, filename)

    pred_nii = nib.nifti1.load(pathname)
    pred_mat = pred_nii.get_fdata().astype(np.uint16)
    seg_new = np.zeros_like(pred_mat)
    seg_new[pred_mat == 3] = 3
    seg_new[pred_mat == 2] = 1
    seg_new[pred_mat == 1] = 2
    pred_mat = seg_new
    
    pred_nii_rm_dust = nib.nifti1.Nifti1Image(pred_mat, affine=AFFINE)
    nib.nifti1.save(pred_nii_rm_dust, os.path.join(outfolder, filename))
    i=i+1