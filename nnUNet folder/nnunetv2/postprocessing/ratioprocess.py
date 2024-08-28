import nibabel as nib
import numpy as np
import scipy
import scipy.ndimage
import cc3d
import os
import shutil
from multiprocessing import Process

AFFINE = np.array([[ -1.,   0.,   0.,  -0.],
                   [  0.,  -1.,   0., 239.],
                   [  0.,   0.,   1.,   0.],
                   [  0.,   0.,   0. ,  1.]])

def rm_dust_fh(foldername, outfolder):

    foldername = foldername.rstrip('/')

    i=0
    for filename in os.listdir(foldername):
        if filename.split('.')[-1] != 'gz':
            continue

        print(i, filename)

        pathname = os.path.join(foldername, filename)

        pred_nii = nib.nifti1.load(pathname)
        pred_mat = pred_nii.get_fdata().astype(np.uint16)
        seg_new = np.zeros_like(pred_mat)
        seg_new[pred_mat == 2] = 3
        seg_new[pred_mat == 1] = 1
        pred_mat = seg_new
        
        # WT = np.sum(pred_mat != 0)
        # ET = np.sum(pred_mat == 3)
        # ED = np.sum(pred_mat == 2)
        # if WT != 0:
        #     ETratio = ET/WT
        #     EDratio = ED/WT
        # else:
        #     ETratio = 0
        #     EDratio = 0

        # if ETratio < 0.03:
        #     pred_mat[pred_mat == 3] = 1

        # if EDratio == 1:
        #     pred_mat[pred_mat == 2] = 1
        
        pred_nii_rm_dust = nib.nifti1.Nifti1Image(pred_mat, affine=AFFINE)
        nib.nifti1.save(pred_nii_rm_dust, os.path.join(outfolder, filename))

        i += 1

if __name__=='__main__':
    foldername= f"/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0801/inference2"
    outfolder= f"/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0801/inference2_pp"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    rm_dust_fh(foldername, outfolder)
    #shutil.make_archive(os.path.join(outfolder, '5fold_ratio_noET'), 'zip', outfolder)
        
        
