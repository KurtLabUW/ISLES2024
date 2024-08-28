import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import cc3d
import os
import shutil

from sklearn.decomposition import PCA

AFFINE = np.array([[ -1.,   0.,   0.,  -0.],
                   [  0.,  -1.,   0., 239.],
                   [  0.,   0.,   1.,   0.],
                   [  0.,   0.,   0. ,  1.]])

def get_tissue_wise_seg(pred_mat, tissue_type):

    if tissue_type == 'WT':
        pred_mat_tissue_wise = pred_mat > 0
    elif tissue_type == 'TC':
        pred_mat_tissue_wise = np.logical_or(pred_mat == 1, pred_mat == 3)
    elif tissue_type == 'ET':
        pred_mat_tissue_wise = pred_mat == 3

    return pred_mat_tissue_wise.astype(np.uint16)

TISSUE_TYPES = ['WT', 'TC', 'ET']

## New processing strategy
'''
1. Remove ET dust
2. If ET dust created holes in TC, replace with NCR
3. Remvoe TC dust
4. If TC dust created holes in WT, replace with ED
5. Remove WT dust
'''

def rm_dust_fh(foldername, outfolder, convert):

    foldername = foldername.rstrip('/')
    # outfolder = f'{foldername}_pp'
    
    i=0
    for filename in os.listdir(foldername):
        if filename.split('.')[-1] != 'gz':
            continue

        print(i, filename)

        pathname = os.path.join(foldername, filename)

        pred_nii = nib.nifti1.load(pathname)
        pred_mat = pred_nii.get_fdata().astype(np.uint16)
        seg_new = np.zeros_like(pred_mat)

        if convert:
            seg_new[pred_mat == 3] = 3
            seg_new[pred_mat == 2] = 1
            seg_new[pred_mat == 1] = 2
            pred_mat = seg_new
            
        pred_mat_new = pred_mat.copy()
        
        thres = 50
            
        pred_mat_et = get_tissue_wise_seg(pred_mat_new, 'ET')
        pred_mat_et_rm_dust = cc3d.dust(pred_mat_et, threshold=thres, connectivity=26)
        rm_et_mask = np.logical_and(pred_mat_et==1, pred_mat_et_rm_dust==0)
        pred_mat_new[rm_et_mask] = 0

        pred_mat_tc = get_tissue_wise_seg(pred_mat_new, 'TC')
        tc_holes = 1 - pred_mat_tc
        tc_holes_rm = cc3d.dust(tc_holes, threshold=thres, connectivity=26)
        tc_filled = 1 - tc_holes_rm
        fill_ncr_mask = np.logical_and(tc_filled==1, pred_mat_new==0) * rm_et_mask
        pred_mat_new[fill_ncr_mask] = 1 #Fill holes with NCR

        pred_mat_tc = get_tissue_wise_seg(pred_mat_new, 'TC')
        pred_mat_tc_rm_dust = cc3d.dust(pred_mat_tc, threshold=thres, connectivity=26)
        rm_tc_mask = np.logical_and(pred_mat_tc==1, pred_mat_tc_rm_dust==0)
        pred_mat_new[rm_tc_mask] = 0

        pred_mat_wt = get_tissue_wise_seg(pred_mat_new, 'WT')
        wt_holes = 1 - pred_mat_wt
        wt_holes_rm = cc3d.dust(wt_holes, threshold=thres, connectivity=26)
        wt_filled = 1- wt_holes_rm
        fill_ed_mask = np.logical_and(wt_filled==1, pred_mat_new==0) * rm_tc_mask
        pred_mat_new[fill_ed_mask] = 2 #Fill holes with ED

        pred_mat_wt = get_tissue_wise_seg(pred_mat_new, 'WT')
        pred_mat_wt_rm_dust = cc3d.dust(pred_mat_wt, threshold=thres, connectivity=26)
        rm_wt_mask = np.logical_and(pred_mat_wt==1, pred_mat_wt_rm_dust==0)
        pred_mat_new[rm_wt_mask] = 0
        
        WT = np.sum(pred_mat_new != 0)
        ET = np.sum(pred_mat_new == 3)
        ED = np.sum(pred_mat_new == 2)
        if WT != 0:
            ETratio = ET/WT
            EDratio = ED/WT
        else:
            ETratio = 0
            EDratio = 0

        if ETratio < 0.03:
            pred_mat_new[pred_mat_new == 3] = 1

        if EDratio == 1:
            pred_mat_new[pred_mat_new == 2] = 1
        
        pred_nii_rm_dust = nib.nifti1.Nifti1Image(pred_mat_new, affine=AFFINE)
        nib.nifti1.save(pred_nii_rm_dust, os.path.join(outfolder, filename))

        i += 1

if __name__=='__main__':
    foldername= "/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0802/evaluation"
    outfolder= "/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0802/evaluation_pp"
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
    rm_dust_fh(foldername, outfolder, False)
    #shutil.make_archive('nnUNet5foldVal', 'zip', outfolder)
    
