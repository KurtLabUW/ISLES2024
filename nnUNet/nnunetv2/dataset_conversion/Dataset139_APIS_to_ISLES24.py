import multiprocessing
import shutil
from multiprocessing import Pool
import argparse

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import nibabel as nib
# AFFINE = np.array([[ -1.,   0.,   0.,  -0.],
#                    [  0.,  -1.,   0., 239.],
#                    [  0.,   0.,   1.,   0.],
#                    [  0.,   0.,   0. ,  1.]])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
       description="Use this to convert APIS datasets")
    parser.add_argument("path", type=str, help='path to dataset folder')
    parser.add_argument("id", type=int, help='task id')
    parser.add_argument("name", type=str, help='task name')
    args = parser.parse_args()
    
    data_dir = args.path
    task_id = args.id
    task_name = args.name
    #brats_data_dir = "/mmfs1/gscratch/kurtlab/brats2024/data/brats-ssa/validation/BraTS2024-SSA-Challenge-ValidationData"
    #task_id = 143
    #task_name = "BraTS2024-SSAVal"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    total = 0
    case_ids = subdirs(data_dir, join=False)
    total += len(case_ids)
    for c in case_ids:
        
        pred_nii = nib.nifti1.load(join(data_dir, c, c + "_ncct.nii.gz"))
        AFFINE = pred_nii.affine
        pred_mat = pred_nii.get_fdata().astype(np.uint16)
        seg_new = np.zeros_like(pred_mat)
        pred_nii_rm_dust = nib.nifti1.Nifti1Image(seg_new, affine=AFFINE)
        nib.nifti1.save(pred_nii_rm_dust, join(imagestr, c + "_0000.nii.gz"))
        nib.nifti1.save(pred_nii_rm_dust, join(imagestr, c + "_0001.nii.gz"))
        nib.nifti1.save(pred_nii_rm_dust, join(imagestr, c + "_0002.nii.gz"))
        nib.nifti1.save(pred_nii_rm_dust, join(imagestr, c + "_0003.nii.gz"))
        
        shutil.copy(join(data_dir, c, c + "_ncct.nii.gz"), join(imagestr, c + '_0004.nii.gz'))
        #shutil.copy("/mmfs1/gscratch/kurtlab/brats2024/data/APIS/cbf.nii.gz", join(imagestr, c + "_0000.nii.gz"))
        #shutil.copy("/mmfs1/gscratch/kurtlab/brats2024/data/APIS/cbv.nii.gz", join(imagestr, c + "_0001.nii.gz"))
        #shutil.copy("/mmfs1/gscratch/kurtlab/brats2024/data/APIS/mtt.nii.gz", join(imagestr, c + "_0002.nii.gz"))
        #shutil.copy("/mmfs1/gscratch/kurtlab/brats2024/data/APIS/tmax.nii.gz", join(imagestr, c + "_0003.nii.gz"))
        
        shutil.copy(join(data_dir, c, "masks", c + "_r1_mask.nii.gz"), join(labelstr, c + '.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'cbf', 1: 'cbv', 2: 'mtt', 3: 'tmax', 4: "ct"},
                          labels={
                              'background': 0,
                              'tumor': 1
                          },
                          num_training_cases=total,
                          file_ending='.nii.gz',
                          license='see https://www.synapse.org/Synapse:syn53708249/wiki/626323',
                          reference='see https://www.synapse.org/Synapse:syn53708249/wiki/626323',
                          dataset_release='1.0')
