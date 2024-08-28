import multiprocessing
import shutil
from multiprocessing import Pool
import argparse

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
       description="Use this to convert ISLES 2024 datasets")
    parser.add_argument("path", type=str, help='path to dataset folder')
    parser.add_argument("id", type=int, help='task id')
    parser.add_argument("name", type=str, help='task name')
    args = parser.parse_args()
    
    data_dir = args.path
    task_id = args.id
    task_name = args.name

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    total = 0
    for batch in subdirs(data_dir, join=False):
        case_ids = subdirs(join(data_dir, batch, "derivatives"), join=False)
        total += len(case_ids)
        for c in case_ids:
            shutil.copy(join(data_dir, batch, "derivatives", c, "ses-01", "perfusion-maps", c + "_ses-01_space-ncct_cbf.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
            shutil.copy(join(data_dir, batch, "derivatives", c, "ses-01", "perfusion-maps", c + "_ses-01_space-ncct_cbv.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
            shutil.copy(join(data_dir, batch, "derivatives", c, "ses-01", "perfusion-maps", c + "_ses-01_space-ncct_mtt.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
            shutil.copy(join(data_dir, batch, "derivatives", c, "ses-01", "perfusion-maps", c + "_ses-01_space-ncct_tmax.nii.gz"), join(imagestr, c + '_0003.nii.gz'))
            shutil.copy(join(data_dir, batch, "derivatives", c, "ses-01", c + "_ses-01_space-ncct_cta.nii.gz"), join(imagestr, c + '_0004.nii.gz'))
            
            shutil.copy(join(data_dir, batch, "derivatives", c, "ses-02", c + "_ses-02_lesion-msk.nii.gz"), join(labelstr, c + '.nii.gz'))

    generate_dataset_json(out_base,
                          channel_names={0: 'cbf', 1: 'cbv', 2: 'mtt', 3: 'tmax', 4: "cta", 5: "ct"},
                          labels={
                              'background': 0,
                              'tumor': 1
                          },
                          num_training_cases=total,
                          file_ending='.nii.gz',
                          license='see https://www.synapse.org/Synapse:syn53708249/wiki/626323',
                          reference='see https://www.synapse.org/Synapse:syn53708249/wiki/626323',
                          dataset_release='1.0')
