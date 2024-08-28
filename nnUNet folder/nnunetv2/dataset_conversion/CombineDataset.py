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
       description="Use this to combine datasets in nnunet format and identical data")
    parser.add_argument("path1", type=str, help='path to first dataset folder')
    parser.add_argument("path2", type=str, help='path to second dataset folder')
    parser.add_argument("id", type=int, help='task id')
    parser.add_argument("name", type=str, help='task name')
    args = parser.parse_args()
    
    data_dir1 = args.path1
    data_dir2 = args.path2
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
    i = 0
    prefix = "sub-stroke"
    case_ids = [x[10:-12] for x in sorted(listdir(join(data_dir1, "imagesTr")))[::5]]
    total += len(case_ids)
    for c in case_ids:
        shutil.copy(join(data_dir1, "imagesTr", prefix + c + "_0000.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0000.nii.gz"))
        shutil.copy(join(data_dir1, "imagesTr", prefix + c + "_0001.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0001.nii.gz"))
        shutil.copy(join(data_dir1, "imagesTr", prefix + c + "_0002.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0002.nii.gz"))
        shutil.copy(join(data_dir1, "imagesTr", prefix + c + "_0003.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0003.nii.gz"))
        shutil.copy(join(data_dir1, "imagesTr", prefix + c + "_0004.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0004.nii.gz"))
        
        shutil.copy(join(data_dir1, "labelsTr", prefix + c + ".nii.gz"), join(labelstr, "stroke_" + f"{i:04}" + ".nii.gz"))
        i = i+1
        
    prefix = "train_"
    case_ids = [x[6:-12] for x in sorted(listdir(join(data_dir2, "imagesTr")))[::5]]
    total += len(case_ids)
    for c in case_ids:
        shutil.copy(join(data_dir2, "imagesTr", prefix + c + "_0000.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0000.nii.gz"))
        shutil.copy(join(data_dir2, "imagesTr", prefix + c + "_0001.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0001.nii.gz"))
        shutil.copy(join(data_dir2, "imagesTr", prefix + c + "_0002.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0002.nii.gz"))
        shutil.copy(join(data_dir2, "imagesTr", prefix + c + "_0003.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0003.nii.gz"))
        shutil.copy(join(data_dir2, "imagesTr", prefix + c + "_0004.nii.gz"), join(imagestr, "stroke_" + f"{i:04}" + "_0004.nii.gz"))
        
        shutil.copy(join(data_dir2, "labelsTr", prefix + c + ".nii.gz"), join(labelstr, "stroke_" + f"{i:04}" + ".nii.gz"))
        i = i+1

    generate_dataset_json(out_base,
                          channel_names={0: 'cbf', 1: 'cbv', 2: 'mtt', 3: 'tmax', 4: "cta"},
                          labels={
                              'background': 0,
                              'tumor': 1
                          },
                          num_training_cases=total,
                          file_ending='.nii.gz',
                          license='see https://www.synapse.org/Synapse:syn53708249/wiki/626323',
                          reference='see https://www.synapse.org/Synapse:syn53708249/wiki/626323',
                          dataset_release='1.0')
