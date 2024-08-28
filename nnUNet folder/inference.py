import subprocess
import sys
import os

# A clean version of the model is located inside /mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0717
# This path should point to the folder within which 'Dataset138_BraTS2024-GoAT' model resides
os.environ['nnUNet_results'] = '/path/to/model/folder'

# Path to nnUNet dataset folder where input files will be converted to. Just needs to be an empty folder
os.environ['nnUNet_raw'] = '/path/to/nnunet/data/folder'

# The main nnUNet folder (/mmfs1/gscratch/kurtlab/brats2024/repos/hitender/nnUNet) should be placed in the current working directory
nnUNetRun = '/project/nnUNet/nnunetv2/inference/predict_from_raw_data.py'
nnUNetDataset = '/project/nnUNet/nnunetv2/dataset_conversion/Dataset138_BraTS24.py'

# Path to input BraTS input files. Should be in the BraTS format 
inputPath = '/path/to/input/data'

# Files will be named in the nnUNet file prefix (i.e. 'BraTS-GoAT-00000.nii.gz', 'BraTS-GoAT-00001.nii.gz', etc.)
outputPath = '/path/to/output/location'

convertedInput = os.join(nnUNet_raw, "Dataset138_BraTS2024-GoAT")

#os.environ['nnUNet_results'] = "/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0717"
#convertedInput = '/mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw/Dataset138_BraTS2024-GoAT/imagesTr'
#outputPath = '/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/inference/data'

subprocess.call([sys.executable, nnUNetDataset, inputPath])
subprocess.call([sys.executable, nnUNetRun, '-d', '138', '-i', convertedInput, '-o', outputPath, '-f', '0', '1', '2', '3', '4', '-tr', 'nnUNetTrainer', '-c', '3d_fullres', '-p', 'nnUNetResEncUNetLPlans'])
