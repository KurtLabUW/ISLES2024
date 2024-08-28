#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --mail-user=hitender@uw.edu
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=a40:1
#SBATCH --account=kurtlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --chdir=/gscratch/kurtlab/brats2024/repos/hitender
#SBATCH --export=all
#SBATCH --output=/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/output_train.txt
#SBATCH --error=/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/error_train.txt

source ~/.bashrc
source activate undergraddl

export nnUNet_raw="/mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw"
#export nnUNet_preprocessed="/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/preprocessed"
export nnUNet_results="/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0717/"
#export inputPath = '/mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw/Dataset138_BraTS2024-GoAT/imagesTr'
#export outputPath = '/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/inference/data'

python -m nnUNet.nnunetv2.inference.predict_from_raw_data -d 138 -i /mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw/Dataset138_BraTS2024-GoAT/imagesTr -o /mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/inference/data1 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans

python -m nnUNet.nnunetv2.inference.predict_from_raw_data -d 144 -i /mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0801/selectedfiles -o /mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0801/inference2 -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans 

python -m nnUNet.nnunetv2.inference.predict_from_raw_data -d 138 -i /mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw/Dataset138_BraTS2024-GoAT/imagesTr -o /mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/inference/data1 -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans
