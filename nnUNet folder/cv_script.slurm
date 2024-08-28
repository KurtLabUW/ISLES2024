#!/bin/bash
#SBATCH --job-name=Train_nnUNet-GoAT
#SBATCH --mail-user=hitender@uw.edu
#SBATCH --mail-type=ALL
#SBATCH -A kurtlab
#SBATCH -p ckpt
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --chdir=/gscratch/kurtlab/brats2024/repos/hitender
#SBATCH --export=all
#SBATCH --output=/mmfs1/gscratch/kurtlab/brats2024/experiments/ISLES/hitender/hitender-0826/output_train.txt
#SBATCH --error=/mmfs1/gscratch/kurtlab/brats2024/experiments/ISLES/hitender/hitender-0826/error_train.txt

# Number of cross-validation folds
NUM_FOLDS=9

# Cross-validation training
for FOLD_NO in $(seq 0 $NUM_FOLDS); do
    # Submit a separate Slurm job for each fold
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=train-fold-3d-${FOLD_NO}-ResEncL
#SBATCH --mail-user=hitender@uw.edu
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=a40:1
#SBATCH --account=kurtlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --chdir=/gscratch/kurtlab/brats2024/repos/hitender
#SBATCH --export=all
#SBATCH --output=/mmfs1/gscratch/kurtlab/brats2024/experiments/ISLES/hitender/hitender-0826/results/Dataset150_ISLES2024_processedv3/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_${FOLD_NO}/output_train.txt
#SBATCH --error=/mmfs1/gscratch/kurtlab/brats2024/experiments/ISLES/hitender/hitender-0826/results/Dataset150_ISLES2024_processedv3/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_${FOLD_NO}/error_train.txt


source ~/.bashrc
source activate undergraddl

export nnUNet_n_proc_DA=24
export nnUNet_raw="/mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw"
export nnUNet_preprocessed="/mmfs1/gscratch/kurtlab/brats2024/experiments/ISLES/hitender/hitender-0826/preprocessed"
export nnUNet_results="/mmfs1/gscratch/kurtlab/brats2024/experiments/ISLES/hitender/hitender-0826/results"

python -m nnUNet.nnunetv2.run.run_training 150 3d_fullres ${FOLD_NO} --c -p nnUNetResEncUNetLPlans

EOT
done