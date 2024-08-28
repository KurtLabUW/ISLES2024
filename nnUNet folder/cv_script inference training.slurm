#!/bin/bash
#SBATCH --job-name=Inference_nnUNet-GoAT
#SBATCH --mail-user=hitender@uw.edu
#SBATCH --mail-type=ALL
#SBATCH -A kurtlab
#SBATCH -p ckpt
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --chdir=/gscratch/kurtlab/brats2024/repos/hitender
#SBATCH --export=all
#SBATCH --output=/gscratch/scrubbed/hitender/augval/outputmain.txt
#SBATCH --error=/gscratch/scrubbed/hitender/augval/errormain.txt

# Number of cross-validation folds
NUM_FOLDS=4

# Cross-validation training
for FOLD_NO in $(seq 0 $NUM_FOLDS); do
    # Submit a separate Slurm job for each fold
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=fold-3d-${FOLD_NO}-ResEncL
#SBATCH --mail-user=hitender@uw.edu
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=a40:1
#SBATCH --account=kurtlab
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --chdir=/gscratch/kurtlab/brats2024/repos/hitender
#SBATCH --export=all
#SBATCH --output=/gscratch/scrubbed/hitender/augval/output-${FOLD_NO}.txt
#SBATCH --error=/gscratch/scrubbed/hitender/augval/error-${FOLD_NO}.txt

source ~/.bashrc
source activate undergraddl

export nnUNet_raw="/mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw"
export nnUNet_results="/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0802/results"


python -m nnUNet.nnunetv2.inference.predict_from_raw_data -d 138 -i /gscratch/kurtlab/brats2024/data/nnUNet_raw/Dataset139_BraTS2024-GoATVal/imagesTr \
    -o /gscratch/scrubbed/hitender/augvalprobs/fold-${FOLD_NO} -f ${FOLD_NO} -tr nnUNetTrainer -c 3d_fullres -p nnUNetResEncUNetLPlans --save_probabilities

EOT
done
