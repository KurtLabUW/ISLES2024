import nibabel as nib
import matplotlib.pyplot as plt

file = '/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/inference/BraTS-GoAT-00000.nii.gz'
filetruth = '/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/inference/BraTS-GoAT-00000_truth.nii.gz'
savepath = '/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender-0622/inference/slices/'

img = nib.load(file)
imgtruth = nib.load(filetruth)

data = img.get_fdata()
truthdata = imgtruth.get_fdata()

plt.imshow(data[:, :, 100]) 

plt.savefig(savepath + 'test01.png') 

f, axarr = plt.subplots(2, 16)

for x in range(1, 16):
    slice = data[:, :, 60 + (x*5)]
    truthslice = truthdata[:, :, 60 + (x*5)]
    axarr[0, x].imshow(slice, cmap=None)
    axes[0, x].axis('off')
    axarr[0, x].axes.set_axis_off()
    axarr[1, x].imshow(truthslice, cmap=None)
    axes[1, x].axis('off')
    axarr[1, x].axes.set_axis_off()
    

plt.savefig(savepath + "test.png")