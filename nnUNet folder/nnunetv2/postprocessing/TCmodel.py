import os
import shutil

files = ["BraTS-GoAT-02620.nii.gz",
"BraTS-GoAT-02644.nii.gz",
"BraTS-GoAT-02722.nii.gz",
"BraTS-GoAT-02724.nii.gz",
"BraTS-GoAT-02885.nii.gz",
"BraTS-GoAT-02920.nii.gz",
"BraTS-GoAT-02928.nii.gz",
"BraTS-GoAT-02590.nii.gz",
"BraTS-GoAT-02506.nii.gz",
"BraTS-GoAT-02658.nii.gz",
"BraTS-GoAT-02760.nii.gz",
"BraTS-GoAT-02843.nii.gz",
"BraTS-GoAT-02858.nii.gz",
"BraTS-GoAT-02874.nii.gz",
"BraTS-GoAT-02884.nii.gz",
"BraTS-GoAT-02902.nii.gz",
"BraTS-GoAT-02838.nii.gz"]
types = [
    "_0000.nii.gz",
    "_0001.nii.gz",
    "_0002.nii.gz",
    "_0003.nii.gz",
]

infolder = '/mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw/Dataset139_BraTS2024-GoATVal/imagesTr'

outfolder = '/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0801/selectedfiles'

for filename in files:
    print(filename)
    for filetype in types:
        filename.split(".")[0] + filetype
        pathname = os.path.join(infolder, filename.split(".")[0] + filetype)
        outname = os.path.join(outfolder, filename.split(".")[0] + filetype)
        shutil.copy(pathname, outname)
        