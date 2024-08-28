import nibabel as nib
import numpy as np
import os
from multiprocessing import Process
import pickle

def pickler(foldername, inputfolder, segfolder, outfolder):

    foldername = foldername.rstrip('/')

    i=0
    for filename in os.listdir(foldername):
        if filename.split('.')[-1] != 'gz':
            continue

        print(i, filename)
        
        pathname = os.path.join(foldername, filename)
        pred_nii = nib.nifti1.load(pathname)
        pred = pred_nii.get_fdata().astype(np.uint16)

        segpath = os.path.join(segfolder, filename)
        pred_nii = nib.nifti1.load(segpath)
        seg = pred_nii.get_fdata().astype(np.uint16)

        filename = filename.split('.')[0]
        
        T1path = os.path.join(inputfolder, filename + "_0000.nii.gz")
        pred_nii = nib.nifti1.load(T1path)
        t1 = pred_nii.get_fdata().astype(np.uint16)
        
        T1cepath = os.path.join(inputfolder, filename + "_0001.nii.gz")
        pred_nii = nib.nifti1.load(T1cepath)
        t1ce = pred_nii.get_fdata().astype(np.uint16)
        
        T2path = os.path.join(inputfolder, filename + "_0002.nii.gz")
        pred_nii = nib.nifti1.load(T2path)
        t2 = pred_nii.get_fdata().astype(np.uint16)
        
        Flairpath = os.path.join(inputfolder, filename + "_0003.nii.gz")
        pred_nii = nib.nifti1.load(Flairpath)
        flair = pred_nii.get_fdata().astype(np.uint16)
        
        
        dataMerged = (t1, t1ce, t2, flair, seg, pred)
        
        filename = f'{filename}.pkl'
        outPath = os.path.join(outfolder, filename)
        with open(outPath, 'wb') as outFile:
            pickle.dump(dataMerged, outFile)

        i += 1

if __name__=='__main__':
    procs = []

    inputfolder = '/mmfs1/gscratch/kurtlab/brats2024/data/nnUNet_raw/Dataset138_BraTS2024-GoAT/imagesTr'
    segfolder = '/mmfs1/gscratch/scrubbed/hitender/segmasks'

    for x in range(5):
        foldername= f"/mmfs1/gscratch/scrubbed/hitender/fold-{x}_pp"
        outfolder= f"/mmfs1/gscratch/scrubbed/hitender/fold-{x}_pickle"
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        proc = Process(target=pickler, args=(foldername, inputfolder, segfolder, outfolder))
        procs.append(proc)
        proc.start()
        
    #shutil.make_archive('nnUNet5foldVal', 'zip', outfolder)

    for proc in procs:
        proc.join()
        
