import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import cc3d
import os
import shutil
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

AFFINE = np.array([[ -1.,   0.,   0.,  -0.],
                   [  0.,  -1.,   0., 239.],
                   [  0.,   0.,   1.,   0.],
                   [  0.,   0.,   0. ,  1.]])

def get_tissue_wise_seg(pred_mat, tissue_type):

    if tissue_type == 'WT':
        pred_mat_tissue_wise = pred_mat > 0
    elif tissue_type == 'TC':
        pred_mat_tissue_wise = np.logical_or(pred_mat == 1, pred_mat == 3)
    elif tissue_type == 'ET':
        pred_mat_tissue_wise = pred_mat == 3

    return pred_mat_tissue_wise.astype(np.uint16)

TISSUE_TYPES = ['WT', 'TC', 'ET']

## New processing strategy
'''
1. Remove ET dust
2. If ET dust created holes in TC, replace with NCR
3. Remvoe TC dust
4. If TC dust created holes in WT, replace with ED
5. Remove WT dust
'''

def rm_dust_fh(foldername, outfolder, radiomicsfile, nnunet):

    foldername = foldername.rstrip('/')
    # outfolder = f'{foldername}_pp'
    
    df = pd.read_csv(radiomicsfile)
    
    newdf = df.drop(columns=[
        'Image',
        'Mask',
        'diagnostics_Versions_PyRadiomics',
        'diagnostics_Versions_Numpy',
        'diagnostics_Versions_SimpleITK',
        'diagnostics_Versions_PyWavelet',
        'diagnostics_Versions_Python',
        'diagnostics_Configuration_Settings',
        'diagnostics_Configuration_EnabledImageTypes',
        'diagnostics_Image-original_Hash',
        'diagnostics_Image-original_Dimensionality',
        'diagnostics_Image-original_Spacing',
        'diagnostics_Image-original_Size',
        'diagnostics_Image-original_Mean',
        'diagnostics_Image-original_Minimum',
        'diagnostics_Image-original_Maximum',
        'diagnostics_Mask-original_Hash',
        'diagnostics_Mask-original_Spacing',
        'diagnostics_Mask-original_Size',
        'diagnostics_Mask-original_VoxelNum',
        'diagnostics_Mask-original_VolumeNum',
    ])
    
    newdf['diagnostics_Mask-original_BoundingBox'] = newdf['diagnostics_Mask-original_BoundingBox'].str[1:-1]
    newdf['diagnostics_Mask-original_CenterOfMassIndex'] = newdf['diagnostics_Mask-original_CenterOfMassIndex'].str[1:-1]
    newdf['diagnostics_Mask-original_CenterOfMass'] = newdf['diagnostics_Mask-original_CenterOfMass'].str[1:-1]
    
    newdf[['BoundingBox1', 'BoundingBox2', 'BoundingBox3', 'BoundingBox4', 'BoundingBox5', 'BoundingBox6']] = newdf['diagnostics_Mask-original_BoundingBox'].str.split(', ', expand=True)
    newdf[['CenterOfMassIndex1', 'CenterOfMassIndex2', 'CenterOfMassIndex3']] = newdf['diagnostics_Mask-original_CenterOfMassIndex'].str.split(', ', expand=True)
    newdf[['CenterOfMass1', 'CenterOfMass2', 'CenterOfMass3']] = newdf['diagnostics_Mask-original_CenterOfMass'].str.split(', ', expand=True)
    
    newdf = newdf.drop(columns=[
        'diagnostics_Mask-original_BoundingBox',
        'diagnostics_Mask-original_CenterOfMassIndex',
        'diagnostics_Mask-original_CenterOfMass'
    ])
    
    
    mask = np.column_stack([newdf[col].astype('str').str.contains(r"\([^)]*\)", na=False) for col in newdf])
    newdf = newdf.where(~mask, other=0)
    
    newdf = newdf.astype(float)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    tempdf = pd.DataFrame(imp.fit_transform(newdf))
    
    #dimensionality reduction
    tempdf = pd.DataFrame(PCA(n_components=10).fit_transform(tempdf))
    newdf = tempdf
    
    
    
    model = KMeans(n_clusters=5).fit(newdf.to_numpy())
    clusters = model.predict(newdf)
    
    np.set_printoptions(threshold=np.inf)

    # try 50 400

    i=0
    
    for filename in os.listdir(foldername):
        if filename.split('.')[-1] != 'gz':
            continue

        print(i, filename)

        pathname = os.path.join(foldername, filename)

        pred_nii = nib.nifti1.load(pathname)
        pred_mat = pred_nii.get_fdata().astype(np.uint16)
        seg_new = np.zeros_like(pred_mat)
        if nnunet:
            seg_new[pred_mat == 3] = 3
            seg_new[pred_mat == 2] = 1
            seg_new[pred_mat == 1] = 2
            pred_mat = seg_new
        pred_mat_new = pred_mat.copy()

        cluster = clusters[df.loc[df['Image'].str.contains(filename.split('.')[0].split("-")[-1])].index[0]]

        if cluster == 0:
            thres = 50
        elif cluster == 1:
            thres = 410
        else:
            thres = 350
        
        pred_mat_et = get_tissue_wise_seg(pred_mat_new, 'ET')
        pred_mat_et_rm_dust = cc3d.dust(pred_mat_et, threshold=thres, connectivity=26)
        rm_et_mask = np.logical_and(pred_mat_et==1, pred_mat_et_rm_dust==0)
        pred_mat_new[rm_et_mask] = 0

        pred_mat_tc = get_tissue_wise_seg(pred_mat_new, 'TC')
        tc_holes = 1 - pred_mat_tc
        tc_holes_rm = cc3d.dust(tc_holes, threshold=thres, connectivity=26)
        tc_filled = 1 - tc_holes_rm
        fill_ncr_mask = np.logical_and(tc_filled==1, pred_mat_new==0) * rm_et_mask
        pred_mat_new[fill_ncr_mask] = 1 #Fill holes with NCR

        pred_mat_tc = get_tissue_wise_seg(pred_mat_new, 'TC')
        pred_mat_tc_rm_dust = cc3d.dust(pred_mat_tc, threshold=thres, connectivity=26)
        rm_tc_mask = np.logical_and(pred_mat_tc==1, pred_mat_tc_rm_dust==0)
        pred_mat_new[rm_tc_mask] = 0

        pred_mat_wt = get_tissue_wise_seg(pred_mat_new, 'WT')
        wt_holes = 1 - pred_mat_wt
        wt_holes_rm = cc3d.dust(wt_holes, threshold=thres, connectivity=26)
        wt_filled = 1- wt_holes_rm
        fill_ed_mask = np.logical_and(wt_filled==1, pred_mat_new==0) * rm_tc_mask
        pred_mat_new[fill_ed_mask] = 2 #Fill holes with ED

        pred_mat_wt = get_tissue_wise_seg(pred_mat_new, 'WT')
        pred_mat_wt_rm_dust = cc3d.dust(pred_mat_wt, threshold=thres, connectivity=26)
        rm_wt_mask = np.logical_and(pred_mat_wt==1, pred_mat_wt_rm_dust==0)
        pred_mat_new[rm_wt_mask] = 0
        
        # get ETratio, if under 3% convert ET to ED
        WT = np.sum(pred_mat_new != 0)
        ET = np.sum(pred_mat_new == 3)
        ED = np.sum(pred_mat_new)
        if WT != 0:
            ratio = ET/WT
        else:
            ratio = 0
        
        if ratio < 0.03:
            pred_mat_new[pred_mat_new == 3] = 2
        
    
        pred_nii_rm_dust = nib.nifti1.Nifti1Image(pred_mat_new, affine=AFFINE)
        nib.nifti1.save(pred_nii_rm_dust, os.path.join(outfolder, filename))

        i += 1

def ratios(inferencefolder, gtfolder):
    #Look at GT and inference ET/WT and ED/WT percentages and plot with box and whisper chart showing comparison on training
    inferencefolder = inferencefolder.rstrip('/')
    gtfolder = gtfolder.rstrip('/')
    
    infETratio = []
    gtETratio = []
    
    infEDratio = []
    gtEDratio = []    
    
    i= 0
    for filename in os.listdir(inferencefolder):
        if filename.split('.')[-1] != 'gz':
            continue
        
        print(i, filename)
        
        predpath = os.path.join(inferencefolder, filename)
        gtpath = os.path.join(gtfolder, filename)

        pred_nii = nib.nifti1.load(predpath)
        pred_mat = pred_nii.get_fdata().astype(np.uint16)
        
        WT = np.sum(pred_mat != 0)
        ET = np.sum(pred_mat == 3)
        ED = np.sum(pred_mat == 2)
    
        if(ET != 0):   
            infETratio.append(ET/WT)
        if(ED != 0):
            infEDratio.append(ED/WT)
        
        gt_nii = nib.nifti1.load(gtpath)
        gt_mat = gt_nii.get_fdata().astype(np.uint16)
        
        WT = np.sum(gt_mat != 0)
        ET = np.sum(gt_mat == 3)
        ED = np.sum(gt_mat == 2)
        
        if(ET != 0):
            gtETratio.append(ET/WT)
        if(ED != 0):
            gtEDratio.append(ED/WT)
        i=i+1
    
    plt.boxplot([infETratio, gtETratio],labels=["inference", "GT"], vert=False)
    plt.savefig('ET.png')
    plt.clf()


    plt.boxplot([infEDratio, gtEDratio],labels=["inference", "GT"], vert=False)
    plt.savefig('ED.png')
    plt.clf()
    return

def ratiosnogt(inferencefolder):
    inferencefolder = inferencefolder.rstrip('/')
    
    infETratio = []
    
    infEDratio = []
    
    i= 0
    for filename in os.listdir(inferencefolder):
        if filename.split('.')[-1] != 'gz':
            continue
        
        print(i, filename)
        
        predpath = os.path.join(inferencefolder, filename)

        pred_nii = nib.nifti1.load(predpath)
        pred_mat = pred_nii.get_fdata().astype(np.uint16)
        
        WT = np.sum(pred_mat != 0)
        ET = np.sum(pred_mat == 3)
        ED = np.sum(pred_mat == 2)
    
        if(ET != 0):   
            infETratio.append(ET/WT)
        if(ED != 0):
            infEDratio.append(ED/WT)
        
        i=i+1
    
    return

if __name__=='__main__':
    foldername= "/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0719/inference"
    outfolder= "/mmfs1/gscratch/kurtlab/brats2024/experiments/brats-goat/hitender/hitender-0730/radiomicsval_pp4_corrected"
    radiomicsfile= "/mmfs1/gscratch/kurtlab/brats2024/repos/tianyi/radiomics_code/radiomics_GoAT/pyradiomics_GoAT_results_t2f_val.csv"
    #gtfolder = '/mmfs1/gscratch/scrubbed/hitender/segmasks'
    nnunet = True
    if not os.path.exists(outfolder):
      os.makedirs(outfolder)
    rm_dust_fh(foldername, outfolder, radiomicsfile, nnunet)
    #ratios(foldername, gtfolder)
    #shutil.make_archive('nnUNet5foldVal', 'zip', outfolder, nnunet)