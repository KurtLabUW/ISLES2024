from ..evaluation.eval import *
import pandas as pd
import os
import argparse

TISSUE_TYPES = ['WT', 'TC', 'ET']

def eval_metrics(gt_data_folder, pred_folder, challenge_name):

    i=0
    pred_files = os.listdir(pred_folder)
    for pred_filename in pred_files:
        bits = pred_filename.split('.', maxsplit=1)
        if bits[1] != 'nii.gz':
            continue
        pred_pathname = os.path.join(pred_folder, pred_filename)
        gt_pathname = os.path.join(gt_data_folder, pred_filename)

        if not os.path.exists(gt_pathname) or not os.path.exists(pred_pathname):
            # print(gt_pathname, '\n', pred_pathname)
            print('Path not exist error')
            continue
        # else:
            # print('Path exists!')
            # print(gt_pathname, '\n', pred_pathname)

        # print('Getting lw results')
        results_df = get_LesionWiseResults(pred_file=pred_pathname, gt_file=gt_pathname, challenge_name=challenge_name)
        
        print('Got')

        if i==0:
            master_df = results_df
        else:
            master_df = pd.concat((master_df, results_df))
        i+=1


    lesion_wise_dice = {}
    lesion_wise_hd95 = {}

    for tt in TISSUE_TYPES:
        df_tt = master_df[master_df.Labels==tt]
        lw_dice = df_tt.LesionWise_Score_Dice.mean()
        lw_hd95 = df_tt.LesionWise_Score_HD95.mean()

        lesion_wise_dice[tt] = lw_dice
        lesion_wise_hd95[tt] = lw_hd95

    return lesion_wise_dice, lesion_wise_hd95


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_data_folder', type=str, required=True)
    parser.add_argument('--pred_folder', type=str, required=True)
    parser.add_argument('--challenge_name', type=str, required=True)

    args = parser.parse_args()

    gt_data_folder = args.gt_data_folder
    pred_folder = args.pred_folder
    challenge_name = args.challenge_name

    lesion_wise_dice, lesion_wise_hd95 = eval_metrics(gt_data_folder, pred_folder, challenge_name)

    for tt in TISSUE_TYPES:
        print(f'Tissue type is {tt}')
        print(f'Dice score = {lesion_wise_dice[tt]}')
        print(f'HD95 score = {lesion_wise_hd95[tt]}')

if __name__ == '__main__':
    main()