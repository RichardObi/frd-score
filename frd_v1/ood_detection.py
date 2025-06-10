# see if different feature representations can be used for OOD detection
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import scipy.stats as sps
import os
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score

from src.radiomics_utils import convert_radiomic_dfs_to_vectors, compute_and_save_imagefolder_radiomics_parallel
from src.dataset import SimpleImageDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_size = 256

import random
# set random seed
fix_seed = True
if fix_seed:
    seed = 1338
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def auc_deviation(labels, scores):
    return 2. * (roc_auc_score(labels, scores) - 0.5)

def main(
        in_img_folder,
        out_img_folder,
        detection_type="image",
        val_frac = 0.1,
        use_val_set = False
):
    radiomics_path1 = os.path.join(in_img_folder, 'radiomics.csv')
    radiomics_path2 = os.path.join(out_img_folder, 'radiomics.csv')

    # if needed, compute radiomics for the images
    if not os.path.exists(radiomics_path1):
        print("No radiomics found computed for image folder 1 at {}, computing now.".format(radiomics_path1))
        compute_and_save_imagefolder_radiomics_parallel(in_img_folder)
        print("Computed radiomics for image folder 1.")
    else:
        print("Radiomics already computed for image folder 1 at {}.".format(radiomics_path1))

    if not os.path.exists(radiomics_path2):
        print("No radiomics found computed for image folder 2 at {}, computing now.".format(radiomics_path2))
        compute_and_save_imagefolder_radiomics_parallel(out_img_folder)
        print("Computed radiomics for image folder 2.")
    else:
        print("Radiomics already computed for image folder 2 at {}.".format(radiomics_path2))

    # load radiomics
    radiomics_df1 = pd.read_csv(radiomics_path1)
    radiomics_df2 = pd.read_csv(radiomics_path2)

    # print shape of radiomics dataframes
    #print(radiomics_df1.shape, radiomics_df2_id.shape, radiomics_df2_ood.shape)

    in_activations, out_activations, in_filenames, out_filenames = convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                                         radiomics_df2,
                                                         match_sample_count=True, # needed for distance measures
                                                         return_image_fnames=True
                                                         ) 


    # randomly split in_activations into train and val, via val_frac
    val_idx = np.random.choice(in_activations.shape[0], int(val_frac*in_activations.shape[0]), replace=False)
    train_idx = np.array([i for i in range(in_activations.shape[0]) if i not in val_idx])

    in_activations_val = in_activations[val_idx]
    in_activations = in_activations[train_idx]

    if not use_val_set:
        in_activations_val = in_activations
        print("Using training set as validation set.")

    in_activations = torch.tensor(in_activations)
    out_activations = torch.tensor(out_activations)

    id_mean = in_activations.mean(dim=0)

    # scores are L2 between mean of in_activations and each out_activations
    scores = torch.stack([torch.norm(id_mean - out, dim=0) for out in out_activations])
    scores = scores.detach().numpy()

    ID_scores_val = torch.stack([torch.norm(id_mean - out, dim=0) for out in in_activations_val])
    ID_scores_val = ID_scores_val.detach().numpy()


    if detection_type == "image":
        # find OOD detection threshold via dist of in-distribution validation set to in dist training set
        # attmpt this using statistical testing and Gaussian assumption
        mu, sigma = np.mean(ID_scores_val), np.std(ID_scores_val)

        # DOF for t-test if needed
        dof = len(ID_scores_val) - 1

        ID_dist_assumption = 'gaussian' # alternatives (similar performance): "counting", "t"
        if ID_dist_assumption == 'gaussian':
            threshOOD = sigma*sps.norm.ppf(0.95) + mu

            # Calculate z-score
            z = (scores - mu) / sigma

            # Calculate the p-values
            p_value = 1 - sps.norm.cdf(z)  # One-tailed test

        elif ID_dist_assumption == 't':
            threshOOD = sigma*sps.t.ppf(0.95, dof) + mu

            # Calculate t-score
            t = (scores - mu) / (sigma / np.sqrt(len(ID_scores_val)))

            # Calculate the p-values
            p_value = 1 - sps.t.cdf(t, dof)

        elif ID_dist_assumption == "counting":
            # get threshold by counting
            threshOOD = np.percentile(ID_scores_val, 95)

            # Calculate p-value by counting
            p_value = np.array([np.sum(ID_scores_val > score) / len(ID_scores_val) for score in scores])

        # compute OOD detection accuracy when using the 95th percentile of the in-distribution scores as threshold
        threshold = threshOOD
        print(f"Predicted Threshold: {threshold}")
        pred = scores > threshold

        # save OOD predictions to file
        out_dir = 'outputs/ood_predictions'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_df = pd.DataFrame({'filename': out_filenames, 'ood_score': scores, 'ood_prediction': pred, 'p_value': p_value})
        out_df.to_csv(os.path.join(out_dir, 'ood_predictions.csv'))
        print("saved OOD detection results to {}.".format(os.path.join(out_dir, 'ood_predictions.csv')))

    elif detection_type == "dataset":
        # second part: develop normalized FRD-based scoring which doesn't need OOD validation data
        # AUC deviation between ID val and test data
        all_scores = np.concatenate([ID_scores_val, scores])
        all_labels = np.concatenate([np.zeros(len(ID_scores_val)), np.ones(len(scores))])
        auc_dev = auc_deviation(all_labels, all_scores)
        print("dataset-level OOD score (nFRD_group) = {}".format(auc_dev))

    else:
        raise ValueError("Detection type must be either 'image' or 'dataset'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--image_folder_reference', type=str, required=True)
    parser.add_argument('--dataset_level', action='store_true')
    args = parser.parse_args()

    detection_type = "dataset" if args.dataset_level else "image"

    main(
        args.image_folder_reference,
        args.image_folder,
        detection_type=detection_type
        )
