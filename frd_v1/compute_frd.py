"""
Compute and interpret fréchet radiomics distances between two datasets.
"""
import os
from argparse import ArgumentParser
from time import time

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from src.radiomics_utils import convert_radiomic_dfs_to_vectors, compute_and_save_imagefolder_radiomics, compute_and_save_imagefolder_radiomics_parallel, interpret_radiomic_differences
from src.utils import frechet_distance


def main(
        image_folder1,
        image_folder2,
        force_compute_fresh = False,
        interpret = False,
        parallelize = True
):
    radiomics_fname = 'radiomics.csv'

    radiomics_path1 = os.path.join(image_folder1, radiomics_fname)
    radiomics_path2 = os.path.join(image_folder2, radiomics_fname)

    # if needed, compute radiomics for the images
    if force_compute_fresh or not os.path.exists(radiomics_path1):
        print("No radiomics found computed for image folder 1 at {}, computing now.".format(radiomics_path1))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder1, radiomics_fname=radiomics_fname)
        else:
            compute_and_save_imagefolder_radiomics(image_folder1, radiomics_fname=radiomics_fname)
        print("Computed radiomics for image folder 1.")
    else:
        print("Radiomics already computed for image folder 1 at {}.".format(radiomics_path1))

    if force_compute_fresh or not os.path.exists(radiomics_path2):
        print("No radiomics found computed for image folder 2 at {}, computing now.".format(radiomics_path2))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder2, radiomics_fname=radiomics_fname)
        else:
            compute_and_save_imagefolder_radiomics(image_folder2, radiomics_fname=radiomics_fname)
        print("Computed radiomics for image folder 2.")
    else:
        print("Radiomics already computed for image folder 2 at {}.".format(radiomics_path2))

    # load radiomics
    radiomics_df1 = pd.read_csv(radiomics_path1)
    radiomics_df2 = pd.read_csv(radiomics_path2)

    feats1, feats2 = convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                                         radiomics_df2,
                                                         match_sample_count=True, # needed for distance measures
                                                         ) 
    # Frechet distance
    fd = frechet_distance(feats1, feats2)
    frd = np.log(fd)

    print("FRD = {}".format(frd))

    if interpret:
        run_tsne = True
        interpret_radiomic_differences(radiomics_path1, radiomics_path2, run_tsne=run_tsne)

    return frd

if __name__ == "__main__":
    tstart = time()
    parser = ArgumentParser()

    parser.add_argument('--image_folder1', type=str, required=True)
    parser.add_argument('--image_folder2', type=str, required=True)
    parser.add_argument('--force_compute_fresh', action='store_true', help='re-compute all radiomics fresh')
    parser.add_argument('--interpret', action='store_true', help='interpret the features underlying Fréchet Radiomics Distance')

    args = parser.parse_args()

    main(
        args.image_folder1,
        args.image_folder2,
        force_compute_fresh=args.force_compute_fresh,
        interpret=args.interpret
        )

    tend = time()
    print("compute time (sec): {}".format(tend - tstart))
