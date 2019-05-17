import filepaths
import os
import numpy as np
import pandas as pd
import random
import sys
import time

from functools import partial
from itertools import combinations
from multiprocessing import Process
from utils import *

def correlation_test(brain_activity_byregion, data_num):
    """
    args:
        brain_activity_byregion: DataFrame created in data.py (cols: subject, block, regionN).
    returns:
        [1 x num_regions] array of pvalues.
    """
    info = pd.DataFrame(columns=['brain_region', 'S_correlation', 'S_correlation_pvalue', 'M32_correlation', 'M32_correlation_pvalue', 'M33_correlation', 'M33_correlation_pvalue'])
    prev_time = time.time()
    stimulus = brain_activity_byregion['block']
    M_region32 = brain_activity_byregion["region32"]
    M_region33 = brain_activity_byregion["region33"]

    possible_regions = list(brain_activity_byregion.columns)
    num_possible_regions = len(possible_regions)
    data_subset_length = int(num_possible_regions / 10)
    for region in possible_regions[data_subset_length * data_num : data_subset_length * (data_num + 1)]:
        if "region" not in region:
            continue
        print("Running correlation test for region %s." % region)
        brain_activity = brain_activity_byregion[region]
        S_correlation, S_correlation_pvalue = permutation_test(stimulus, brain_activity)
        M32_correlation, M32_correlation_pvalue = permutation_test(M_region32, brain_activity)
        M33_correlation, M33_correlation_pvalue = permutation_test(M_region33, brain_activity)
        info = info.append({'brain_region':region,
               'S_correlation':S_correlation, 'S_correlation_pvalue':S_correlation_pvalue,
               'M32_correlation':M32_correlation, 'M32_correlation_pvalue':M32_correlation_pvalue,
               'M33_correlation':M33_correlation, 'M33_correlation_pvalue':M33_correlation_pvalue}, ignore_index=True)
    return info

def run_correlation_test(region_type):
    block0_name = "lh"
    block1_name = "rh"

    if region_type == "greyordinate":
        brain_activity_byregion = get_brain_activity_byregion(filepaths.GREYORDINATES_AVERAGES_FILENAME, block0_name, block1_name)
        save_path = filepaths.GREYORDINATES_CORRELATIONS_FILENAME
    elif region_type == "parcel":
        brain_activity_byregion = get_brain_activity_byregion(filepaths.PARCELS_AVERAGES_FILENAME, block0_name, block1_name)
        save_path = filepaths.PARCELS_CORRELATIONS_FILENAME
    else:
        raise Exception("Invalid region type. Region type must be 'greyordinate' or 'parcel'. Provided region was %s." % region_type)
    print("Starting correlation test for %s regions." % region_type)
    correlation_test_df = correlation_test(brain_activity_byregion, data_num)
    correlation_test_df.to_pickle(save_path)

if __name__ == '__main__':
    region_type = sys.argv[1]
    data_num = int(sys.argv[2])
    run_correlation_test(region_type)
