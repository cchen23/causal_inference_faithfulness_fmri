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
from sklearn import linear_model
from utils import *

def partial_correlation_test(S, X, Y):
    S = np.expand_dims(np.squeeze(S), axis=1)
    X = np.expand_dims(np.squeeze(X), axis=1)
    Y = np.expand_dims(np.squeeze(Y), axis=1)
    assert(S.shape[1] == X.shape[1] == Y.shape[1] == 1)
    assert(S.shape[0] == X.shape[0] == Y.shape[0])

    lm_S = linear_model.LinearRegression()
    model_S = lm_S.fit(X, S)
    predictions_S = np.expand_dims(np.squeeze(lm_S.predict(X)), axis=1)
    residuals_S = np.array(S) - np.array(predictions_S)

    # regression for brain region
    lm_Y = linear_model.LinearRegression()
    model_Y = lm_Y.fit(X, Y)
    predictions_Y = np.expand_dims(np.squeeze(lm_Y.predict(X)), axis=1)
    residuals_Y = np.array(Y) - np.array(predictions_Y)

    correlation, pvalue = permutation_test(residuals_S, residuals_Y)
    return correlation, pvalue

def partial_correlation_test_oneregion(y_br, stimulus, X_region32, X_region33, results_directory):
    info = pd.DataFrame(columns=["brain_region", "conditioned_subset", "correlation", "pvalue"])
    column_name = y_br[0]
    if "region" not in column_name:
        return
    region = column_name.split("region")[1]
    if not region.isdigit():
        return
    region = int(region)
    print("Running partial correlation test for region %d." % region)
    correlation, pvalue = partial_correlation_test(stimulus, X_region32, y_br[1])
    info = info.append({"brain_region":region, "conditioned_subset":"region32", "correlation":correlation, "pvalue":pvalue}, ignore_index=True)
    correlation, pvalue = partial_correlation_test(stimulus, X_region33, y_br[1])
    info = info.append({"brain_region":region, "conditioned_subset":"region33", "correlation":correlation, "pvalue":pvalue}, ignore_index=True)
    info.to_pickle(os.path.join(results_directory, "%d.p" % region))

def run_partial_correlation_test(region_type):
    block0_name = "lh"
    block1_name = "rh"

    brain_activity_byparcel = get_brain_activity_byregion(filepaths.PARCELS_AVERAGES_FILENAME, block0_name, block1_name)
    X_region32 = brain_activity_byparcel[["region32"]]
    X_region33 = brain_activity_byparcel[["region33"]]
    stimulus = brain_activity_byparcel["block"]

    if region_type == "greyordinate":
        brain_activity_byregion = get_brain_activity_byregion(filepaths.GREYORDINATES_AVERAGES_FILENAME, block0_name, block1_name)
        # Make sure subjects and blocks are matched up.
        assert(list(brain_activity_byregion.subject) == list(brain_activity_byparcel.subject))
        assert(list(brain_activity_byregion.block) == list(brain_activity_byparcel.block))
        subject_save_dir = filepaths.GREYORDINATES_PARTIALCORRELATIONS_SEPARATE_DIR
        merged_savepath = filepaths.GREYORDINATES_PARTIALCORRELATIONS
    elif region_type == "parcel":
        brain_activity_byregion = brain_activity_byparcel
        subject_save_dir = filepaths.PARCELS_PARTIALCORRELATIONS_SEPARATE_DIR
        merged_savepath = filepaths.PARCELS_PARTIALCORRELATIONS
    else:
        raise Exception("Invalid region type. Region type must be 'greyordinate' or 'parcel'. Provided region was %s." % region_type)

    if not os.path.exists(subject_save_dir):
        os.mkdir(subject_save_dir)

    info_list = list(brain_activity_byregion.iteritems())
    for region_info in info_list:
        partial_correlation_test_oneregion(region_info, stimulus, X_region32, X_region33, subject_save_dir)

    # Merge results.
    merged_results = pd.DataFrame(columns=["brain_region", "conditioned_subset", "correlation", "pvalue"])
    for filename in os.listdir(subject_save_dir):
        part_results = pd.read_pickle(os.path.join(subject_save_dir, filename))
        merged_results = merged_results.append(part_results, ignore_index=True)
    merged_results.to_pickle(merged_savepath)

if __name__ == '__main__':
    region_type = sys.argv[1]
    run_partial_correlation_test(region_type)
