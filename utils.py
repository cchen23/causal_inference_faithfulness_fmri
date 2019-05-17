import numpy as np
import pandas as pd

NUM_PERMUTATIONS = 10000
np.random.seed(seed=1234)
NUM_PROCESSES = 16
EPSILON = 1e-5

TIMEPOINTS_PER_SECOND = 1 / 0.72
NUM_GREYORDINATES = 91282
NUM_PARCELLATION_DIMENSIONS = 50

def one_permutation_correlation(stimulus, brain_activity):
    """
    Finds the correlation for one permutation of stimulus and brain activity.
    args:
        stimulus: [1 x num_samples] array of stimulus categories.
        brain_activity: [1 x num_samples] array of brain activity values.
    """
    stimulus_permuted = np.expand_dims(np.random.permutation(np.squeeze(stimulus)), axis=0)
    correlation = np.corrcoef(stimulus_permuted, brain_activity)[0, 1]
    return correlation

def permutation_test(stimulus, brain_activity):
    """
    Runs permutation test between vectors containing stimulus and brain_activity time series.
    args:
        stimulus: [1 x num_samples] array of stimulus categories.
        brain_activity: [1 x num_samples] array of brain activity values.
    """
    stimulus = np.expand_dims(np.squeeze(stimulus), axis=0)
    brain_activity = np.expand_dims(np.squeeze(brain_activity), axis=0)
    assert(stimulus.shape[0] == brain_activity.shape[0] == 1)
    assert(stimulus.shape[1] == brain_activity.shape[1])
    if np.sum(np.abs(brain_activity)) == 0:
        return np.nan, np.nan
    correlation_actual = np.corrcoef(stimulus, brain_activity)[0, 1]
    num_higher_correlations = 0
    for i in range(NUM_PERMUTATIONS):
        correlation_permuted = one_permutation_correlation(stimulus, brain_activity)
        # Count number of more extreme (greater absolute value) correlations.
        if np.abs(correlation_permuted) > np.abs(correlation_actual):
            num_higher_correlations += 1
    pvalue = (num_higher_correlations * 1.0) / NUM_PERMUTATIONS
    return correlation_actual, pvalue

def get_brain_activity_byregion(averages_filename, block0_name="lh", block1_name="rh"):
    activity_df = pd.read_pickle(averages_filename)
    activity_df = activity_df.sort_values(["subject", "block"])
    block00_name = block0_name + "0"
    block01_name = block0_name + "1"
    block10_name = block1_name + "0"
    block11_name = block1_name + "1"
    activity_df = activity_df.loc[activity_df.block.isin([block00_name, block01_name, block10_name, block11_name])]
    # Only choose subjects with all four blocks. First num is block type, second num is block order.
    block00_df = activity_df[activity_df.block == block00_name]
    block01_df = activity_df[activity_df.block == block01_name]
    block10_df = activity_df[activity_df.block == block10_name]
    block11_df = activity_df[activity_df.block == block11_name]
    block00_subjects, block00_counts = np.unique(block00_df.subject, return_counts=True)
    block01_subjects, block01_counts = np.unique(block01_df.subject, return_counts=True)
    block10_subjects, block10_counts = np.unique(block10_df.subject, return_counts=True)
    block11_subjects, block11_counts = np.unique(block11_df.subject, return_counts=True)
    eligible_subjects = set(block00_subjects).intersection(set(block01_subjects)).intersection(set(block10_subjects)).intersection(set(block11_subjects))
    activity_df = activity_df.loc[activity_df.subject.isin(eligible_subjects)]
    # Rename blocks for permutation test.
    activity_df = activity_df.replace(block0_name + "0", -1)
    activity_df = activity_df.replace(block0_name + "1", -1)
    activity_df = activity_df.replace(block1_name + "0", 1)
    activity_df = activity_df.replace(block1_name + "1", 1)
    return activity_df
