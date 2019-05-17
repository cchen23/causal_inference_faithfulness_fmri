"""
Creates dataframes containing blocked activation data.

Requires that tfMRI subject data folders are in tfMRI_DATA_DIR folder in filepaths.py. (download details in [TODO])

"""
import filepaths
import nibabel as nib
import numpy as np
import pandas as pd
import pickle
import os

from functools import partial
from multiprocessing import Pool
from nibabel import cifti2 as cif
from utils import *

def get_blocktimes():
    """
    Saves and returns start and end times of each block.
    """

    def append_block(filename, blocks0, blocks1):
        with open(filename) as f:
            a = np.loadtxt(f)
            blocks0 = np.append(blocks0, np.expand_dims(a[0], axis=0), axis=0)
            blocks1 = np.append(blocks1, np.expand_dims(a[1], axis=0), axis=0)
        return blocks0, blocks1

    def get_average_block_startend(blocks):
        start = np.mean(blocks, axis=0)[0] + CUE_LENGTH
        end = start + MOVEMENT_LENGTH
        return start, end

    NUM_INFOS = 3
    CUE_LENGTH = 3
    MOVEMENT_LENGTH = 12

    lh_blocks0 = np.empty([0, NUM_INFOS])
    lh_blocks1 = np.empty([0, NUM_INFOS])
    rh_blocks0 = np.empty([0, NUM_INFOS])
    rh_blocks1 = np.empty([0, NUM_INFOS])
    lf_blocks0 = np.empty([0, NUM_INFOS])
    lf_blocks1 = np.empty([0, NUM_INFOS])
    rf_blocks0 = np.empty([0, NUM_INFOS])
    rf_blocks1 = np.empty([0, NUM_INFOS])
    t_blocks0 = np.empty([0, NUM_INFOS])
    t_blocks1 = np.empty([0, NUM_INFOS])

    for foldername in os.listdir(filepaths.tfMRI_DATA_DIR):
        if foldername.isdigit():
            print("***********************%s**********************" % foldername)
            EVs_path = os.path.join(filepaths.tfMRI_DATA_DIR, foldername, 'MNINonLinear', 'Results', 'tfMRI_MOTOR_LR', 'EVs')
            if not os.path.isdir(EVs_path):
                print("Subject path not found.")
                continue
            lh_blocks0, lh_blocks1 = append_block(os.path.join(EVs_path, "lh.txt"), lh_blocks0, lh_blocks1)
            rh_blocks0, rh_blocks1 = append_block(os.path.join(EVs_path, "rh.txt"), rh_blocks0, rh_blocks1)
            lf_blocks0, lf_blocks1 = append_block(os.path.join(EVs_path, "lf.txt"), lf_blocks0, lf_blocks1)
            rf_blocks0, rf_blocks1 = append_block(os.path.join(EVs_path, "rf.txt"), rf_blocks0, rf_blocks1)
            t_blocks0, t_blocks1 = append_block(os.path.join(EVs_path, "t.txt"), t_blocks0, t_blocks1)

    blocktimes_dict = {"lh0":get_average_block_startend(lh_blocks0), "lh1":get_average_block_startend(lh_blocks1),
                       "rh0":get_average_block_startend(rh_blocks0), "rh1":get_average_block_startend(rh_blocks1),
                       "lf0":get_average_block_startend(lf_blocks0), "lf1":get_average_block_startend(lf_blocks1),
                       "rf0":get_average_block_startend(rf_blocks0), "rf1":get_average_block_startend(rf_blocks1),
                       "t0":get_average_block_startend(t_blocks0), "t1":get_average_block_startend(t_blocks1)}

    with open(os.path.join(filepaths.BLOCKTIMES_FILENAME), "wb") as f:
        pickle.dump(blocktimes_dict, f)

    return blocktimes_dict

def save_parcel_assignments():
    """
    Saves assignments of greyordinates to parcels.
    """
    a = cif.load(os.path.join(filepaths.GROUP_PARCELLATIONS_DIR % NUM_PARCELLATION_DIMENSIONS, "melodic_IC.dscalar.nii"))
    assignments = np.argmax(a.get_data(), axis=0)
    np.save(os.path.join(filepaths.PARCEL_ASSIGNMENTS_FILENAME % NUM_PARCELLATION_DIMENSIONS), assignments)

"""
Region Block Activity for Each Subject
"""

def save_averageactivation_subject(filename, blocktimes_in_frames, region_type, greyordinates_assignments):
    """
    Saves greyordinate average block data.
    args:
        filename: filename of subject parcellation.
    """
    if region_type == "greyordinate":
        subject_averages_filename = filepaths.GREYORDINATES_SUBJECT_AVERAGES_FILENAME
        num_regions = NUM_GREYORDINATES
    elif region_type == "parcel":
        subject_averages_filename = filepaths.PARCEL_SUBJECT_AVERAGES_FILENAME
        num_regions = NUM_PARCELLATION_DIMENSIONS
    else:
        raise Exception("Invalid region type. Region type must be 'greyordinate' or 'parcel'.")

    subject = filename.split(".")[0]
    if not subject.isdigit():
        return
    if os.path.isfile(os.path.join(subject_averages_filename % (subject))):
        return

    # Get subject data.
    subject_motordata_path = filepaths.SUBJECT_MOTOR_GREYORDINATES_FILENAME % int(subject)
    if not os.path.isfile(subject_motordata_path):
        return
    region_activations_df = pd.DataFrame(columns = (["subject", "block"] + ["region%d" % i for i in range(num_regions)]))
    img = nib.load(subject_motordata_path)
    subject_data = img.get_data() # [num_timepoints x num_greyordinates]

    subject_data = (subject_data - np.mean(subject_data, axis=0)) / (np.std(subject_data, axis=0) + EPSILON)

    # Get averaged region data.
    for block in blocktimes_in_frames:
        # Subset by timepoints.
        block_start, block_end = blocktimes_in_frames[block]
        regions_dict = {}
        #
        for region in range(num_regions):
            if region_type == "greyordinate":
                subject_data_region = np.expand_dims(subject_data[:, region], axis=1) # [num_timepoints x 1]
            else:
                subject_data_region = subject_data[:, np.where(greyordinates_assignments == region)].squeeze()
            subject_data_region_averaged = np.mean(subject_data_region, axis=1) # [num_timepoints]
            regions_dict["region%d" % region] = np.mean(subject_data_region_averaged[block_start:block_end])
        #
        regions_dict.update({"subject":subject, "block":block})
        region_activations_df = region_activations_df.append(regions_dict, ignore_index=True)
    region_activations_df.to_pickle(os.path.join(subject_averages_filename % (subject)))
    print("Saved average %s activation for subject %s. %s" % (region_type, subject, time.strftime("%H%M")))

def save_averageactivations_parallelized(region_type):
    subject_data_dir = filepaths.tfMRI_DATA_DIR
    if region_type == "parcel":
        parcellations_filename = filepaths.PARCEL_ASSIGNMENTS_FILENAME % NUM_PARCELLATION_DIMENSIONS
        greyordinates_assignments = np.load(parcellations_filename)
        subject_averages_dir = filepaths.PARCELS_SUBJECT_AVERAGES_DIR
        averages_filename = filepaths.PARCELS_AVERAGES_FILENAME
        num_regions = NUM_PARCELLATION_DIMENSIONS
    elif region_type == "greyordinate":
        greyordinates_assignments = None
        subject_averages_dir = filepaths.GREYORDINATES_SUBJECT_AVERAGES_DIR
        averages_filename = filepaths.GREYORDINATES_AVERAGES_FILENAME
        num_regions = NUM_GREYORDINATES
    else:
        raise Exception("Invalid region type. Region type must be 'greyordinate' or 'parcel'.")
    if not os.path.exists(subject_averages_dir):
        os.mkdir(subject_averages_dir)

    with open(filepaths.BLOCKTIMES_FILENAME, "rb") as f:
        blocktimes_in_seconds = pickle.load(f)
    blocktimes_in_frames = {block:[int(round(times[0] * TIMEPOINTS_PER_SECOND)), int(round(times[1] * TIMEPOINTS_PER_SECOND))] for block, times in blocktimes_in_seconds.items()}

    p = Pool(NUM_PROCESSES)
    p.map(partial(save_averageactivation_subject, blocktimes_in_frames=blocktimes_in_frames, region_type=region_type, greyordinates_assignments=greyordinates_assignments), os.listdir(subject_data_dir))
    p.close()
    p.join()

    region_activations_df = pd.DataFrame(columns =  (["subject", "block"] + ["region%d" % i for i in range(num_regions)]))
    print("Combining average %s activations for all subjects." % region_type)
    for filename in os.listdir(subject_averages_dir):
        subject_activations = pd.read_pickle(os.path.join(subject_averages_dir, filename))
        region_activations_df = region_activations_df.append(subject_activations, ignore_index=True)
    region_activations_df.to_pickle(averages_filename)

    # Remove subject files.
    filelist = os.listdir(subject_averages_dir)
    for filename in filelist:
        os.remove(os.path.join(subject_averages_dir, filename))

if __name__ == '__main__':
    save_averageactivations_parallelized("greyordinate")
