import os
import sys

MAPPED_DRIVE = "Y:\\"
if "linux" in sys.platform:
  MAPPED_DRIVE = "/Y"

DATA_DIR = os.path.join(MAPPED_DRIVE, "data")
tfMRI_DATA_DIR = os.path.join(DATA_DIR, "3T_tfMRI_MOTOR_preproc")
SUBJECT_MOTOR_GREYORDINATES_FILENAME = os.path.join(tfMRI_DATA_DIR, "%d", "MNINonLinear", "Results", "tfMRI_MOTOR_LR", "tfMRI_MOTOR_LR_Atlas_MSMAll.dtseries.nii")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

GROUP_PARCELLATIONS_DIR = os.path.join(DATA_DIR, "HCP_PTN1200", "groupICA_3T_HCP1200_MSMAll", "groupICA", "groupICA_3T_HCP1200_MSMAll_d%d.ica")
PARCEL_ASSIGNMENTS_FILENAME = os.path.join(GROUP_PARCELLATIONS_DIR, "parcel_assignments.npy")

BLOCKTIMES_FILENAME = os.path.join(tfMRI_DATA_DIR, "blocktimes_all.p")

GREYORDINATES_SUBJECT_AVERAGES_DIR = os.path.join(tfMRI_DATA_DIR, "greyordinates_averages_subject")
GREYORDINATES_SUBJECT_AVERAGES_FILENAME = os.path.join(GREYORDINATES_SUBJECT_AVERAGES_DIR, "%s.p")
GREYORDINATES_AVERAGES_FILENAME = os.path.join(DATA_DIR, "greyordinates_averages.p")
PARCELS_SUBJECT_AVERAGES_DIR = os.path.join(tfMRI_DATA_DIR, "parcels_averages_subject")
PARCEL_SUBJECT_AVERAGES_FILENAME = os.path.join(PARCELS_SUBJECT_AVERAGES_DIR, "%s.p")
PARCELS_AVERAGES_FILENAME = os.path.join(DATA_DIR, "parcels_averages.p")

GREYORDINATES_PARTIALCORRELATIONS_SEPARATE_DIR = os.path.join(RESULTS_DIR, "greyordinates_partialcorrelations_separate")
PARCELS_PARTIALCORRELATIONS_SEPARATE_DIR = os.path.join(RESULTS_DIR, "greyordinates_partialcorrelations_separate")
GREYORDINATES_PARTIALCORRELATIONS = os.path.join(RESULTS_DIR, "greyordinates_partialcorrelations.p")
PARCELS_PARTIALCORRELATIONS = os.path.join(RESULTS_DIR, "parcels_partialcorrelations.p")

GREYORDINATES_CORRELATIONS_FILENAME = os.path.join(RESULTS_DIR, "greyordinates_correlationtest_results.p")
PARCELS_CORRELATIONS_FILENAME = os.path.join(RESULTS_DIR, "parcels_correlationtest_results.p")
