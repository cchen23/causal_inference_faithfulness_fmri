# Causal Inference and Faithfulness in fMRI.

This project contains code, for testing violations of strong faithfulness. It is specifically applied to fMRI data from the Human Connectome Project (https://www.humanconnectome.org).

## Data Download.
Instructions for downloading data used for our fMRI analyses is in data_download_instructions.pdf.
Subject data should be extracted into the directory name `tfMRI_DATA_DIR` in `filepaths.py`.
Group parcellations should be extracted into the directory name `GROUP_PARCELLATIONS_DIR` in `filepaths.py`

## Data Processing.
Run the following command to process the HCP fMRI data.

* `create_data_blocks.get_blocktimes()`
  * Finds the start- and endpoints for each task block.
* `create_data_blocks.save_parcel_assignments()`
  * Saves the greyordinates assigned to each parcel. We perform this assignment using group ICA maps released by HCP. We assign each greyordinate, to the parcel for which the ICA map has the highest absolute value.
* `create_data_blocks.save_averageactivations_parallelized("greyordinate")`
  * Saves the average activation over each block, for each greyordinate, for each subject.
* `create_data_blocks.save_averageactivations_parallelized("parcel")`
  * Saves the average activation over each block, for each parcel, for each subject.

## Analyses.
* `correlation_test.run_correlation_test("greyordinate")`
  * Computes the correlation between each greyordinate and the stimulus, left motor region, and right motor region. Also runs a permutation test for each correlation.
* `run_partial_correlation_test.run_partial_correlation_test("greyordinate")`
  * Computes the partial correlation between each greyordinate and the stimulus, conditioned on the left motor region and the right motor region. Also runs a permutation test for each partial correlation.
