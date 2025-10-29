# Python scripts for data elaboration



## preproc
The scripts in this package allows preprocessing of dataset file to get an uniform representation for the elaboration of the next steps

- dataset5_ecg2csv.py extract the ecg series sampled at 100hz from the txt files in the data/dataset5 directory and save them as single column csv files


## detection
### compute_parameters.py 
takes in input one column csv files that contains ecg values at 100hz and compute dhr, LF/HF, and edr values.
Output files are named {file_id}_delta_hr_vals.csv, {file_id}_lfhf_vals.csv, {file_id}_edr_vals.csv
It needs to move it to data/dataset5
The data directory and the csv files can be changed into the source. 

### visualize_parameters.py 
Generate plots which show the time variation of computing parameters and save it in files
### disorder_detection.py
Compute precision and recall per subject, and aggregated, using the subjects and ground_truth defined data/dataset5/gt_summary.csv
It generates two csv files: results_per_subject.csv and results_aggregate.csv