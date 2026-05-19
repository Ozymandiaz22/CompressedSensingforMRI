## to do:
#select the correct python path witoput depending on vscode
# a list data to be tested has to be given 
# this shoudl loop though all of the datasets with the correct parameters for each dataset

$Wavelet_level = 5
$target_size = 256
$Wavelet_name = "haar"
$Percent_sampled = 0.5
$Regularization_parameter = 0.01
$Iteration_number = 20
$Tolerance = 1e-6

& python .\Volumetric\Volumetric_wholedata.py "Volumetric/knee_mri_clinical_seq_batch2/1FB_1001820591____1FB,_3331562518/study_2f43b031/MR4_53c76c27" $target_size $Wavelet_name $Wavelet_level $Percent_sampled $Regularization_parameter $Iteration_number $Tolerance