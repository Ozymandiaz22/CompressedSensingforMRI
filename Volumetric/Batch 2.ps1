

##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Test Batch 2"

# Get all directories with their depth
$allDirs = Get-ChildItem -Path $root -Directory -Recurse | Select-Object  FullName, @{
    Name = "Depth"
    Expression = { ($_.FullName -split '\\').Count }
}

# Find the maximum depth
$maxDepth = ($allDirs | Measure-Object -Property Depth -Maximum).Maximum

# Return only the deepest directories
$deepestDirs = @($allDirs | Where-Object { $_.Depth -eq $maxDepth } | Select-Object -ExpandProperty FullName)
#$allDirs | Where-Object { $_.Depth -eq $maxDepth } | Select-Object -ExpandProperty FullName

##in the root directory, create a new directory to store the resultant bitmap files, and one for the resultant comparison images
$deepestDirnames = Split-Path $deepestDirs -Leaf
$rootparent = Split-Path $root -Parent -Resolve
$rootleaf = Split-Path $root -Leaf
$resultsLeaf = $rootleaf + ' Results'
$resultspath = $rootparent + "\" + $resultsleaf
    
## make the results directory for the test batch
New-Item -Path $resultspath -ItemType Directory



### all directories have been generated

$Wavelet_level = 5
$target_size = 128
$Wavelet_name = "haar"
#$Percent_sampled = 0.75
$Regularization_parameter = 0.01
$Iteration_number = 10
$Tolerance = 1e-6
$sampling_regime = "2d"


$runtest = 1
$runanal = 1
#run for each set of dicoms
# if ($runtest -eq 1) {
# for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
#     $currentpath = $deepestDirs[$i]
#     $currentdirname = $deepestDirnames[$i]
#     $currentoutput = "$resultspath\$currentdirname Results"
#     & python .\Volumetric\Volumetric_kspace.py $currentpath $target_size $Wavelet_name $Wavelet_level $Percent_sampled $Regularization_parameter $Iteration_number $Tolerance $currentoutput $sampling_regime
# }
# }

###now run analysis script for reach results directory, getting snr measurements and difference images
# if ($runanal -eq 1) {
# foreach ($item in $deepestDirnames) {
#     ##grab the results directory for each set of dicoms
#     $currentinput = "$resultspath\$item Results"
#     #define a new directory to store the analysis results
#     New-Item -Path "$resultspath\$item Analysis" -ItemType Directory
#     New-Item -Path "$resultspath\$item Analysis\Difference_Images" -ItemType Directory
#     ##run the analysis script with the input directory as the results directory and the output directory as the analysis directory
#     & python .\Volumetric\SNR_measurments.py $currentinput "$resultspath\$item Analysis"
# }
# }

##set of tests for batch 2:

#indepedent variable : percent sampling 
#depedent variable: ssim measuremente between reconstructed and original images
#hypothesis: ssim decreases as percent sampling increases, but at some point it will plateau

#define the percent sampling as an array from 0.4 to 0.9 in increments of 0.05
$Percent_sampled_array = @(0.4)
#acceleration is the inverse of percent sampled

#in the resuls directory, make a sub-direcotry for each of the 7 datasets, and within each of those make a subdirectory for each percent sampling value#
#then after each set has run, run the analysis script to get the ssim measurements for each percent value and plot the results

foreach ($datasetName in $deepestDirnames) {
    $datasetResultsPath = Join-Path $resultspath "$datasetName Results"

    foreach ($percent in $Percent_sampled_array) {
        $percentFolder = ($percent.ToString().Replace('.', '_'))
        $percentPath = Join-Path $datasetResultsPath $percentFolder

        if (-not (Test-Path $percentPath)) {
            New-Item -Path $percentPath -ItemType Directory | Out-Null
        }
    }
} 

##the folders are now made for each percent sampling value for each dataset, now we can run the reconstrcution script for each percent sampling value for each dataset, and save the results in the corresponding folder
foreach ($datasetName in $deepestDirnames) {
    $datasetResultsPath = Join-Path $resultspath "$datasetName Results"
    $currentpath = $deepestDirs | Where-Object { $_ -like "*\$datasetName" }


    foreach ($percent in $Percent_sampled_array) {
        $percentFolder = ($percent.ToString().Replace('.', '_'))
        $percentPath = Join-Path $datasetResultsPath $percentFolder

        # Run the reconstruction script with the current percent sampling value
        & python .\Volumetric\Volumetric_kspace_maskfrombitmap.py $currentpath $target_size $Wavelet_name $Wavelet_level $percent $Regularization_parameter $Iteration_number $Tolerance $percentPath $sampling_regime
    }
    #run the analysis script for each percent sampling value for each dataset, and save the results in the corresponding folder
    #make a new directory for the analysis results for each dataset
    New-Item -Path "$resultspath\$datasetName Analysis" -ItemType Directory | Out-Null
    & python .\Volumetric\SSIM_Analysis.py $datasetResultsPath "$resultspath\$datasetName Analysis"
}