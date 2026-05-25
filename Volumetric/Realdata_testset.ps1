

##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Batch 1"

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


foreach ($item in $deepestDirnames) {
    ##make the results directory for each set of dicoms
        New-Item -Path "$resultspath\$item Results" -ItemType Directory
}

### all directories have been generated

$Wavelet_level = 5
$target_size = 256
$Wavelet_name = "haar"
$Percent_sampled = 0.75
$Regularization_parameter = 0.01
$Iteration_number = 100
$Tolerance = 1e-6
$sampling_regime = "1d"


$runtest = 1
$runanal = 1
#run for each set of dicoms
if ($runtest -eq 1) {
for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
    $currentpath = $deepestDirs[$i]
    $currentdirname = $deepestDirnames[$i]
    $currentoutput = "$resultspath\$currentdirname Results"
    & python .\Volumetric\Volumetric_kspace.py $currentpath $target_size $Wavelet_name $Wavelet_level $Percent_sampled $Regularization_parameter $Iteration_number $Tolerance $currentoutput $sampling_regime
}
}

###now run analysis script for reach results directory, getting snr measurements and difference images
if ($runanal -eq 1) {
foreach ($item in $deepestDirnames) {
    ##grab the results directory for each set of dicoms
    $currentinput = "$resultspath\$item Results"
    #define a new directory to store the analysis results
    New-Item -Path "$resultspath\$item Analysis" -ItemType Directory
    New-Item -Path "$resultspath\$item Analysis\Difference_Images" -ItemType Directory
    ##run the analysis script with the input directory as the results directory and the output directory as the analysis directory
    & python .\Volumetric\SNR_measurments.py $currentinput "$resultspath\$item Analysis"
    & python .\Volumetric\Difference_images.py $currentinput "$resultspath\$item Analysis\Difference_Images"
}
}