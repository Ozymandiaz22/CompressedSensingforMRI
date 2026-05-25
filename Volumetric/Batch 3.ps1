

##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Test Batch 3"

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
$Iteration_number = 100
$Tolerance = 1e-6



#run for each set of dicoms
# if ($runtest -eq 1) {
# for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
#     $currentpath = $deepestDirs[$i]
#     $currentdirname = $deepestDirnames[$i]
#     $currentoutput = "$resultspath\$currentdirname Results"
#     & python .\Volumetric\Volumetric_kspace.py $currentpath $target_size $Wavelet_name $Wavelet_level $Percent_sampled $Regularization_parameter $Iteration_number $Tolerance $currentoutput $sampling_regime
# }
# }

##set of tests for batch 2:

#indepedent variable : sampling mask
#depedent variable: ssim measuremente between reconstructed and original images
#hypothesis: ssim decreases as percent sampling increases, but at some point it will plateau

##import the bitmaps for spiral from this directory
$spiralpath = "C:\Users\osman\Documents\GitHub\CompressedSensingforMRI\output\exLUT_spiral_trajectories"
#extract the paths of the bitmap files for the spiral trajectories, and the percent sampling values from the file names, there are 5
#file format is exLUT_spiral_trajectory_y128_z128_pct{percent}_a{angle}_r{reps}.bmp, we want to extract the percent value from the file name and store it in an array
$spiralfiles = Get-ChildItem -Path $spiralpath -Filter "*.bmp"
$Percent_sampled_array = @()
foreach ($file in $spiralfiles) {
    $filename = $file.Name
    $percent = (($filename -split '_pct')[1] -split '_a')[0]
    $Percent_sampled_array += [double]$percent
}


#make a dictionary to store the percent sampling values and the corresponding file paths for the spiral trajectories
$spiralDict = @{}
foreach ($file in $spiralfiles) {
    $filename = $file.Name
    $percent = (($filename -split '_pct')[1] -split '_a')[0]
    $spiralDict[$percent] = $file.FullName
}

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

        # Run the reconstruction script for each mask file with the current percent sampling value
        #print the current percent sampling value and the corresponding file path for the spiral trajectory
        $percentKey = $percent.ToString()
        Write-Host "Running reconstruction for dataset: $datasetName, percent sampling: $percent, spiral trajectory file: $($spiralDict[$percentKey])"
        & python .\Volumetric\Volumetric_kspace_maskfrombitmap.py $currentpath $target_size $Wavelet_name $Wavelet_level $percent $Regularization_parameter $Iteration_number $Tolerance $percentPath $spiralDict[$percentKey]
    }
    #run the analysis script for each percent sampling value for each dataset, and save the results in the corresponding folder
    #make a new directory for the analysis results for each dataset
    New-Item -Path "$resultspath\$datasetName Analysis" -ItemType Directory | Out-Null
    & python .\Volumetric\SSIM_Analysis.py $datasetResultsPath "$resultspath\$datasetName Analysis"
}