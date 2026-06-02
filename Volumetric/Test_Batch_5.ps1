##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Test Batch 4"

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


$Wavelet_level = 7
$target_size = 256
$Wavelet_name = "haar"
$Regularization_parameter = 0.01
$Iteration_number = 20
$Tolerance = 1e-6

$samplingpath = "C:\Users\osman\Documents\GitHub\CompressedSensingforMRI\output\exLUT_radial_trajectories\exLUT_radial_trajectory_y150_z150_pct69.6_a16.95_r1.bmp"

$run_recon = 1
$run_anal = 1

if ($run_recon -eq 1) {
    for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
        $currentpath = $deepestDirs[$i]
        Write-Host "Current path: " + $currentpath
        $currentdirname = $deepestDirnames[$i]
        Write-Host "Current dirname: " + $currentdirname
        $currentoutput = "$resultspath\$currentdirname Results"
        Write-Host "Current output path: " + $currentoutput

        New-Item -Path $currentoutput -ItemType Directory

        & python .\Volumetric\Volumetric_kspace_maskfrombitmap.py $currentpath $target_size $Wavelet_name $Wavelet_level $Regularization_parameter $Iteration_number $Tolerance $currentoutput $samplingpath
    }
}

if ($run_anal -eq 1) {
    foreach ($item in $deepestDirnames) {
        ##grab the results directory for each set of dicoms
        $currentinput = "$resultspath\$item Results"
        #define a new directory to store the analysis results
        $currentanaloutput = "$resultspath\$item Analysis Results"
        New-Item -Path $currentanaloutput -ItemType Directory

        & python .\Volumetric\SNR_measurments.py $currentinput $currentanaloutput
    }
}