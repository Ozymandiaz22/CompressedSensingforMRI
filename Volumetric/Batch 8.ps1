##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Batch 8"
$samples = "C:\Users\osman\Documents\FYP Datasets\Batch 8 Sample"
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
Write-Host "Root parent directory: " + $rootparent
$rootleaf = Split-Path $root -Leaf
$resultsLeaf = $rootleaf + ' Results 2'
$resultspath = $rootparent + "\" + $resultsleaf
    
## make the results directory for the test batch
New-Item -Path $resultspath -ItemType Directory

##find sample directory a

$Wavelet_level = 7
$target_size = 256
$Regularization_parameter = 5
$Iteration_number = 10
$Tolerance = 1e-3

$Wavelets = @("haar", "db4", "db6", "sym4", "sym6", "coif1", "coif3", "coif5")

##get the single sampling pattern file from the samples directory
$samplingfile = Get-ChildItem -Path $samples -File -Filter "*.bmp" | Select-Object -First 1

##make a directory for nifti outputs if it doesn't exist
$niftipath = $rootparent + "\" + $rootleaf + " Nifti Outputs 2"
New-Item -Path $niftipath -ItemType Directory

$run_recon = 1
if ($run_recon -eq 1) {
    $samplingpath = $samplingfile.FullName
    $samplingname = $samplingfile.BaseName
    Write-Host "Current sampling path: " + $samplingpath
    Write-Host "Current sampling name: " + $samplingname

    foreach ($wavelet in $Wavelets) {
        for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
            $currentpath = $deepestDirs[$i]
            Write-Host "Current path: " + $currentpath
            $currentdirname = $deepestDirnames[$i]
            Write-Host "Current dirname: " + $currentdirname

            $currentoutput = "$resultspath\$currentdirname _ $wavelet Results 2"
            Write-Host "Current output path: " + $currentoutput

            New-Item -Path $currentoutput -ItemType Directory

            & python .\Volumetric\Volumetric_kspace_no_resize.py $currentpath $target_size $wavelet $Wavelet_level $Regularization_parameter $Iteration_number $Tolerance $currentoutput $samplingpath $niftipath
        }
    }
}

if ($run_anal -eq 1) {
    foreach ($item in $deepestDirnames) {
        ##grab the results directory for each set of dicoms
        $currentinput = "$resultspath\$item Results 2"
        #define a new directory to store the analysis results
        $currentanaloutput = "$resultspath\$item Analysis Results 2"
        New-Item -Path $currentanaloutput -ItemType Directory

        & python .\Volumetric\SNR_measurments.py $currentinput $currentanaloutput
    }
}