##write test root directory
$root = "C:\Users\osman\Documents\Final_FYP_Dataset\Final_1"
$samples = "C:\Users\osman\Documents\FYP Datasets\Batch 5 Sample"
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
$resultsLeaf = $rootleaf + ' Groundtruth 2 Results'
$resultspath = $rootparent + "\" + $resultsleaf
    
## make the results directory for the test batch
New-Item -Path $resultspath -ItemType Directory

##find sample directory a

# $Wavelet_level = 7
 $target_size = 256
# $Wavelet_name = "haar"
# $Regularization_parameter = 0.01
# $Iteration_number = 30
# $Tolerance = 1e-6

##loop through the different sampling patterns for the batch, foudn in the samples directory, and extract their names to be used in the reconstruction script
these files are bmp
$samplingfiles = Get-ChildItem -Path $samples -File -Filter "*.bmp"

$run_recon = 1
if ($run_recon -eq 1) {

        for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
            $currentpath = $deepestDirs[$i]
            Write-Host "Current path: " + $currentpath
            $currentdirname = $deepestDirnames[$i]
            Write-Host "Current dirname: " + $currentdirname

            $currentoutput = "$resultspath\$currentdirname Truth"
            Write-Host "Current output path: " + $currentoutput

            New-Item -Path $currentoutput -ItemType Directory

            & python .\Volumetric\Groundtruth.py $currentpath $target_size  $currentoutput
        }
}

