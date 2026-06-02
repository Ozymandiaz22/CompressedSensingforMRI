
##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Batch 9"

# Get all directories with their depth
$allDirs = Get-ChildItem -Path $root -Directory -Recurse | Select-Object  FullName, @{
    Name = "Depth"
    Expression = { ($_.FullName -split '\\').Count }
}

# Find the maximum depth
$maxDepth = ($allDirs | Measure-Object -Property Depth -Maximum).Maximum

# Return only the deepest directories
$deepestDirs = @($allDirs | Where-Object { $_.Depth -eq $maxDepth } | Select-Object -ExpandProperty FullName)

##in the root directory, create a new directory to store the resultant bitmap files, and one for the resultant comparison images
$deepestDirnames = Split-Path $deepestDirs -Leaf
$rootparent = Split-Path $root -Parent -Resolve
$rootleaf = Split-Path $root -Leaf
$resultsLeaf = $rootleaf + ' Results - kspace_noresize'
$resultspath = $rootparent + "\" + $resultsleaf

## make the results directory for the test batch
New-Item -Path $resultspath -ItemType Directory -Force


### all directories have been generated

$Wavelet_level = 7
$target_size = 256
$Wavelet_name = "haar"
$Iteration_number = 30
$Tolerance = 1e-6

$Regs_list = @(10,50,100,500,1000,5000,10000)

# Folder containing mask bitmap files
$masksfolder = "C:\Users\osman\Documents\FYP Datasets\Batch 9 sample"
$nifoutputfolder = "C:\Users\osman\Documents\FYP Datasets\Batch 9 niftis"

# Get all mask files from the folder
$maskfiles = @(Get-ChildItem -Path $masksfolder -Filter "*.bmp" | Select-Object -ExpandProperty FullName)

Write-Host "Found $($maskfiles.Count) mask files in $masksfolder"

$run_recon = 1

if ($run_recon -eq 1) {
    foreach ($Reg in $Regs_list) {
        for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
            $currentpath = $deepestDirs[$i]
            Write-Host "Current path: " + $currentpath
            $currentdirname = $deepestDirnames[$i]
            Write-Host "Current dirname: " + $currentdirname

            # Iterate through each mask file
            foreach ($maskfile in $maskfiles) {
                $maskname = [System.IO.Path]::GetFileNameWithoutExtension($maskfile)
                $currentoutput = "$resultspath\$currentdirname\Mask_$maskname\Reg $Reg Results"
                Write-Host "Current output path: " + $currentoutput

                New-Item -Path $currentoutput -ItemType Directory -Force

                # Run the reconstruction with the current mask
                & python .\Volumetric\Volumetric_kspace_no_resize.py $currentpath $target_size $Wavelet_name $Wavelet_level $Reg $Iteration_number $Tolerance $currentoutput $maskfile $nifoutputfolder
            }
        }
    }
}

$run_analysis = 0

###data folder has all the bitmap files
##batch 4 results/$dataset/Mask_$maskname/Reg $Reg Results/ contains the bitmap files for each regularization parameter and mask

if ($run_analysis -eq 1) {
    foreach ($Reg in $Regs_list) {
        for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
            $currentdirname = $deepestDirnames[$i]

            # Iterate through each mask file
            foreach ($maskfile in $maskfiles) {
                $maskname = [System.IO.Path]::GetFileNameWithoutExtension($maskfile)
                $bmpfilepath = "$resultspath\$currentdirname\Mask_$maskname\Reg $Reg Results"
                $outputpath = "$resultspath\$currentdirname\Mask_$maskname\Reg $Reg Results\Analysis"
                New-Item -Path $outputpath -ItemType Directory -Force
                & python .\Volumetric\SNR_measurments.py $bmpfilepath $outputpath
            }
        }
    }
}
