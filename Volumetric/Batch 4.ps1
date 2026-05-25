

##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Batch 4"

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
$Percent_sampled = 0.75
#$Regularization_parameter = 0.01
$Iteration_number = 50
$Tolerance = 1e-6

$Regs_list = @(0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10)
$samplingpath = "C:\Users\osman\Documents\GitHub\CompressedSensingforMRI\output\nrLUT_2D_Gauss_trajectories\nrLUT_2D_Gauss_R1.51_pct0.5.bmp"



$run_recon = 1

if ($run_recon -eq 1) {
    foreach ($Reg in $Regs_list) {
        for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
            $currentpath = $deepestDirs[$i]
            Write-Host "Current path: " + $currentpath
            $currentdirname = $deepestDirnames[$i]
            Write-Host "Current dirname: " + $currentdirname
            $currentoutput = "$resultspath\$currentdirname\Reg $Reg Results"
            Write-Host "Current output path: " + $currentoutput

            New-Item -Path $currentoutput -ItemType Directory

            & python .\Volumetric\Volumetric_kspace_maskfrombitmap.py $currentpath $target_size $Wavelet_name $Wavelet_level $Percent_sampled $Reg $Iteration_number $Tolerance $currentoutput $samplingpath
        }
    }
}

$run_analysis = 1


###data folder has all the bitmap files
##batch 4 results/$dataset/Reg $Reg Results/ contains the bitmap files for each regularization parameter

if ($run_analysis -eq 1) {
    foreach ($Reg in $Regs_list) {
        for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
            $currentdirname = $deepestDirnames[$i]
            $bmpfilepath = "$resultspath\$currentdirname\Reg $Reg Results"
            $outputpath = "$resultspath\$currentdirname\Reg $Reg Results\Analysis"
            New-Item -Path $outputpath -ItemType Directory
            & python .\Volumetric\SNR_measurments.py $bmpfilepath $outputpath
        }
    }
}



##make a direcotry for 