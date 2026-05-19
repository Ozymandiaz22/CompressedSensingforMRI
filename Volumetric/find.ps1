


##write test root directory
$root = "C:\Users\osman\Documents\FYP Datasets\Test batch 1"

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

New-Item -Path $resultspath -ItemType Directory


foreach ($item in $deepestDirnames) {
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

#run for each set of dicoms
for ($i = 0; $i -lt $deepestDirnames.Count; $i++) {
    $currentpath = $deepestDirs[$i]
    $currentdirname = $deepestDirnames[$i]
    $currentoutput = "$resultspath\$currentdirname Results"
    & python .\Volumetric\Volumetric_wholedata.py $currentpath $target_size $Wavelet_name $Wavelet_level $Percent_sampled $Regularization_parameter $Iteration_number $Tolerance $currentoutput
}