# PowerShell script to run Volumetric_kspace_final for a set of sampling masks

# ============== Configuration ==============
$inputRoot = "C:\Users\osman\Documents\Final_FYP_Dataset\Final_1"  # Update with your input data path
$maskFolder = "C:\Users\osman\Documents\Final_FYP_Dataset\Final_1_Sample"  # Folder containing sampling masks (BMPs)
$niftiFolder = "C:\Users\osman\Documents\Final_FYP_Dataset\Final_1_nifti"  # Optional: folder for NIFTI outputs, leave null to create in results

# Reconstruction parameters
$target_size = 256
$Wavelet_name = "haar"
$Wavelet_level = 7
$Regularization_parameter = 5
$Iteration_number = 10
$Tolerance = 1e-3

# Flags
$run_recon = 1
$run_analysis = 0

# ============== Activate Virtual Environment ==============

$venvPath = "C:\Users\osman\Documents\GitHub\CompressedSensingforMRI\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
} else {
    Write-Host "WARNING: Virtual environment not found at $venvPath" -ForegroundColor Yellow
}

# ============== Setup ==============

# Get all directories with their depth
$allDirs = Get-ChildItem -Path $inputRoot -Directory -Recurse | Select-Object FullName, @{
    Name = "Depth"
    Expression = { ($_.FullName -split '\\').Count }
}

# Find the maximum depth
$maxDepth = ($allDirs | Measure-Object -Property Depth -Maximum).Maximum

# Get only the deepest directories
$deepestDirs = @($allDirs | Where-Object { $_.Depth -eq $maxDepth } | Select-Object -ExpandProperty FullName)
$deepestDirnames = Split-Path $deepestDirs -Leaf

# Create results directory
$rootparent = Split-Path $inputRoot -Parent -Resolve
$rootleaf = Split-Path $inputRoot -Leaf
$resultsLeaf = $rootleaf + ' Results Volumetric Final'
$resultspath = Join-Path $rootparent $resultsLeaf

if (-not (Test-Path $resultspath)) {
    New-Item -Path $resultspath -ItemType Directory | Out-Null
    Write-Host "Created results directory: $resultspath"
}

# Create NIFTI output directory if not specified
if ($null -eq $niftiFolder) {
    $niftiFolder = Join-Path $rootparent ($rootleaf + " Nifti Outputs")
}

if (-not (Test-Path $niftiFolder)) {
    New-Item -Path $niftiFolder -ItemType Directory | Out-Null
    Write-Host "Created NIFTI output directory: $niftiFolder"
}

Write-Host "Found $($deepestDirs.Count) datasets"
Write-Host "Looking for masks in: $maskFolder"

# ============== Run Reconstruction ==============

if ($run_recon -eq 1) {
    # Get all BMP files in the mask folder
    $maskFiles = Get-ChildItem -Path $maskFolder -File -Filter "*.bmp"

    if ($maskFiles.Count -eq 0) {
        Write-Host "ERROR: No BMP files found in $maskFolder" -ForegroundColor Red
        exit 1
    }

    Write-Host "Found $($maskFiles.Count) sampling mask(s)" -ForegroundColor Green

    # Loop through each mask
    foreach ($maskFile in $maskFiles) {
        $maskPath = $maskFile.FullName
        $maskName = $maskFile.BaseName
        Write-Host "`n--- Processing mask: $maskName ---" -ForegroundColor Cyan

        # Loop through each dataset
        for ($i = 0; $i -lt $deepestDirs.Count; $i++) {
            $inputPath = $deepestDirs[$i]
            $datasetName = $deepestDirnames[$i]
            $outputPath = Join-Path $resultspath "$datasetName _ $maskName Results"

            Write-Host "  Dataset: $datasetName"
            Write-Host "  Input: $inputPath"
            Write-Host "  Output: $outputPath"

            # Create output directory
            if (-not (Test-Path $outputPath)) {
                New-Item -Path $outputPath -ItemType Directory | Out-Null
            }

            # Run the reconstruction script
            & python .\Volumetric\Volumetric_kspace_final.py `
                $inputPath `
                $target_size `
                $Wavelet_name `
                $Wavelet_level `
                $Regularization_parameter `
                $Iteration_number `
                $Tolerance `
                $outputPath `
                $maskPath `
                $niftiFolder

            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Script failed for dataset $datasetName with mask $maskName" -ForegroundColor Red
            } else {
                Write-Host "  [OK] Completed successfully" -ForegroundColor Green
            }
        }
    }
}

# ============== Run Analysis ==============

if ($run_analysis -eq 1) {
    Write-Host "`n--- Running Analysis ---" -ForegroundColor Cyan

    foreach ($datasetName in $deepestDirnames) {
        $analysisInputPath = Join-Path $resultspath "$datasetName Results"
        $analysisOutputPath = Join-Path $resultspath "$datasetName Analysis"

        # Create analysis output directory
        if (-not (Test-Path $analysisOutputPath)) {
            New-Item -Path $analysisOutputPath -ItemType Directory | Out-Null
        }

        Write-Host "  Analyzing: $datasetName"

        & python .\Volumetric\SNR_measurments.py $analysisInputPath $analysisOutputPath

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  WARNING: Analysis script failed for $datasetName" -ForegroundColor Yellow
        } else {
            Write-Host "  [OK] Analysis completed" -ForegroundColor Green
        }
    }
}

Write-Host "`n[COMPLETE] Batch processing complete!" -ForegroundColor Green
Write-Host "Results saved to: $resultspath"
