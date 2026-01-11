# OpenFace Pipeline Verification Script
# Usage: .\pipeline_test.ps1 [VideoName]

param (
    [string]$VideoName = "test_bash1"
)

$ErrorActionPreference = "Continue"

# --- CONFIGURATION ---
$RootDir = Get-Location
$BinDir = "$RootDir\x64\Release"
$SampleDir = "$RootDir\samples\recover_au_test"
$RecoverScriptsDir = "$RootDir\exe\RecoverAU"

$VideoPath = "$SampleDir\$VideoName.mp4"
$HogPath = "$SampleDir\$VideoName.hog"
$LandmarkCsvPath = "$SampleDir\$VideoName.csv"
$RecoverOutCsv = "$SampleDir\${VideoName}_recovered.csv"

$DumpGT = "$RootDir\dump_openface.csv"
$DumpTarget = "$RootDir\dump_recover.csv"

# --- CHECK EXECUTABLES ---
if (-not (Test-Path "$BinDir\FeatureExtraction.exe")) {
    Write-Error "FeatureExtraction.exe not found in $BinDir. Please build the project."
    exit 1
}
if (-not (Test-Path "$BinDir\RecoverAU.exe")) {
    Write-Error "RecoverAU.exe not found in $BinDir. Please build the project."
    exit 1
}

# --- CLEANUP ---
Write-Host "Cleaning up old dump and output files..." -ForegroundColor Cyan
if (Test-Path $DumpGT) { Remove-Item $DumpGT }
if (Test-Path $DumpTarget) { Remove-Item $DumpTarget }
# Also remove old HOG and CSV to ensure fresh generation
# 古いHOGとCSVも削除して、新しいファイルが生成されるようにする
if (Test-Path $HogPath) { 
    Write-Host "  Removing old HOG file: $HogPath"
    Remove-Item $HogPath 
}
if (Test-Path $LandmarkCsvPath) { 
    Write-Host "  Removing old CSV file: $LandmarkCsvPath"
    Remove-Item $LandmarkCsvPath 
}

# --- STEP 1: FeatureExtraction (Ground Truth) ---
Write-Host "`n[1/3] Running FeatureExtraction (Ground Truth)..." -ForegroundColor Cyan
if (Test-Path $VideoPath) {
    # Run FeatureExtraction
    # Note: Using -2Dfp -3Dfp -aus -pose -hogalign to ensure all standard processing happens
    # The -hogalign flag generates the .hog file (not -hog!)
    # 正しいフラグは -hogalign (-hog ではない!)
    $FeatCmd = "$BinDir\FeatureExtraction.exe"
    $FeatArgs = "-f", "$VideoPath", "-out_dir", "$SampleDir", "-2Dfp", "-3Dfp", "-aus", "-pose", "-hogalign"
    
    Write-Host "Command: $FeatCmd $FeatArgs"
    & $FeatCmd $FeatArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "FeatureExtraction exited with code $LASTEXITCODE."
    }
}
else {
    Write-Warning "Video file '$VideoPath' not found."
    Write-Warning "Skipping FeatureExtraction execution. existing 'dump_openface.csv' will be used if available."
}

if (-not (Test-Path $DumpGT)) {
    Write-Error "Ground Truth dump '$DumpGT' was not generated. Cannot proceed with verification."
    exit 1
}

# --- STEP 2: RecoverAU (Target) ---
Write-Host "`n[2/3] Running RecoverAU (Target)..." -ForegroundColor Cyan
if ((Test-Path $HogPath) -and (Test-Path $LandmarkCsvPath)) {
    $RecovCmd = "$BinDir\RecoverAU.exe"
    # Arguments: <hog_file> <landmark_csv> <output_csv>
    $RecovArgs = "$HogPath", "$LandmarkCsvPath", "$RecoverOutCsv"
    
    Write-Host "Command: $RecovCmd $RecovArgs"
    & $RecovCmd $HogPath $LandmarkCsvPath $RecoverOutCsv
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "RecoverAU failed with code $LASTEXITCODE."
        exit 1
    }
}
else {
    Write-Error "Input files for RecoverAU not found:`n HOG: $HogPath`n CSV: $LandmarkCsvPath"
    exit 1
}

if (-not (Test-Path $DumpTarget)) {
    Write-Error "Target dump '$DumpTarget' was not generated."
    exit 1
}

# --- STEP 3: Verification ---
Write-Host "`n[3/3] Running Verification Script..." -ForegroundColor Cyan
$VerifyArgs = "$DumpGT", "$DumpTarget"
Write-Host "Command: python $RecoverScriptsDir\verify_pipeline.py ..."

try {
    python "$RecoverScriptsDir\verify_pipeline.py" $DumpGT $DumpTarget
}
catch {
    Write-Error "Failed to run python script. Ensure python is in your PATH."
}

Write-Host "`nDone." -ForegroundColor Cyan
