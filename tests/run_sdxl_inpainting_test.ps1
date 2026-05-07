<#
.SYNOPSIS
    Smoke test for --train_inpainting on SDXL (full FT and LoRA modes).

.DESCRIPTION
    Modes:
      ft   - sdxl_train.py (full fine-tune)
             Accepts SDXL inpainting checkpoints (in_channels=9) AND standard
             SDXL base models (in_channels=4 - conv_in is auto-expanded).
             Output is a full UNet checkpoint; verifier asserts conv_in=9ch.
      lora - sdxl_train_network.py (LoRA)
             Requires an SDXL inpainting checkpoint (in_channels=9). LoRA does
             not extend conv_in, so a standard SDXL will fail at UNet forward.
             Output is a LoRA-only file; verifier checks for lora_unet_* keys.

    The active virtualenv (with sd-scripts dependencies) must be activated
    before running this script, so that 'accelerate' and 'python' resolve to
    the project's venv binaries.

.PARAMETER Mode
    'ft' (full FT via sdxl_train.py) or 'lora' (via sdxl_train_network.py).

.PARAMETER Model
    Path to .safetensors or .ckpt checkpoint.

.PARAMETER Data
    Optional training data directory (DreamBooth folder layout).
    Falls back to tests\downloaded_data, then synthetic test_data.

.PARAMETER Steps
    Optional max_train_steps override (default: 20 from TOML).
    Pass a positive integer to override; 0 (the default) keeps the TOML value.

.EXAMPLE
    .\tests\run_sdxl_inpainting_test.ps1 -Mode ft -Model D:\Models\sdxl.safetensors

.EXAMPLE
    .\tests\run_sdxl_inpainting_test.ps1 -Mode lora -Model D:\Models\sdxl-inpainting.safetensors -Steps 5
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('ft', 'lora')]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$Data = '',

    [int]$Steps = 0
)

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$RepoDir   = Split-Path -Parent $ScriptDir

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $Model)) {
    Write-Error "Model not found: $Model"
    exit 1
}

# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------

if ([string]::IsNullOrEmpty($Data)) {
    $DownloadedDir = Join-Path $ScriptDir 'downloaded_data'
    $SyntheticDir  = Join-Path $ScriptDir 'test_data'

    $useDownloaded = $false
    if (Test-Path -LiteralPath $DownloadedDir -PathType Container) {
        $entries = Get-ChildItem -LiteralPath $DownloadedDir -Force -ErrorAction SilentlyContinue
        if ($entries -and @($entries).Count -gt 0) { $useDownloaded = $true }
    }

    if ($useDownloaded) {
        $Data = $DownloadedDir
        Write-Host "==> Using downloaded data: $Data"
    }
    else {
        if (-not (Test-Path -LiteralPath $SyntheticDir -PathType Container)) {
            Write-Host "==> Generating synthetic test images..."
            & python (Join-Path $ScriptDir 'generate_inpainting_test_data.py')
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Synthetic data generation failed (exit $LASTEXITCODE)."
                exit $LASTEXITCODE
            }
        }
        $Data = $SyntheticDir
        Write-Host "==> Using synthetic test images: $Data"
        Write-Host "    (Pass -Data DIR or pre-run download_training_data.py for real images.)"
    }
}

# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------

if ($Mode -eq 'ft') {
    $TrainScript = Join-Path $RepoDir   'sdxl_train.py'
    $BaseToml    = Join-Path $ScriptDir 'sdxl_inpainting_test_ft.toml'
    $OutputDir   = Join-Path $ScriptDir 'test_output_sdxl_ft'
    $OutputName  = 'sdxl_inpainting_test_ft'
    $VerifyArgs  = @('--expect-in-channels', '9')
}
else {
    $TrainScript = Join-Path $RepoDir   'sdxl_train_network.py'
    $BaseToml    = Join-Path $ScriptDir 'sdxl_inpainting_test_lora.toml'
    $OutputDir   = Join-Path $ScriptDir 'test_output_sdxl_lora'
    $OutputName  = 'sdxl_inpainting_test_lora'
    $VerifyArgs  = @('--expect-lora')
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# ---------------------------------------------------------------------------
# Assemble accelerate launch arguments (TOML provides defaults; CLI overrides)
# ---------------------------------------------------------------------------

$LaunchArgs = @(
    'launch',
    '--dynamo_backend',              'no',
    '--dynamo_mode',                 'default',
    '--num_processes',               '1',
    '--num_machines',                '1',
    '--num_cpu_threads_per_process', '2',
    $TrainScript,
    '--config_file',                 $BaseToml,
    '--pretrained_model_name_or_path', $Model,
    '--train_data_dir',              $Data,
    '--output_dir',                  $OutputDir,
    '--output_name',                 $OutputName
)
if ($Steps -gt 0) {
    $LaunchArgs += @('--max_train_steps', "$Steps")
}

$Expected = Join-Path $OutputDir ("{0}.safetensors" -f $OutputName)

Write-Host ""
Write-Host "==> SDXL inpainting smoke test (mode=$Mode)"
Write-Host "    script : $TrainScript"
Write-Host "    config : $BaseToml"
Write-Host "    model  : $Model"
Write-Host "    data   : $Data"
Write-Host "    output : $Expected"
Write-Host ""

& accelerate @LaunchArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "accelerate launch failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------

if (-not (Test-Path -LiteralPath $Expected -PathType Leaf)) {
    Write-Error "FAIL: expected output not found: $Expected"
    exit 1
}

$Verifier = Join-Path $ScriptDir '_verify_inpainting_checkpoint.py'
& python $Verifier $Expected @VerifyArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAIL: verifier reported failure (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "==> PASS"
