# PowerShell script to run Python scripts with TensorFlow environment
# Usage: .\run_with_tensorflow.ps1 script_name.py

param(
    [Parameter(Mandatory=$true)]
    [string]$ScriptName
)

if (-not (Test-Path $ScriptName)) {
    Write-Error "Script file '$ScriptName' not found!"
    exit 1
}

Write-Host "Running $ScriptName with TensorFlow environment..." -ForegroundColor Green
& C:\tf\Scripts\python.exe $ScriptName