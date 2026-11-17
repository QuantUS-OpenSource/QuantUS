param([string]$PythonPath = "python")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Exe $($Arguments -join ' ')" }
}

Write-Host "==> Pre-flight checks"
if (-not (Test-Path "engines\qus\pyproject.toml")) {
    throw "engines\qus is empty. Run: git submodule update --init --recursive"
}

Write-Host "==> Fresh environment"
foreach ($d in @(".build_venv", "dist", "build", "dist_wheels", "engines\qus\build")) {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d }
}
Get-ChildItem engines\qus -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

Invoke-Checked $PythonPath @("-m", "venv", ".build_venv")
$py = ".\.build_venv\Scripts\python.exe"
$env:PATH = "$PWD\.build_venv\Scripts;$env:PATH"   # lets saveQt.ps1 find pyuic6

Invoke-Checked $py @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-Checked $py @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked $py @("-m", "pip", "install", "pyinstaller==6.10.0")

Write-Host "==> Building backend wheel (never use an editable install for freezing)"
Invoke-Checked $py @("-m", "pip", "wheel", ".\engines\qus", "--no-deps", "--no-build-isolation", "-w", "dist_wheels")
$wheel = Get-ChildItem dist_wheels\quantus-*.whl | Select-Object -First 1
Invoke-Checked $py @("-m", "pip", "install", $wheel.FullName)

Invoke-Checked $py @("-c", "import quantus; assert 'site-packages' in quantus.__file__, quantus.__file__; print('quantus OK:', quantus.__file__)")

Write-Host "==> Icon and Qt UI files"
if (-not (Test-Path "Images\logo.ico")) { Invoke-Checked $py @("make_icon.py") }
& .\saveQt.ps1
if (-not $?) { throw "saveQt.ps1 failed" }

Write-Host "==> PyInstaller (takes 15-30 min)"
Invoke-Checked $py @("-m", "PyInstaller", "quantus_qus.spec", "--clean", "--noconfirm")

if (-not (Test-Path "dist\QuantUS\QuantUS.exe")) { throw "dist\QuantUS\QuantUS.exe was not produced" }
if (Test-Path "dist\QuantUS\_internal\engines") { Write-Warning "duplicate engines folder found in the bundle" }

Write-Host "Done: dist\QuantUS\QuantUS.exe"