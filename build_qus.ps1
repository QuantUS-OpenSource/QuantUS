param([string]$PythonPath = "python")

Write-Host "==> QuantUS QUS build — using $PythonPath"

# fresh build venv
if (Test-Path ".build_venv") { Remove-Item -Recurse -Force .build_venv }
& $PythonPath -m virtualenv .build_venv
& .\.build_venv\Scripts\Activate.ps1

# dependencies
pip install --upgrade pip setuptools wheel
pip install numpy
pip install -r requirements.txt
pip install -e .\engines\QuantUS-Backend-QUS
pip install pyinstaller==6.10.0

# compile Qt .ui files
Write-Host "==> Compiling .ui files..."
& .\saveQt.ps1

# freeze
Write-Host "==> PyInstaller..."
python -m PyInstaller quantus_qus.spec --clean --noconfirm

Write-Host "Done."