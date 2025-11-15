# PowerShell script to convert .ui files to .py files
# Navigate to the target directory
Set-Location -Path ".\src"

# Find all .ui files in subdirectories and convert them to .py files
Get-ChildItem -Recurse -Filter "*.ui" | ForEach-Object {
    $uiFile = $_.FullName
    $pyFile = $uiFile -replace '\.ui$', '_ui.py'
    Write-Host "Converting $($_.Name) to $($pyFile)"
    pyuic6 $uiFile -o $pyFile
}

Write-Host "UI file conversion completed!" 