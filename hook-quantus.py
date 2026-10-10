from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Force PyInstaller to recursively include every submodule in the quantus package
hiddenimports = collect_submodules('quantus')

# Include all non-Python files too
datas = collect_data_files('quantus', include_py_files=True)