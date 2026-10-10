# quantus_qus.spec
# Run with:  python -m PyInstaller quantus_qus.spec --clean --noconfirm

import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

block_cipher = None

# Collect packages that have dynamic imports, entry points, or data files
_COLLECT_ALL = [
    "numpy",
    "scipy",
    "napari",
    "vispy",
    "magicgui",
    "qtpy",
    "pyvista",
    "vtkmodules",
    "pyqtdarktheme",
    "skimage",
    "imageio",
    "tifffile",
    "superqt",          # napari UI dependency
]

# Packages whose distribution metadata (entry points) must be present at runtime
_COPY_META = [
    "napari",
    "quantus",          # the backend — needed for entry-point plugin discovery
    "pyqtdarktheme",
    "imageio",
    "magicgui",
    "superqt",
]

all_datas: list = []
all_binaries: list = []
all_hiddenimports: list = []

for pkg in _COLLECT_ALL:
    try:
        d, b, h = collect_all(pkg)
        all_datas += d
        all_binaries += b
        all_hiddenimports += h
    except Exception as e:
        print(f"[spec] WARNING: collect_all({pkg!r}) failed: {e}")

for pkg in _COPY_META:
    try:
        all_datas += copy_metadata(pkg)
    except Exception as e:
        print(f"[spec] WARNING: copy_metadata({pkg!r}) failed: {e}")

# Our own source files and assets
all_datas += [
    ("Images",  "Images"),      # icons, logo
    ("src",     "src"),         # compiled Qt UI .py files + MVC source
    # The backend package is installed, so PyInstaller finds it automatically.
    # But we need its configs folder too:
    ("engines/qus/configs", "configs"),
]

# Hidden imports that static analysis reliably misses
_EXTRA_HIDDEN = [
    "numpy._core",
    "numpy._core._exceptions",
    "numpy._core._multiarray_umath",
    "numpy._core.multiarray",
    "numpy._core.numeric",
    "numpy._core.arrayprint",
    "numpy._core.fromnumeric",
    "numpy._core.function_base",
    "numpy._core.getlimits",
    "numpy._core.shape_base",
    "numpy._core.einsumfunc",
    # PyQt6
    "PyQt6.sip",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtOpenGL",
    "PyQt6.QtOpenGLWidgets",
    "PyQt6.QtPrintSupport",
    "PyQt6.QtSvg",
    # scipy internals
    "scipy.special._ufuncs_cxx",
    "scipy.spatial.transform._rotation_groups",
    "scipy._lib.messagestream",
    "scipy.io.matlab.mio5_utils",
    "scipy.io.matlab.streams",
    "scipy._lib._array_api_compat_vendor",
    "scipy._lib._uarray",
    "scipy._lib.array_api_compat",
    "scipy.linalg._fblas",
    "scipy.linalg._flapack",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "scipy.linalg._decomp_update",
    "scipy.sparse.csgraph._flow",
    "scipy.sparse.csgraph._matching",
    "scipy.sparse.csgraph._min_spanning_tree",
    "scipy.sparse.csgraph._reordering",
    "scipy.sparse.csgraph._shortest_path",
    "scipy.sparse.csgraph._tools",
    "scipy.sparse.csgraph._traversal",
    "scipy.sparse._csparsetools",
    "scipy.sparse._sparsetools",
    "scipy.special._cython_special",
    "scipy.special._ufuncs",
    "scipy.special.cython_special",
    "scipy.io._harwell_boeing.hb",
    "scipy.io.matlab._mio5_utils",
    "scipy.io.matlab._mio_utils",
    "scipy.io.matlab._streams",
    # imaging / dicom
    "nibabel",
    "nibabel.nifti1",
    "pydicom",
    "pydicom.encoders.gdcm",
    "pydicom.encoders.pylibjpeg",
    "pylibjpeg",
    # SimpleITK
    "SimpleITK",
    # Jupyter kernel (napari console uses this)
    "ipykernel",
    "ipykernel.kernelapp",
    "jupyter_client",
    "comm",
    "nest_asyncio",
    # misc
    "colorama",
    "appdirs",
    "platformdirs",
    "yaml",
    "mat73",
    # our backend
    "quantus",
    "quantus.plugin_loader",
    "quantus.full_workflow",
    "quantus.entrypoints",
    "quantus.image_loading",
    "quantus.seg_loading",
    "quantus.analysis",
    "quantus.visualizations",
    "quantus.data_export",
    "quantus.data_objs",
]

all_hiddenimports += _EXTRA_HIDDEN

# Analysis
a = Analysis(
    ["qus_gui.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=["."],          # look for hook-*.py files in repo root
    hooksconfig={},
    runtime_hooks=["rthook_quantus.py"],
    excludes=[
        "ceus_gui",           # don't drag in the CEUS side
        "tkinter",            # Qt app doesn't need Tk
        "matplotlib.backends.backend_tkagg",
    ],
    cipher=block_cipher,
    noarchive=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QuantUS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                # don't UPX — breaks VTK/Qt binaries
    console=True,            # no black terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="Images/logo.ico" if sys.platform == "win32" else (
         "Images/logo.icns" if sys.platform == "darwin" else None
    ),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="QuantUS",
)

# macOS .app bundle (ignored on Windows/Linux)
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="QuantUS.app",
        icon="Images/logo.icns",
        bundle_identifier="edu.quantus.qus",
        info_plist={
            "NSHighResolutionCapable": "True",
            "NSRequiresAquaSystemAppearance": "False",   # respect dark mode
            "CFBundleShortVersionString": "0.1.0",
        },
    )