#!/usr/bin/env bash
# Build QuantUS (QUS) as a macOS .app. Usage: ./build_qus.sh [--dmg]
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.11}"
VENV=".build_venv"
MAKE_DMG=0
[ "${1:-}" = "--dmg" ] && MAKE_DMG=1

echo "==> Pre-flight checks"
command -v "$PYTHON" >/dev/null || { echo "ERROR: $PYTHON not found (install Python 3.11 from python.org)"; exit 1; }
[ -f engines/qus/pyproject.toml ] || { echo "ERROR: engines/qus is empty. Run: git submodule update --init --recursive"; exit 1; }
# Without this, a missing Xcode license only fails at the very END of a 30 minute build
lipo -info /bin/ls >/dev/null 2>&1 || { echo "ERROR: Xcode tools unusable. Run: sudo xcodebuild -license accept"; exit 1; }

echo "==> Fresh environment"
rm -rf "$VENV" dist build dist_wheels engines/qus/build engines/qus/*.egg-info
"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"   # also puts pyuic6 on PATH for saveQt.sh

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt     # pinned versions
python -m pip install pyinstaller==6.10.0

echo "==> Building backend wheel (never use an editable install for freezing)"
python -m pip wheel ./engines/qus --no-deps --no-build-isolation -w dist_wheels
python -m pip install dist_wheels/quantus-*.whl   # adds backend deps missing from requirements.txt

python - <<'EOF'
import quantus
assert "site-packages" in quantus.__file__, f"quantus imported from source tree: {quantus.__file__}"
print("quantus OK:", quantus.__file__)
EOF

echo "==> Icon and Qt UI files"
if [ ! -f Images/logo.icns ]; then
python - <<'EOF'
from PIL import Image
Image.open("Images/logo.png").convert("RGBA").resize((1024, 1024)).save("Images/logo.icns")
EOF
fi
chmod +x saveQt.sh
./saveQt.sh

echo "==> PyInstaller (takes 5-30 min)"
python -m PyInstaller quantus_qus.spec --clean --noconfirm

[ -d dist/QuantUS.app ] || { echo "ERROR: dist/QuantUS.app was not produced"; exit 1; }
if find dist/QuantUS.app -type d -name engines | grep -q .; then
  echo "WARNING: duplicate engines/ directory found in the bundle"
fi

if [ "$MAKE_DMG" = "1" ]; then
  echo "==> DMG"
  rm -rf dmg_staging && mkdir -p dmg_staging installer_output
  cp -R dist/QuantUS.app dmg_staging/
  ln -s /Applications dmg_staging/Applications
  hdiutil create -volname "QuantUS" -srcfolder dmg_staging -ov -format UDZO installer_output/QuantUS-0.1.0-mac.dmg
  rm -rf dmg_staging
fi

echo "==> Done: dist/QuantUS.app"