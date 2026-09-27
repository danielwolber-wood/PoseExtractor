#!/bin/zsh
# Builds a release ClayStudio.app (with the converted SMPL models inside) and the `clay` CLI.
set -euo pipefail
cd "${0:A:h}/.."

[[ -f Models/smpl_neutral/meta.json ]] || uv run --with numpy --with scipy tools/convert_models.py

swift build -c release
BIN=$(swift build -c release --show-bin-path)

APP=build/ClayStudio.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/ClayStudio" "$APP/Contents/MacOS/"
cp -R Models "$APP/Contents/Resources/Models"

# App icon, rendered by the pipeline itself (see Sources/clay-icon).
ICONSET=build/AppIcon.iconset
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
"$BIN/clay-icon" build/icon_1024.png >/dev/null
for s in 16 32 128 256 512; do
  sips -z $s $s build/icon_1024.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) build/icon_1024.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Clay Studio</string>
  <key>CFBundleIdentifier</key><string>local.claystudio</string>
  <key>CFBundleExecutable</key><string>ClayStudio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --sign - "$APP"
cp "$BIN/clay" build/clay
echo "Built $APP and build/clay"
