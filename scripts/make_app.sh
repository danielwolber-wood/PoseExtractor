#!/bin/zsh
# Builds a release Armature.app (with the converted SMPL models inside) and the `armature` CLI.
set -euo pipefail
cd "${0:A:h}/.."

[[ -f Models/smpl_neutral/meta.json ]] || uv run --with numpy --with scipy tools/convert_models.py

swift build -c release
BIN=$(swift build -c release --show-bin-path)

APP=build/Armature.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/ArmatureApp" "$APP/Contents/MacOS/Armature"
cp -R Models "$APP/Contents/Resources/Models"

# App icon, from the Icon Composer file. actool ships with Xcode (not the Command Line Tools) and names
# the compiled icon after the source file, so compile a copy called AppIcon.icon.
ICON_SRC=design/icon/v1.icon
DEV=$(xcode-select -p)
[[ -x "$DEV/usr/bin/actool" ]] || DEV=/Applications/Xcode.app/Contents/Developer
rm -rf build/AppIcon.icon build/AppIcon.iconset build/icon_1024.png
cp -R "$ICON_SRC" build/AppIcon.icon
if ! DEVELOPER_DIR="$DEV" xcrun actool build/AppIcon.icon --compile "$APP/Contents/Resources" --app-icon AppIcon \
    --platform macosx --target-device mac --minimum-deployment-target 14.0 \
    --output-partial-info-plist build/AppIcon-partial.plist --output-format human-readable-text >/dev/null; then
  echo "actool couldn't compile $ICON_SRC. It needs Xcode 26 or later with its license accepted" >&2
  echo "(sudo xcodebuild -license accept; xcodebuild -runFirstLaunch)." >&2
  exit 1
fi
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Armature</string>
  <key>CFBundleIdentifier</key><string>local.armature</string>
  <key>CFBundleExecutable</key><string>Armature</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIconName</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --sign - "$APP"
cp "$BIN/armature" build/armature
echo "Built $APP and build/armature"
