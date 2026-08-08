#!/usr/bin/env bash
# Wrap a PyInstaller onedir tree as a portable AppImage.
#
# Usage:
#   make-appimage.sh <pyinstaller-dist-dir> <version> <output.AppImage>
#
# dist-dir is packaging/linux/dist/whisper-flow (the COLLECT output): it must
# contain the whisper-flow binary. engine/ and models/ may sit beside it.
#
# The result is one ELF file that double-clicks on most x86_64 Linux desktops
# whose glibc is at least as new as the build host (CI uses Ubuntu 22.04).
# No pip, no system Python package for the app itself.
set -euo pipefail

DIST="${1:?usage: make-appimage.sh <pyinstaller-dist-dir> <version> <output.AppImage>}"
VERSION="${2:?usage: make-appimage.sh <pyinstaller-dist-dir> <version> <output.AppImage>}"
OUT="${3:?usage: make-appimage.sh <pyinstaller-dist-dir> <version> <output.AppImage>}"

[[ -d "$DIST" ]] || { echo "error: dist dir not found: $DIST" >&2; exit 1; }
[[ -x "$DIST/whisper-flow" || -f "$DIST/whisper-flow" ]] || {
    echo "error: $DIST/whisper-flow binary missing — run PyInstaller first" >&2
    exit 1
}

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

APPDIR="$WORKDIR/WhisperFlow.AppDir"
mkdir -p "$APPDIR"

# The whole frozen tree lives at the AppDir root so sys.executable's parent
# (engine/, models/, bundled .so) stays next to the binary.
cp -a "$DIST"/. "$APPDIR"/
chmod +x "$APPDIR/whisper-flow"

# AppImage metadata at the AppDir root.
if [[ -f "$APPDIR/whisper-flow.png" ]]; then
    :
elif [[ -f "$DIST/whisper-flow.png" ]]; then
    cp "$DIST/whisper-flow.png" "$APPDIR/whisper-flow.png"
else
    # Generate from the frozen tree if PIL is on the host; else placeholder.
    python3 - <<'PY' "$APPDIR/whisper-flow.png" 2>/dev/null || printf '\x89PNG\r\n\x1a\n' > "$APPDIR/whisper-flow.png"
import sys
from pathlib import Path
sys.path.insert(0, "src")
try:
    from whisper_flow.icon import APP_COLOR, draw_mic
    draw_mic(256, APP_COLOR).save(Path(sys.argv[1]), format="PNG")
except Exception:
    Path(sys.argv[1]).write_bytes(b"\x89PNG\r\n\x1a\n")
PY
fi

cat > "$APPDIR/whisper-flow.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=WhisperFlow
GenericName=Voice typing
Comment=Hold a key, talk, and the words appear where you type
Exec=AppRun
Icon=whisper-flow
Terminal=false
Categories=Utility;Accessibility;
Keywords=voice;dictation;speech;whisper;transcription;
StartupNotify=false
X-AppImage-Version=${VERSION}
EOF

# AppRun: double-click entry. No terminal. Hands off to the frozen binary.
# Nothing is exported into LD_LIBRARY_PATH: PyInstaller already knows where
# its own libs live, and the bundle contains no host-system libraries, so
# children (sh, notify-send, the engine) never pick up conflicts.
cat > "$APPDIR/AppRun" <<'APPRUN'
#!/usr/bin/env bash
set -euo pipefail
HERE="${APPDIR:-}"
if [[ -z "$HERE" ]]; then
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# The HUD needs gtk4-layer-shell to load before libwayland-client (it is
# dlopen'd through a typelib otherwise, which is too late). LD_PRELOAD is
# read at process start, so it has to be set here, before exec. The daemon
# and the HUD/settings children it spawns all inherit it.
for LAYER in \
    /usr/lib/x86_64-linux-gnu/libgtk4-layer-shell.so.0 \
    /usr/lib64/libgtk4-layer-shell.so.0 \
    /usr/lib/libgtk4-layer-shell.so.0 \
    /usr/local/lib/libgtk4-layer-shell.so.0
do
    if [[ -f "$LAYER" ]]; then
        export LD_PRELOAD="$LAYER${LD_PRELOAD:+:$LD_PRELOAD}"
        break
    fi
done
exec "$HERE/whisper-flow" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# appimagetool
ARCH=$(uname -m)
case "$ARCH" in
    x86_64|amd64) AI_ARCH=x86_64 ;;
    aarch64|arm64) AI_ARCH=aarch64 ;;
    *) echo "error: unsupported arch $ARCH" >&2; exit 1 ;;
esac

TOOL="$WORKDIR/appimagetool"
if [[ -n "${APPIMAGETOOL:-}" && -x "${APPIMAGETOOL}" ]]; then
    TOOL="$APPIMAGETOOL"
elif command -v appimagetool >/dev/null 2>&1; then
    TOOL=$(command -v appimagetool)
else
    URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${AI_ARCH}.AppImage"
    echo "Downloading appimagetool…"
    curl -fsSL -o "$TOOL" "$URL"
    chmod +x "$TOOL"
    if ! "$TOOL" --appimage-help >/dev/null 2>&1; then
        (cd "$WORKDIR" && "$TOOL" --appimage-extract >/dev/null)
        TOOL="$WORKDIR/squashfs-root/AppRun"
    fi
fi

export ARCH="$AI_ARCH"
export VERSION
export APPIMAGETOOL_APP_UPDATE=0

OUT_ABS=$(mkdir -p "$(dirname "$OUT")" && cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
if ! "$TOOL" -n "$APPDIR" "$OUT_ABS" 2>"$WORKDIR/appimagetool.err"; then
    cat "$WORKDIR/appimagetool.err" >&2 || true
    if ! "$TOOL" "$APPDIR" "$OUT_ABS" 2>>"$WORKDIR/appimagetool.err"; then
        cat "$WORKDIR/appimagetool.err" >&2
        exit 1
    fi
fi

if [[ ! -f "$OUT_ABS" ]]; then
    found=$(find "$WORKDIR" -maxdepth 2 -name '*.AppImage' ! -name 'appimagetool*' | head -1 || true)
    [[ -n "$found" ]] || { echo "error: AppImage not produced" >&2; exit 1; }
    mv "$found" "$OUT_ABS"
fi
chmod +x "$OUT_ABS"
printf 'wrote %s (%s)\n' "$OUT_ABS" "$(du -h "$OUT_ABS" | cut -f1)"
