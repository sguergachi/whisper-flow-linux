#!/usr/bin/env bash
# Build a WhisperFlow AppImage — one file, double-click to run.
#
# Usage:
#   make-appimage.sh <stage-dir> <version> <output.AppImage>
#
# stage-dir: wheel + install.sh + gui_install.py + desktop template + units
#            (+ optional whisper-flow.png)
#
# First double-click installs for the current user (desktop dialogs, no
# terminal). Later double-clicks start the tray app. No .deb, no archive.
set -euo pipefail

STAGE="${1:?usage: make-appimage.sh <stage-dir> <version> <output.AppImage>}"
VERSION="${2:?usage: make-appimage.sh <stage-dir> <version> <output.AppImage>}"
OUT="${3:?usage: make-appimage.sh <stage-dir> <version> <output.AppImage>}"

[[ -d "$STAGE" ]] || { echo "error: stage dir not found: $STAGE" >&2; exit 1; }
[[ -f "$STAGE/install.sh" ]] || { echo "error: install.sh missing" >&2; exit 1; }
[[ -f "$STAGE/gui_install.py" ]] || {
    echo "error: gui_install.py missing" >&2
    exit 1
}

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

APPDIR="$WORKDIR/WhisperFlow.AppDir"
PAYLOAD="$APPDIR/usr/share/whisper-flow"
mkdir -p "$PAYLOAD" "$APPDIR/usr/bin" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -a "$STAGE"/. "$PAYLOAD"/
chmod +x "$PAYLOAD/install.sh" "$PAYLOAD/uninstall.sh" "$PAYLOAD/gui_install.py"

# Icon at the AppDir root (required name for appimagetool) and hicolor.
if [[ -f "$PAYLOAD/whisper-flow.png" ]]; then
    cp "$PAYLOAD/whisper-flow.png" "$APPDIR/whisper-flow.png"
    cp "$PAYLOAD/whisper-flow.png" \
        "$APPDIR/usr/share/icons/hicolor/256x256/apps/whisper-flow.png"
else
    # 1x1 placeholder so appimagetool still accepts the tree.
    printf '\x89PNG\r\n\x1a\n' > "$APPDIR/whisper-flow.png"
fi

# Desktop entry at AppDir root. Terminal=false = no shell window on click.
cat > "$APPDIR/whisper-flow.desktop" <<'EOF'
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
X-AppImage-Version=REPLACE_VERSION
EOF
sed -i "s/REPLACE_VERSION/${VERSION}/" "$APPDIR/whisper-flow.desktop"

# AppRun: the thing that actually runs when you double-click the AppImage.
# No terminal. First run → GUI install. After that → start the tray.
cat > "$APPDIR/AppRun" <<'APPRUN'
#!/usr/bin/env bash
set -euo pipefail

# Resolve this AppImage's mounted AppDir (appimagetool sets APPDIR).
HERE="${APPDIR:-}"
if [[ -z "$HERE" ]]; then
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PAYLOAD="$HERE/usr/share/whisper-flow"
PREFIX="${WHISPER_FLOW_PREFIX:-$HOME/.local/share/whisper-flow}"
REAL="$PREFIX/venv/bin/whisper-flow"

# Desktop dialogs only when the file manager launched us (no TTY).
gui_error() {
    local text="$1"
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title=WhisperFlow --width=420 --text="$text" 2>/dev/null || true
    elif command -v kdialog >/dev/null 2>&1; then
        kdialog --title WhisperFlow --error "$text" 2>/dev/null || true
    elif command -v notify-send >/dev/null 2>&1; then
        notify-send "WhisperFlow" "$text" 2>/dev/null || true
    else
        printf '%s\n' "$text" >&2
    fi
}

start_app() {
    # Prefer the user service (starts at login too); fall back to a direct
    # background daemon so a double-click never needs a terminal.
    if command -v systemctl >/dev/null 2>&1; then
        if systemctl --user is-active --quiet whisper-flow.service 2>/dev/null; then
            # Already running — bring settings or just leave the tray alone.
            if command -v notify-send >/dev/null 2>&1; then
                notify-send "WhisperFlow" "Already running — look for the mic in the tray." 2>/dev/null || true
            fi
            exit 0
        fi
        if systemctl --user start whisper-flow.service 2>/dev/null; then
            exit 0
        fi
    fi
    # Direct start, detached from this AppImage mount.
    if [[ -x "$REAL" ]]; then
        nohup "$REAL" daemon >/dev/null 2>&1 &
        exit 0
    fi
    gui_error "WhisperFlow is installed but the launcher is missing.\nTry running the AppImage again, or reinstall."
    exit 1
}

if [[ -x "$REAL" ]]; then
    start_app
fi

# --- first run: install via the GUI (no terminal) --------------------
if [[ ! -f "$PAYLOAD/gui_install.py" ]]; then
    gui_error "This AppImage is incomplete (installer missing)."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    gui_error "Python 3 is required.\nInstall python3 from your software centre, then double-click again."
    exit 1
fi

# gui_install.py expects install.sh beside it (payload layout).
cd "$PAYLOAD"
if ! python3 "$PAYLOAD/gui_install.py"; then
    exit 1
fi

if [[ -x "$REAL" ]]; then
    start_app
fi
exit 0
APPRUN
chmod +x "$APPDIR/AppRun"

# Fetch appimagetool if needed (CI and developer machines).
ARCH=$(uname -m)
case "$ARCH" in
    x86_64|amd64) AI_ARCH=x86_64 ;;
    aarch64|arm64) AI_ARCH=aarch64 ;;
    *) echo "error: unsupported arch $ARCH" >&2; exit 1 ;;
esac

TOOL="$WORKDIR/appimagetool"
if [[ -n "${APPIMAGETOOL:-}" && -x "${APPIMAGETOOL}" ]]; then
    TOOL="$APPIMAGETOOL"
elif ! command -v appimagetool >/dev/null 2>&1; then
    URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${AI_ARCH}.AppImage"
    echo "Downloading appimagetool…"
    curl -fsSL -o "$TOOL" "$URL"
    chmod +x "$TOOL"
    # appimagetool is itself an AppImage; on some CI hosts FUSE is missing,
    # so extract and run the embedded binary.
    if ! "$TOOL" --appimage-help >/dev/null 2>&1; then
        (cd "$WORKDIR" && "$TOOL" --appimage-extract >/dev/null)
        TOOL="$WORKDIR/squashfs-root/AppRun"
    fi
else
    TOOL=$(command -v appimagetool)
fi

# Build into a known path; appimagetool names by desktop Name + arch by default.
export ARCH="$AI_ARCH"
export VERSION
# No runtime download surprises in CI.
export APPIMAGETOOL_APP_UPDATE=0

OUT_ABS=$(mkdir -p "$(dirname "$OUT")" && cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
# appimagetool writes VERSION-arch.AppImage unless -n / absolute out given.
if ! "$TOOL" -n "$APPDIR" "$OUT_ABS" 2>"$WORKDIR/appimagetool.err"; then
    cat "$WORKDIR/appimagetool.err" >&2
    # Older continuous builds used different flags; retry without -n.
    if ! "$TOOL" "$APPDIR" "$OUT_ABS" 2>>"$WORKDIR/appimagetool.err"; then
        cat "$WORKDIR/appimagetool.err" >&2
        exit 1
    fi
fi

chmod +x "$OUT_ABS"
# If the tool ignored our path and wrote next to AppDir, move it.
if [[ ! -f "$OUT_ABS" ]]; then
    found=$(find "$WORKDIR" -maxdepth 2 -name '*.AppImage' ! -name 'appimagetool*' | head -1 || true)
    [[ -n "$found" ]] || { echo "error: AppImage not produced" >&2; exit 1; }
    mv "$found" "$OUT_ABS"
fi

printf 'wrote %s (%s)\n' "$OUT_ABS" "$(du -h "$OUT_ABS" | cut -f1)"
