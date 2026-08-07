#!/usr/bin/env bash
# Build a .deb you can double-click in the file manager / Software app.
#
# Usage:
#   make-deb.sh <stage-dir> <version> <output-deb-path>
#
# stage-dir must contain the wheel, install.sh, gui_install.py, unit
# templates, desktop template, and uninstall.sh (same payload as the
# self-extracting setup).
set -euo pipefail

STAGE="${1:?usage: make-deb.sh <stage-dir> <version> <output.deb>}"
VERSION="${2:?usage: make-deb.sh <stage-dir> <version> <output.deb>}"
OUT="${3:?usage: make-deb.sh <stage-dir> <version> <output.deb>}"

[[ -d "$STAGE" ]] || { echo "error: stage dir not found: $STAGE" >&2; exit 1; }
[[ -f "$STAGE/install.sh" ]] || { echo "error: install.sh missing" >&2; exit 1; }

# Debian versions must be numeric-ish; strip a leading v if present.
DEB_VERSION="${VERSION#v}"

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

ROOT="$WORKDIR/whisper-flow_${DEB_VERSION}_all"
LIB="$ROOT/usr/lib/whisper-flow"
BIN="$ROOT/usr/bin"
APPS="$ROOT/usr/share/applications"
ICONS="$ROOT/usr/share/icons/hicolor/256x256/apps"
DEBIAN="$ROOT/DEBIAN"

mkdir -p "$LIB" "$BIN" "$APPS" "$ICONS" "$DEBIAN"

# Payload the postinst / first-run path uses.
cp -a "$STAGE"/. "$LIB"/
# Stage dirs from mktemp are often 0700; the package must be world-readable.
chmod -R a+rX "$ROOT"
chmod +x "$LIB"/install.sh "$LIB"/uninstall.sh
[[ -f "$LIB/gui_install.py" ]] && chmod +x "$LIB/gui_install.py"


# Launcher on PATH: never opens a terminal. First run finishes per-user setup.
cat > "$BIN/whisper-flow" <<'WRAP'
#!/usr/bin/env bash
# System wrapper installed by the .deb. Sets up the per-user venv once, then
# hands off to the real CLI. Double-click / app-menu safe (no terminal).
set -euo pipefail
PREFIX="${WHISPER_FLOW_PREFIX:-$HOME/.local/share/whisper-flow}"
LIB=/usr/lib/whisper-flow
REAL="$PREFIX/venv/bin/whisper-flow"

if [[ ! -x "$REAL" ]]; then
    if [[ -x "$LIB/gui_install.py" ]] && command -v python3 >/dev/null; then
        # GUI path when the user launched us from the app menu / file manager.
        if [[ ! -t 1 ]] && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
            exec python3 "$LIB/gui_install.py"
        fi
    fi
    # CLI / postinst path.
    WHISPER_FLOW_NO_START="${WHISPER_FLOW_NO_START:-0}" \
        bash "$LIB/install.sh"
    REAL="$PREFIX/venv/bin/whisper-flow"
fi

exec "$REAL" "$@"
WRAP
chmod +x "$BIN/whisper-flow"

# Same binary name the desktop entry uses for the tray daemon.
ln -sf whisper-flow "$BIN/whisper-flow-daemon"

# Desktop entry: Terminal=false is what keeps double-click from opening a shell.
# Login start is the systemd --user unit install.sh enables, not XDG autostart.
sed 's|@BIN@|/usr/bin|g' "$LIB/whisper-flow.desktop.in" \
    > "$APPS/whisper-flow.desktop"

# App icon (PNG). Prefer a pre-rendered one in the stage; otherwise draw it.
if [[ -f "$STAGE/whisper-flow.png" ]]; then
    cp "$STAGE/whisper-flow.png" "$ICONS/whisper-flow.png"
else
    python3 - <<'PY' "$ICONS/whisper-flow.png" 2>/dev/null || true
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    # When built from a checkout with the package importable.
    from whisper_flow.icon import APP_COLOR, draw_mic
    draw_mic(256, APP_COLOR).save(path, format="PNG")
except Exception:
    # Minimal valid 1x1 PNG so the package still builds in a clean CI stage
    # that has not installed the package yet. install.sh also drops an icon.
    import struct, zlib
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )
    raw = zlib.compress(b"\x00" + b"\x00\x00\x00\xff")
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", raw)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
PY
fi

# Installed-size in KiB for the control file.
SIZE_KB=$(du -sk "$ROOT" | cut -f1)

cat > "$DEBIAN/control" <<EOF
Package: whisper-flow
Version: ${DEB_VERSION}
Section: sound
Priority: optional
Architecture: all
Maintainer: Sammy Guergachi <sguergachi@gmail.com>
Installed-Size: ${SIZE_KB}
Depends: python3 (>= 3.11), python3-gi, python3-venv, python3-pip, gir1.2-gtk-4.0, libgtk-4-1
Recommends: ydotool, zenity | kdialog
Description: Voice typing — hold a key, talk, words appear where you type
 WhisperFlow is a tray app for offline speech-to-text through a local
 whisper.cpp server. Double-click this package to install, then look for
 the microphone in the notification area (or the app menu).
 Hold Super+Alt to dictate.
Homepage: https://github.com/sguergachi/whisper-flow-linux
EOF

cat > "$DEBIAN/postinst" <<'POST'
#!/bin/bash
set -e
# Finish the per-user install for whoever authorised the package install.
# PackageKit / sudo set SUDO_USER; pkexec sets PKEXEC_UID.
user="${SUDO_USER:-}"
if [[ -z "$user" || "$user" == root ]]; then
    if [[ -n "${PKEXEC_UID:-}" ]]; then
        user=$(getent passwd "$PKEXEC_UID" | cut -d: -f1 || true)
    fi
fi

if [[ -n "$user" && "$user" != root ]]; then
    # Run as that user so the venv and systemd --user unit land in their home.
    # No start failure should fail the whole package configure.
    su - "$user" -c \
        'export PATH="$HOME/.local/bin:$PATH"; \
         WHISPER_FLOW_FORCE=1 bash /usr/lib/whisper-flow/install.sh' \
        || true
    # Refresh icon cache if the tool exists.
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -f -t /usr/share/icons/hicolor 2>/dev/null || true
    fi
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database -q /usr/share/applications 2>/dev/null || true
    fi
fi

exit 0
POST
chmod 755 "$DEBIAN/postinst"

cat > "$DEBIAN/prerm" <<'PRERM'
#!/bin/bash
set -e
# Stop the user service when the package is removed; leave config alone.
user="${SUDO_USER:-}"
if [[ -z "$user" || "$user" == root ]]; then
    if [[ -n "${PKEXEC_UID:-}" ]]; then
        user=$(getent passwd "$PKEXEC_UID" | cut -d: -f1 || true)
    fi
fi
if [[ -n "$user" && "$user" != root ]]; then
    su - "$user" -c 'systemctl --user disable --now whisper-flow.service' \
        2>/dev/null || true
fi
exit 0
PRERM
chmod 755 "$DEBIAN/prerm"

# Build. dpkg-deb is on the Ubuntu CI runner; locally on Arch install 'dpkg'.
if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "error: dpkg-deb not found (apt: dpkg, pacman: dpkg)" >&2
    exit 1
fi

# Root ownership inside the archive is what dpkg expects.
dpkg-deb --root-owner-group --build "$ROOT" "$OUT"
printf 'wrote %s (%s)\n' "$OUT" "$(du -h "$OUT" | cut -f1)"
