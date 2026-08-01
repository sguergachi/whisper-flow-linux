#!/usr/bin/env bash
# Install whisper-flow for the current user.
#
# Everything lands under the user's own directories - no root, no files in
# /usr - because the daemon has to run inside the graphical session to reach
# the compositor, and a system service cannot.
set -euo pipefail

PREFIX="${WHISPER_FLOW_PREFIX:-$HOME/.local/share/whisper-flow}"
BIN_DIR="${WHISPER_FLOW_BIN:-$HOME/.local/bin}"
UNIT_DIR="${WHISPER_FLOW_UNIT_DIR:-$HOME/.config/systemd/user}"
CONFIG_DIR="$HOME/.config/whisper-flow"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warning:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checks --
[[ "$(uname -s)" == "Linux" ]] || die "this installer is for Linux"
command -v python3 >/dev/null || die "python3 not found"

PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 - <<'PY' || die "Python 3.11 or newer is required (found $PYVER)"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY

say "Checking runtime dependencies"
missing=()
have_gi=$(python3 -c 'import gi' 2>/dev/null && echo yes || echo no)
[[ "$have_gi" == yes ]] || missing+=("python-gobject (PyGObject)")
python3 - <<'PY' 2>/dev/null || missing+=("gtk4 + gobject-introspection")
import gi
gi.require_version("Gtk", "4.0")
PY
python3 - <<'PY' 2>/dev/null || missing+=("gtk4-layer-shell")
import gi
gi.require_version("Gtk4LayerShell", "1.0")
PY
command -v ydotool >/dev/null || missing+=("ydotool (typing on Wayland)")
command -v kdotool >/dev/null || warn "kdotool not found - the HUD cannot follow the focused window"

if (( ${#missing[@]} )); then
    warn "missing system packages:"
    printf '         - %s\n' "${missing[@]}"
    echo
    echo "  Arch:     sudo pacman -S python-gobject gtk4 gtk4-layer-shell ydotool"
    echo "  Fedora:   sudo dnf install python3-gobject gtk4 gtk4-layer-shell ydotool"
    echo "  Debian:   sudo apt install python3-gi libgtk-4-1 gir1.2-gtk-4.0 ydotool"
    echo "            (gtk4-layer-shell may need building from source)"
    echo
    [[ "${WHISPER_FLOW_FORCE:-}" == "1" ]] || die "install the above, or re-run with WHISPER_FLOW_FORCE=1"
fi

# The evdev hotkey listener reads /dev/input directly.
if ! id -nG "$USER" | grep -qw input; then
    warn "$USER is not in the 'input' group; global hotkeys will not work."
    echo "         sudo usermod -aG input $USER    # then log out and back in"
fi

# ------------------------------------------------------------- install ----
say "Installing into $PREFIX"
mkdir -p "$PREFIX" "$BIN_DIR" "$UNIT_DIR" "$CONFIG_DIR"

# --system-site-packages so the venv can see PyGObject, which is not
# pip-installable in any reliable way.
python3 -m venv --system-site-packages "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --quiet --upgrade pip
"$PREFIX/venv/bin/pip" install --quiet "$SRC_DIR"

say "Linking commands into $BIN_DIR"
for cmd in whisper-flow whisper-flow-daemon; do
    ln -sf "$PREFIX/venv/bin/$cmd" "$BIN_DIR/$cmd"
done

# --------------------------------------------------------------- units ----
say "Installing systemd user units into $UNIT_DIR"

# Never clobber a unit silently. These get hand-tuned - which GPU build to
# run, which model, extra environment - and losing that to a reinstall is a
# worse outcome than asking.
install_unit() {
    local name="$1" body="$2"
    local dest="$UNIT_DIR/$name"
    if [[ -f "$dest" ]]; then
        if [[ "${WHISPER_FLOW_OVERWRITE_UNITS:-}" != "1" ]]; then
            warn "$name already exists; keeping it"
            printf '%s' "$body" > "$dest.new"
            echo "         new version written to $dest.new"
            return
        fi
        cp "$dest" "$dest.bak.$(date +%Y%m%d-%H%M%S)"
        warn "overwrote $name (previous version kept as .bak.*)"
    fi
    printf '%s' "$body" > "$dest"
}

install_unit whisper-flow.service "$(
    sed "s|@VENV@|$PREFIX/venv|g; s|@HOME@|$HOME|g" \
        "$SRC_DIR/packaging/linux/whisper-flow.service.in"
)"

# Only offered, never assumed: whoever built whisper.cpp chose the binary and
# model, and this template cannot know which.
if [[ ! -f "$UNIT_DIR/whisper-server.service" ]] \
   && [[ -x "$HOME/Dev/whisper.cpp/build/bin/whisper-server" ]]; then
    say "Found a local whisper.cpp build; installing a starter server unit"
    install_unit whisper-server.service "$(
        sed "s|@HOME@|$HOME|g" "$SRC_DIR/packaging/linux/whisper-server.service.in"
    )"
fi

if [[ ! -f "$CONFIG_DIR/.env" ]]; then
    say "Writing a starter config to $CONFIG_DIR/.env"
    cat > "$CONFIG_DIR/.env" <<'EOF'
# Transcription backend. Point at a local whisper.cpp server, or leave the URL
# blank and set an OpenAI key instead.
WHISPER_FLOW_LOCAL_WHISPER_URL=http://127.0.0.1:8082
# WHISPER_FLOW_OPENAI_API_KEY=sk-...

WHISPER_FLOW_HOTKEY_TRANSCRIBE=super+alt
WHISPER_FLOW_HOTKEY_AUTO_TRANSCRIBE=ctrl+alt+space
WHISPER_FLOW_HOTKEY_COMMAND=cmd+shift+alt

# Type words as you speak them rather than all at once on release.
WHISPER_FLOW_LIVE_TRANSCRIPTION=true
EOF
fi

systemctl --user daemon-reload

say "Done."
echo
echo "  Start it:   systemctl --user enable --now whisper-flow"
echo "  Watch it:   journalctl --user -u whisper-flow -f"
echo "  Configure:  $CONFIG_DIR/.env"
echo
if ! printf '%s' "$PATH" | grep -q "$BIN_DIR"; then
    warn "$BIN_DIR is not on your PATH; add it to use the commands directly."
fi
