#!/usr/bin/env bash
# Remove whisper-flow. Leaves ~/.config/whisper-flow alone so settings survive
# a reinstall; delete it by hand to go back to nothing.
set -euo pipefail
PREFIX="${WHISPER_FLOW_PREFIX:-$HOME/.local/share/whisper-flow}"
BIN_DIR="${WHISPER_FLOW_BIN:-$HOME/.local/bin}"

systemctl --user disable --now whisper-flow.service 2>/dev/null || true
systemctl --user disable --now whisper-server.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/whisper-flow.service"
rm -f "$HOME/.config/systemd/user/whisper-server.service"
systemctl --user daemon-reload 2>/dev/null || true

rm -f "$BIN_DIR/whisper-flow" "$BIN_DIR/whisper-flow-daemon"
rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/applications/whisper-flow.desktop"
rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps/whisper-flow.png"
rm -rf "$PREFIX"
echo "Removed. Config kept at ~/.config/whisper-flow"
