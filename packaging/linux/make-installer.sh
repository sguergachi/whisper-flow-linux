#!/usr/bin/env bash
# Build a single self-extracting Linux installer from a staged payload directory.
#
# Usage:
#   make-installer.sh <stage-dir> <output-path>
#
# Double-click: when the file manager runs this with no terminal, the header
# launches gui_install.py so every message is a desktop dialog — no shell
# window. From a terminal it runs install.sh directly for a normal log.
#
# Note: browsers download files without the executable bit. On GNOME you may
# need right-click → Properties → "Allow executing file as program" once, or
# use the .deb (Software app handles that for you).
set -euo pipefail

STAGE="${1:?usage: make-installer.sh <stage-dir> <output-path>}"
OUT="${2:?usage: make-installer.sh <stage-dir> <output-path>}"

[[ -d "$STAGE" ]] || { echo "error: stage dir not found: $STAGE" >&2; exit 1; }
[[ -f "$STAGE/install.sh" ]] || {
    echo "error: $STAGE/install.sh is missing" >&2
    exit 1
}

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

tar -C "$STAGE" -czf "$WORKDIR/payload.tar.gz" .

cat > "$WORKDIR/header.sh" <<'HEADER'
#!/usr/bin/env bash
# whisper-flow — double-click to install (no terminal needed).
set -euo pipefail

PAYLOAD_OFFSET=@@PAYLOAD_OFFSET@@

SELF="${BASH_SOURCE[0]:-$0}"
if command -v realpath >/dev/null 2>&1; then
    SELF=$(realpath "$SELF")
elif command -v readlink >/dev/null 2>&1; then
    SELF=$(readlink -f "$SELF" 2>/dev/null || echo "$SELF")
fi

[[ -f "$SELF" ]] || {
    printf 'error: cannot locate this installer (%s)\n' "$SELF" >&2
    exit 1
}

TMP=$(mktemp -d)
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

# File managers often run scripts with no TTY; keep UI off the terminal then.
GUI=0
if [[ ! -t 1 ]] && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    GUI=1
fi

if [[ "$GUI" -eq 0 ]]; then
    printf '==> Extracting whisper-flow\n'
fi
tail -c +"$PAYLOAD_OFFSET" "$SELF" | tar -xz -C "$TMP"

if [[ "$GUI" -eq 1 ]] && [[ -f "$TMP/gui_install.py" ]] \
   && command -v python3 >/dev/null 2>&1; then
    exec python3 "$TMP/gui_install.py"
fi

INSTALL="$TMP/install.sh"
[[ -f "$INSTALL" ]] || {
    printf 'error: install.sh missing from payload\n' >&2
    exit 1
}
chmod +x "$INSTALL"
exec bash "$INSTALL"
HEADER

PLACEHOLDER='0000000000000000'
sed "s/@@PAYLOAD_OFFSET@@/${PLACEHOLDER}/" "$WORKDIR/header.sh" \
    > "$WORKDIR/header.pad"
HEADER_BYTES=$(wc -c < "$WORKDIR/header.pad")
OFFSET=$((HEADER_BYTES + 1))
OFFSET_PAD=$(printf '%016d' "$OFFSET")
sed "s/@@PAYLOAD_OFFSET@@/${OFFSET_PAD}/" "$WORKDIR/header.sh" \
    > "$WORKDIR/header.final"

FINAL_BYTES=$(wc -c < "$WORKDIR/header.final")
[[ "$FINAL_BYTES" -eq "$HEADER_BYTES" ]] || {
    echo "error: header size changed after offset substitution" \
         "($FINAL_BYTES vs $HEADER_BYTES)" >&2
    exit 1
}

cat "$WORKDIR/header.final" "$WORKDIR/payload.tar.gz" > "$OUT"
chmod +x "$OUT"

LIST=$(tail -c +"$OFFSET" "$OUT" | tar -tz)
echo "$LIST" | grep -qE '(^|./)install\.sh$' || {
    echo "error: built installer does not contain install.sh" >&2
    echo "$LIST" >&2
    exit 1
}

printf 'wrote %s (%s, payload at byte %s)\n' \
    "$OUT" "$(du -h "$OUT" | cut -f1)" "$OFFSET"
