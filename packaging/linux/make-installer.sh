#!/usr/bin/env bash
# Build a single self-extracting Linux installer from a staged payload directory.
#
# Usage:
#   make-installer.sh <stage-dir> <output-path>
#
# The stage dir must contain install.sh and the wheel (same layout the CI
# package job assembles). The output is one executable: run it and it installs
# whisper-flow for the current user, the same way a Windows setup.exe does.
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

# Flatten to the archive root so extract puts install.sh at the top level.
tar -C "$STAGE" -czf "$WORKDIR/payload.tar.gz" .

# Header template. PAYLOAD_OFFSET is rewritten below to the first byte of the
# embedded tar.gz so extraction never greps through binary data (grep refuses
# to print line matches from "binary" files without -a, and a false match
# inside the gzip stream would be worse).
cat > "$WORKDIR/header.sh" <<'HEADER'
#!/usr/bin/env bash
# whisper-flow Linux installer — single executable, no archive to unpack.
# Download, chmod +x, run. Installs for the current user under ~/.local.
set -euo pipefail

# Byte offset of the gzip payload; filled in by make-installer.sh.
PAYLOAD_OFFSET=@@PAYLOAD_OFFSET@@

SELF="${BASH_SOURCE[0]:-$0}"
if command -v realpath >/dev/null 2>&1; then
    SELF=$(realpath "$SELF")
elif command -v readlink >/dev/null 2>&1; then
    SELF=$(readlink -f "$SELF" 2>/dev/null || echo "$SELF")
fi

[[ -f "$SELF" ]] || {
    printf '\033[1;31m error:\033[0m cannot locate this installer (%s)\n' \
        "$SELF" >&2
    exit 1
}

TMP=$(mktemp -d)
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

printf '\033[1;34m==>\033[0m Extracting whisper-flow\n'
# Skip the script header; everything from PAYLOAD_OFFSET is the tar.gz.
tail -c +"$PAYLOAD_OFFSET" "$SELF" | tar -xz -C "$TMP"

INSTALL="$TMP/install.sh"
[[ -f "$INSTALL" ]] || {
    printf '\033[1;31m error:\033[0m install.sh missing from payload\n' >&2
    exit 1
}
chmod +x "$INSTALL"
exec bash "$INSTALL"
HEADER

# First payload byte is 1-based for tail -c (tail -c +N starts at byte N).
# Measure the header with a stand-in the same width as the final digits so
# the offset does not shift when we substitute. 16 digits is enough for any
# header we will ever write.
PLACEHOLDER='0000000000000000'
sed "s/@@PAYLOAD_OFFSET@@/${PLACEHOLDER}/" "$WORKDIR/header.sh" \
    > "$WORKDIR/header.pad"
HEADER_BYTES=$(wc -c < "$WORKDIR/header.pad")
# tail -c +N is 1-based: first payload byte is HEADER_BYTES + 1.
OFFSET=$((HEADER_BYTES + 1))
# Zero-pad to the same width so the header size stays exactly HEADER_BYTES.
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

# Self-check: extract the payload from the built file and list install.sh.
# tar may list "install.sh" or "./install.sh" depending on version/flags.
LIST=$(tail -c +"$OFFSET" "$OUT" | tar -tz)
echo "$LIST" | grep -qE '(^|./)install\.sh$' || {
    echo "error: built installer does not contain install.sh" >&2
    echo "$LIST" >&2
    exit 1
}

printf 'wrote %s (%s, payload at byte %s)\n' \
    "$OUT" "$(du -h "$OUT" | cut -f1)" "$OFFSET"
