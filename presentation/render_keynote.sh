#!/bin/bash
# Convert the generated .pptx with Keynote (macOS): writes .key, .pdf and one PNG per slide.
#
#   bash presentation/render_keynote.sh presentation/out/SRP_Group7_Final.pptx
#
# Keynote opens briefly; the source .pptx is never modified.
set -euo pipefail
PPTX="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
BASE="${PPTX%.pptx}"
rm -rf "${BASE}_slides"; mkdir -p "${BASE}_slides"

osascript - "$PPTX" "$BASE.key" "$BASE.pdf" "${BASE}_slides" <<'APPLESCRIPT'
on run argv
    set inPath to item 1 of argv
    set keyPath to item 2 of argv
    set pdfPath to item 3 of argv
    set imgDir to item 4 of argv
    tell application "Keynote"
        set theDoc to open (POSIX file inPath)
        delay 2
        export theDoc to (POSIX file pdfPath) as PDF with properties {PDF image quality:Best}
        export theDoc to (POSIX file imgDir) as slide images with properties {image format:PNG, skipped slides:false}
        save theDoc in (POSIX file keyPath)
        close theDoc saving no
    end tell
end run
APPLESCRIPT
echo "wrote $BASE.key, $BASE.pdf and ${BASE}_slides/"
