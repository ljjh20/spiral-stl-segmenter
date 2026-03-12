#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/dxf_spiral_splitter_gui_autochain.py" "$@"
