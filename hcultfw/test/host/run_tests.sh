#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmake -B "$SCRIPT_DIR/build" -S "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SCRIPT_DIR/build"
ctest --test-dir "$SCRIPT_DIR/build" --output-on-failure
