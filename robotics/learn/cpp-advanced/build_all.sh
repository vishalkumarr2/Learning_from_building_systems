#!/usr/bin/env bash
# Build all (or specific) exercise modules
# Usage: ./build_all.sh        # build everything
#        ./build_all.sh 03     # build only week 03
#        ./build_all.sh 08     # build capstone
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTER="${1:-}"   # optional: "01", "02", ..., "10"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

build_module() {
    local dir="$1"
    local name
    name="$(basename "$dir")"
    
    if [[ -n "$FILTER" && ! "$name" == "$FILTER"* ]]; then
        return 0
    fi

    # Find CMakeLists.txt — could be in exercises/ or at top level
    local cmake_dir=""
    if [[ -f "$dir/exercises/CMakeLists.txt" ]]; then
        cmake_dir="$dir/exercises"
    elif [[ -f "$dir/CMakeLists.txt" ]]; then
        cmake_dir="$dir"
    else
        echo -e "${YELLOW}SKIP${NC} $name (no CMakeLists.txt)"
        return 0
    fi

    echo -e "${YELLOW}BUILD${NC} $name ..."
    local build_dir="$cmake_dir/build"
    mkdir -p "$build_dir"
    
    if cmake -S "$cmake_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tail -1 && \
       cmake --build "$build_dir" -j"$(nproc)" 2>&1 | tail -3; then
        echo -e "${GREEN}  OK${NC} $name"
    else
        echo -e "${RED}FAIL${NC} $name"
        return 1
    fi
}

echo "=== Advanced C++ Study Plan — Build ==="
echo ""

FAILED=0
for dir in "$SCRIPT_DIR"/0*/ "$SCRIPT_DIR"/1*/; do
    [[ -d "$dir" ]] || continue
    build_module "$dir" || FAILED=$((FAILED + 1))
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All modules built successfully.${NC}"
else
    echo -e "${RED}$FAILED module(s) failed.${NC}"
    exit 1
fi
