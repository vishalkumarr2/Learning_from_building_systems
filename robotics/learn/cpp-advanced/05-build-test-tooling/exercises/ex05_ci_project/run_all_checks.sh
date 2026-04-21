#!/usr/bin/env bash
# ============================================================================
# run_all_checks.sh — Run all CI checks and report pass/fail
#
# Usage:
#   cd ex05_ci_project
#   ./run_all_checks.sh
#
# Exit code: 0 if all checks pass, 1 if any fail.
#
# Checks run in order:
#   1. Debug build
#   2. Release build
#   3. Unit tests
#   4. AddressSanitizer + UBSan
#   5. ThreadSanitizer
#   6. UBSan (standalone)
#   7. Code coverage (≥80% gate)
#   8. Fuzzing (60 seconds, requires clang)
#
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULTS=()

run_check() {
    local name="$1"
    local target="$2"
    local required="${3:-true}"

    printf "${BOLD}[%d] Running: %s${NC}\n" $((PASS_COUNT + FAIL_COUNT + SKIP_COUNT + 1)) "$name"

    if make "$target" 2>&1 | tail -20; then
        RESULTS+=("${GREEN}PASS${NC}  $name")
        ((PASS_COUNT++))
    else
        if [ "$required" = "true" ]; then
            RESULTS+=("${RED}FAIL${NC}  $name")
            ((FAIL_COUNT++))
        else
            RESULTS+=("${YELLOW}SKIP${NC}  $name (optional)")
            ((SKIP_COUNT++))
        fi
    fi
    echo ""
}

# Header
echo ""
printf "${BOLD}══════════════════════════════════════════════════${NC}\n"
printf "${BOLD}  C++ CI Pipeline — All Checks${NC}\n"
printf "${BOLD}══════════════════════════════════════════════════${NC}\n"
echo ""

# Run checks
run_check "Debug Build"          "build"
run_check "Release Build"        "build-release"
run_check "Unit Tests"           "test"
run_check "ASan + UBSan Tests"   "test-asan"
run_check "TSan Tests"           "test-tsan"
run_check "UBSan Tests"          "test-ubsan"
run_check "Code Coverage (≥80%)" "coverage"

# Fuzzing is optional (requires clang)
if command -v clang++ &> /dev/null; then
    run_check "Fuzzing (60s)" "fuzz"
else
    RESULTS+=("${YELLOW}SKIP${NC}  Fuzzing (clang++ not found)")
    ((SKIP_COUNT++))
    echo "  Skipping fuzzing: clang++ not available"
fi

# Summary
echo ""
printf "${BOLD}══════════════════════════════════════════════════${NC}\n"
printf "${BOLD}  Results${NC}\n"
printf "${BOLD}══════════════════════════════════════════════════${NC}\n"
for r in "${RESULTS[@]}"; do
    printf "  %b\n" "$r"
done
echo ""
printf "  Passed: ${GREEN}%d${NC}  Failed: ${RED}%d${NC}  Skipped: ${YELLOW}%d${NC}\n" \
    "$PASS_COUNT" "$FAIL_COUNT" "$SKIP_COUNT"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    printf "${RED}${BOLD}PIPELINE FAILED${NC}\n"
    exit 1
else
    printf "${GREEN}${BOLD}PIPELINE PASSED${NC}\n"
    exit 0
fi
