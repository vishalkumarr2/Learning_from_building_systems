# ex05: CI-Ready C++ Project
#
# Targets:
#   make build          — Debug build
#   make build-release  — Release build
#   make test           — Run tests (debug build)
#   make test-asan      — Build & test with AddressSanitizer + UBSan
#   make test-tsan      — Build & test with ThreadSanitizer
#   make test-ubsan     — Build & test with UBSan standalone
#   make coverage       — Build with coverage, run tests, generate HTML report
#   make fuzz           — 60-second fuzzing run (requires clang)
#   make clean          — Remove all build directories
#
# Dependencies:
#   - CMake >= 3.20, Ninja (or Make)
#   - gcc or clang (clang required for fuzzing)
#   - lcov + genhtml (for coverage)
#   - Internet access (FetchContent downloads GoogleTest)

.PHONY: build build-release test test-asan test-tsan test-ubsan coverage fuzz clean

BUILD_DIR      := build
ASAN_DIR       := build-asan
TSAN_DIR       := build-tsan
UBSAN_DIR      := build-ubsan
COV_DIR        := build-coverage
FUZZ_DIR       := build-fuzz
COVERAGE_HTML  := coverage_html
FUZZ_CORPUS    := fuzz_corpus
FUZZ_TIME      := 60

CMAKE_COMMON   := cmake -G Ninja
SRC_DIR        := ..

# ─── Build Targets ──────────────────────────────────────────────

build:
	$(CMAKE_COMMON) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug $(SRC_DIR)
	cmake --build $(BUILD_DIR) -j$$(nproc)

build-release:
	$(CMAKE_COMMON) -B $(BUILD_DIR)-release -DCMAKE_BUILD_TYPE=Release $(SRC_DIR)
	cmake --build $(BUILD_DIR)-release -j$$(nproc)

# ─── Test Targets ───────────────────────────────────────────────

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure -j$$(nproc)

test-asan:
	$(CMAKE_COMMON) -B $(ASAN_DIR) \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
		-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
		$(SRC_DIR)
	cmake --build $(ASAN_DIR) -j$$(nproc)
	cd $(ASAN_DIR) && ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
		ctest --output-on-failure -j$$(nproc)

test-tsan:
	$(CMAKE_COMMON) -B $(TSAN_DIR) \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=thread -fno-omit-frame-pointer" \
		-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
		$(SRC_DIR)
	cmake --build $(TSAN_DIR) -j$$(nproc)
	cd $(TSAN_DIR) && TSAN_OPTIONS=halt_on_error=1 \
		ctest --output-on-failure -j$$(nproc)

test-ubsan:
	$(CMAKE_COMMON) -B $(UBSAN_DIR) \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
		-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined" \
		$(SRC_DIR)
	cmake --build $(UBSAN_DIR) -j$$(nproc)
	cd $(UBSAN_DIR) && ctest --output-on-failure -j$$(nproc)

# ─── Coverage ───────────────────────────────────────────────────

coverage:
	$(CMAKE_COMMON) -B $(COV_DIR) \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS="-O0 -g --coverage -fprofile-arcs -ftest-coverage" \
		-DCMAKE_EXE_LINKER_FLAGS="--coverage" \
		$(SRC_DIR)
	cmake --build $(COV_DIR) -j$$(nproc)
	cd $(COV_DIR) && ctest --output-on-failure
	lcov --capture --directory $(COV_DIR) --output-file $(COV_DIR)/coverage.info
	lcov --remove $(COV_DIR)/coverage.info '/usr/*' '*/googletest/*' '*/test/*' \
		-o $(COV_DIR)/filtered.info
	genhtml $(COV_DIR)/filtered.info --output-directory $(COVERAGE_HTML)
	@echo ""
	@echo "Coverage report: $(COVERAGE_HTML)/index.html"
	@COVERAGE=$$(lcov --summary $(COV_DIR)/filtered.info 2>&1 | grep 'lines' | grep -oP '\d+\.\d+'); \
	echo "Line coverage: $${COVERAGE}%"; \
	if [ $$(echo "$${COVERAGE} < 80.0" | bc -l) -eq 1 ]; then \
		echo "FAIL: Coverage $${COVERAGE}% < 80% threshold"; \
		exit 1; \
	else \
		echo "PASS: Coverage meets 80% threshold"; \
	fi

# ─── Fuzzing ────────────────────────────────────────────────────

fuzz:
	@which clang++ > /dev/null 2>&1 || (echo "ERROR: clang++ required for fuzzing" && exit 1)
	$(CMAKE_COMMON) -B $(FUZZ_DIR) \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS="-O1 -g -fsanitize=fuzzer,address -fno-omit-frame-pointer" \
		-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=fuzzer,address" \
		-DENABLE_FUZZING=ON \
		$(SRC_DIR)
	cmake --build $(FUZZ_DIR) -j$$(nproc) --target ex03_fuzzer
	mkdir -p $(FUZZ_CORPUS)
	$(FUZZ_DIR)/ex03_fuzzer $(FUZZ_CORPUS)/ -max_total_time=$(FUZZ_TIME)
	@echo "PASS: Fuzzing completed without crashes"

# ─── Clean ──────────────────────────────────────────────────────

clean:
	rm -rf $(BUILD_DIR) $(BUILD_DIR)-release \
		$(ASAN_DIR) $(TSAN_DIR) $(UBSAN_DIR) \
		$(COV_DIR) $(FUZZ_DIR) \
		$(COVERAGE_HTML) $(FUZZ_CORPUS)
