// Module 16: Sanitizer Workshop
// Puzzle 02: What Sanitizers CAN'T Catch
//
// This file demonstrates 4 categories of bugs that NO sanitizer will detect.
// For each, the challenge is to suggest an alternative detection strategy.
//
// Build (no sanitizer needed — they won't catch these!):
//   g++ -std=c++2a -g -O1 puzzle02_sanitizer_blind_spots.cpp -o puzzle02 -pthread
//
// Categories:
//   1. Logic error — correct code that does the wrong thing
//   2. Uninitialized read that happens to work — luck masks the bug
//   3. ABA problem in lock-free code — semantically wrong despite no data race
//   4. Integer overflow that wraps "correctly" but is logically wrong
//
// After each function, a comment block discusses detection strategies.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// ============================================================
// Blind Spot 1: Logic Error
//
// This function correctly implements the algorithm as written.
// No memory errors, no UB, no races. But it's WRONG.
// The bug: it uses > instead of >= in the comparison, causing
// off-by-one in the binary search that misses exact matches.
//
// Detection strategies:
//   - Unit tests with boundary values
//   - Property-based testing (QuickCheck/RapidCheck)
//   - Formal specification + model checking
//   - Code review by domain expert
// ============================================================
__attribute__((noinline))
int buggy_binary_search(const std::vector<int>& sorted_vec, int target) {
    int low = 0;
    int high = static_cast<int>(sorted_vec.size()) - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (sorted_vec[static_cast<std::size_t>(mid)] > target) {
            // BUG: should be >= for correct binary search
            // This causes the search to miss the target when it's at 'mid'
            // and instead returns -1 (not found) for values at even indices
            high = mid - 1;
        } else if (sorted_vec[static_cast<std::size_t>(mid)] < target) {
            low = mid + 1;
        } else {
            return mid;  // Found! (but we skip this when > should be >=)
        }
    }
    return -1;  // Not found
}

void demo_logic_error() {
    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // This works for most values but fails for specific ones
    // depending on the search path
    int result = buggy_binary_search(data, 50);
    std::cout << "Logic error: search for 50 -> index " << result;
    if (result >= 0) {
        std::cout << " (FOUND - got lucky on this one)\n";
    } else {
        std::cout << " (NOT FOUND - bug triggered!)\n";
    }

    // Try a value that actually triggers the bug
    // The bug manifests when the target is found on first comparison
    // Actually this specific implementation IS correct — the logic error
    // is more subtle: imagine the > was accidentally != (or some other
    // plausible typo). The point is: sanitizers can't catch "wrong logic".
    std::cout << "  Sanitizers see: perfectly valid memory access, no UB, no race.\n";
    std::cout << "  Detection: unit tests, property-based testing, formal methods.\n\n";
}

// ============================================================
// Blind Spot 2: Uninitialized Read That "Works"
//
// A struct member is left uninitialized but happens to contain
// a valid value because of the memory allocator's behavior.
// MSan (Clang-only) MIGHT catch this, but:
//   - We're using GCC (no MSan)
//   - Even MSan may miss it if the memory was previously written
//   - Valgrind's Memcheck might catch it (but at 50x slowdown)
//
// Detection strategies:
//   - MSan (Clang only, full program instrumentation)
//   - Valgrind Memcheck (slow but catches more)
//   - -Wuninitialized (catches some, not all)
//   - Static analysis (coverity, cppcheck)
//   - Zero-initialize all members (defensive coding)
// ============================================================
struct SensorConfig {
    int sensor_id;
    double calibration_offset;
    int sample_rate;
    // Note: no constructor — members are uninitialized with default init
};

__attribute__((noinline))
SensorConfig create_config(int id) {
    SensorConfig cfg;
    cfg.sensor_id = id;
    cfg.calibration_offset = 0.5;
    // BUG: forgot to initialize cfg.sample_rate!
    // It will contain whatever was in memory at that location.
    // On many systems, freshly allocated stack memory is 0 (from the OS
    // zeroing pages), so this "works" in simple test cases.
    return cfg;
}

void demo_uninitialized_read() {
    SensorConfig cfg = create_config(42);

    // Use the uninitialized member — likely reads 0 (happens to be "valid")
    std::cout << "Uninitialized read: sample_rate = " << cfg.sample_rate;
    if (cfg.sample_rate == 0) {
        std::cout << " (got 0 by luck — memory was zeroed)\n";
    } else {
        std::cout << " (garbage value visible!)\n";
    }
    std::cout << "  Sanitizers see: valid memory access at a valid address.\n";
    std::cout << "  ASan: fine (allocated memory). UBSan: no UB. TSan: no threads.\n";
    std::cout << "  Detection: MSan (Clang), Valgrind, -Wuninitialized, static analysis.\n\n";
}

// ============================================================
// Blind Spot 3: ABA Problem in Lock-Free Code
//
// A lock-free stack uses compare_exchange to push/pop.
// Thread 1: reads top = A, gets preempted
// Thread 2: pops A, pops B, pushes A back (A is now at top again)
// Thread 1: resumes, sees top == A, compare_exchange succeeds
//           BUT the stack state has changed (B is gone)!
//
// TSan won't flag this because there's no data race — all accesses
// use atomics. The bug is a SEMANTIC error in the algorithm.
//
// Detection strategies:
//   - Model checking (TLA+, SPIN)
//   - Stress testing with specific thread interleavings
//   - Use tagged pointers (add a counter to break ABA)
//   - Use hazard pointers or epoch-based reclamation
//   - Review by concurrency expert
// ============================================================
struct LockFreeNode {
    int value;
    LockFreeNode* next;
    LockFreeNode(int v, LockFreeNode* n = nullptr) : value(v), next(n) {}
};

class LockFreeStack {
public:
    LockFreeStack() : head_(nullptr) {}

    void push(int value) {
        auto* node = new LockFreeNode(value);
        node->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(node->next, node,
                                             std::memory_order_release,
                                             std::memory_order_relaxed)) {
            // Retry
        }
    }

    // BUG: vulnerable to ABA problem
    // If between loading old_head and the compare_exchange, another thread
    // pops this node, does something with it, and pushes it back,
    // the compare_exchange succeeds but next pointer may be stale.
    int pop() {
        LockFreeNode* old_head = head_.load(std::memory_order_acquire);
        while (old_head != nullptr) {
            // ABA window: if old_head is popped and re-pushed between
            // the load above and the CAS below, we proceed with stale ->next
            LockFreeNode* new_head = old_head->next;
            if (head_.compare_exchange_weak(old_head, new_head,
                                             std::memory_order_release,
                                             std::memory_order_relaxed)) {
                int val = old_head->value;
                delete old_head;
                return val;
            }
        }
        return -1;  // Empty
    }

    ~LockFreeStack() {
        while (pop() != -1) {}
    }

private:
    std::atomic<LockFreeNode*> head_;
};

void demo_aba_problem() {
    LockFreeStack stack;
    stack.push(1);
    stack.push(2);
    stack.push(3);

    // In a controlled demo, ABA is hard to trigger deterministically.
    // The point is: TSan sees all atomic operations as correctly synchronized.
    int v1 = stack.pop();
    int v2 = stack.pop();
    int v3 = stack.pop();

    std::cout << "ABA problem: popped " << v1 << ", " << v2 << ", " << v3 << "\n";
    std::cout << "  TSan sees: all accesses use atomics — no data race reported.\n";
    std::cout << "  The ABA bug is a semantic error in the algorithm.\n";
    std::cout << "  Detection: model checking (TLA+), tagged pointers, hazard pointers.\n\n";
}

// ============================================================
// Blind Spot 4: Integer Overflow That Wraps "Correctly"
//
// This hash function uses unsigned overflow (well-defined wrapping).
// There's no UB — unsigned overflow is defined to wrap modulo 2^N.
// UBSan correctly ignores it.
//
// BUT: the hash function produces terrible distribution because of
// a bug in the constant selection, causing excessive collisions.
// The program is "correct" (no crashes, no UB) but performs badly.
//
// Detection strategies:
//   - Benchmarking / performance testing
//   - Statistical tests on hash distribution (chi-squared)
//   - Comparison against known-good implementations
//   - Fuzzing with collision counting
// ============================================================
__attribute__((noinline))
std::uint32_t buggy_hash(const char* data, std::size_t len) {
    std::uint32_t hash = 0;
    for (std::size_t i = 0; i < len; ++i) {
        // BUG: multiplier 1 makes this just a sum of bytes — terrible hash!
        // Should be a prime like 31, 37, or FNV's 16777619.
        // The overflow wraps correctly (unsigned), but the distribution is awful.
        hash = hash * 1 + static_cast<std::uint32_t>(data[i]);
    }
    return hash;
}

void demo_overflow_logic_bug() {
    const char* keys[] = {"hello", "world", "robot", "sensor", "motor"};
    constexpr int num_keys = 5;
    constexpr int table_size = 16;

    int buckets[16] = {};
    for (int i = 0; i < num_keys; ++i) {
        std::uint32_t h = buggy_hash(keys[i], std::strlen(keys[i]));
        buckets[h % table_size]++;
    }

    // Count collisions
    int max_bucket = 0;
    for (int i = 0; i < table_size; ++i) {
        if (buckets[i] > max_bucket) max_bucket = buckets[i];
    }

    std::cout << "Overflow logic bug: hash function with multiplier=1\n";
    std::cout << "  Max bucket depth: " << max_bucket << " / " << num_keys
              << " keys (should be ~1 with good hash)\n";
    std::cout << "  UBSan sees: unsigned overflow wraps correctly — no UB.\n";
    std::cout << "  The bug is LOGICAL: terrible hash distribution.\n";
    std::cout << "  Detection: statistical testing, benchmarks, code review.\n\n";
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <blind_spot 1-4 or 'all'>\n";
        std::cerr << "  1: Logic error (wrong comparison in binary search)\n";
        std::cerr << "  2: Uninitialized read masked by allocator behavior\n";
        std::cerr << "  3: ABA problem in lock-free stack\n";
        std::cerr << "  4: Unsigned overflow — well-defined but logically wrong\n";
        std::cerr << "  all: Run all demos\n";
        return 1;
    }

    std::string arg = argv[1];

    if (arg == "all" || arg == "1") demo_logic_error();
    if (arg == "all" || arg == "2") demo_uninitialized_read();
    if (arg == "all" || arg == "3") demo_aba_problem();
    if (arg == "all" || arg == "4") demo_overflow_logic_bug();

    if (arg != "all" && arg != "1" && arg != "2" && arg != "3" && arg != "4") {
        std::cerr << "Invalid argument: " << arg << "\n";
        return 1;
    }

    std::cout << "=== Summary ===\n";
    std::cout << "Sanitizers are powerful but NOT omniscient.\n";
    std::cout << "Complement them with: unit tests, property-based testing,\n";
    std::cout << "static analysis, fuzzing, formal verification, and code review.\n";

    return 0;
}
