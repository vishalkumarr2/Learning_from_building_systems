// Exercise 7: Memory Corruption — Watchpoint Debugging
//
// This program contains a DELIBERATE memory corruption bug.
// The corruption is subtle: a stale pointer writes through freed memory.
//
// TASK: Use GDB to find the corruption:
//   1. gdb ./ex07_watchpoint_corruption
//   2. break main
//   3. run
//   4. Find the corrupted variable with print commands
//   5. Set a watchpoint: watch -l <address>
//   6. continue — GDB will break at the exact corruption point
//
// TOOL PRACTICE:
//   gdb -ex 'break main' -ex run ./ex07_watchpoint_corruption
//   (gdb) watch sentinel
//   (gdb) continue
//
// The self-test detects corruption and prints diagnostics.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

// ========================================================================
// Bug scenario: Sensor buffer with dangling pointer corruption
// ========================================================================

namespace corruption {

// A sensor reading buffer
struct SensorBuffer {
    uint32_t sentinel_before = 0xDEADBEEF;
    double readings[8] = {};
    uint32_t sentinel_after = 0xCAFEBABE;
    size_t count = 0;

    void add_reading(double value) {
        if (count < 8) {
            readings[count] = value;
            ++count;
        }
    }

    bool check_integrity() const {
        return sentinel_before == 0xDEADBEEF &&
               sentinel_after == 0xCAFEBABE;
    }
};

// A "legacy" component that holds a raw pointer to a buffer
// This simulates a common embedded pattern that causes corruption
class LegacyProcessor {
    double* target_ = nullptr;  // points into SensorBuffer::readings
    size_t offset_ = 0;

public:
    void attach(double* buffer, size_t offset) {
        target_ = buffer;
        offset_ = offset;
    }

    // Write to the attached buffer — DANGEROUS if buffer moved/freed
    void write_filtered_value(double raw) {
        if (target_ != nullptr) {
            // Simple low-pass filter
            double filtered = raw * 0.3 + target_[offset_] * 0.7;
            target_[offset_] = filtered;
        }
    }
};

// Demonstrates the corruption pattern
// Returns true if corruption was detected (expected)
bool demonstrate_corruption() {
    // Phase 1: Create buffer and attach processor
    auto buf = std::make_unique<SensorBuffer>();
    LegacyProcessor proc;
    proc.attach(buf->readings, 0);

    // Phase 2: Normal operation — works fine
    buf->add_reading(1.0);
    proc.write_filtered_value(2.0);  // writes to buf->readings[0]

    if (!buf->check_integrity()) {
        std::printf("  [BUG] Corruption during normal operation!\n");
        return true;
    }

    // Phase 3: Buffer is "reallocated" — simulates vector growth,
    // buffer replacement during config reload, etc.
    double* old_ptr = buf->readings;
    auto new_buf = std::make_unique<SensorBuffer>();
    std::memcpy(new_buf->readings, buf->readings, sizeof(buf->readings));
    new_buf->count = buf->count;

    // Old buffer is freed (reset unique_ptr)
    buf.reset();  // ← buf's memory is freed

    // Phase 4: LegacyProcessor still points to OLD (freed) buffer!
    // This write goes to freed memory — undefined behavior
    proc.write_filtered_value(5.0);  // ← DANGLING POINTER WRITE

    // Phase 5: Check if new_buf got corrupted
    // (In practice, the allocator may reuse the freed block, so the
    //  dangling write might overwrite new_buf's memory)
    // We check new_buf integrity as a proxy
    bool integrity_ok = new_buf->check_integrity();

    std::printf("  new_buf integrity: %s\n", integrity_ok ? "OK" : "CORRUPTED");
    std::printf("  old_ptr was:     %p (now freed)\n", static_cast<void*>(old_ptr));
    std::printf("  new_buf readings: %p\n",
                static_cast<void*>(new_buf->readings));

    // The bug is: proc still writes to old_ptr, which is freed
    // This is undefined behavior regardless of whether we can detect it
    return true;  // Bug exists even if we can't observe corruption
}

// The FIXED version: use indices, not pointers
class SafeProcessor {
    SensorBuffer* owner_ = nullptr;
    size_t offset_ = 0;

public:
    void attach(SensorBuffer* buffer, size_t offset) {
        owner_ = buffer;
        offset_ = offset;
    }

    void detach() {
        owner_ = nullptr;
    }

    void write_filtered_value(double raw) {
        if (owner_ != nullptr && offset_ < 8) {
            double filtered = raw * 0.3 + owner_->readings[offset_] * 0.7;
            owner_->readings[offset_] = filtered;
        }
    }
};

bool demonstrate_fix() {
    auto buf = std::make_unique<SensorBuffer>();
    SafeProcessor proc;
    proc.attach(buf.get(), 0);

    buf->add_reading(1.0);
    proc.write_filtered_value(2.0);
    assert(buf->check_integrity());

    // When replacing buffer: detach first!
    proc.detach();  // ← Key fix: invalidate before freeing

    auto new_buf = std::make_unique<SensorBuffer>();
    std::memcpy(new_buf->readings, buf->readings, sizeof(buf->readings));
    new_buf->count = buf->count;
    buf.reset();

    // Re-attach to new buffer
    proc.attach(new_buf.get(), 0);
    proc.write_filtered_value(5.0);  // Safe: writes to new_buf

    assert(new_buf->check_integrity());
    return true;
}

}  // namespace corruption

// ========================================================================
// Exercise: Stack buffer overflow detectable by watchpoint
// ========================================================================

namespace stack_corruption {

struct ControlBlock {
    uint32_t guard1 = 0xAAAAAAAA;
    int data[4] = {10, 20, 30, 40};
    uint32_t guard2 = 0xBBBBBBBB;
};

// This function has an off-by-one write that corrupts guard2
void buggy_fill(ControlBlock& cb, int value) {
    // Bug: writes 5 elements into a 4-element array
    for (int i = 0; i <= 4; ++i) {  // ← should be i < 4
        cb.data[i] = value + i;
    }
    // cb.data[4] overwrites guard2!
}

void safe_fill(ControlBlock& cb, int value) {
    for (int i = 0; i < 4; ++i) {
        cb.data[i] = value + i;
    }
}

void test() {
    // Test buggy version detects corruption
    ControlBlock cb1;
    assert(cb1.guard2 == 0xBBBBBBBB);
    buggy_fill(cb1, 100);
    // guard2 should be corrupted (overwritten with data[4] = 104)
    // Note: this depends on struct layout, which the compiler controls
    // In practice, this USUALLY works on common platforms
    bool corrupted = (cb1.guard2 != 0xBBBBBBBB);
    std::printf("  buggy_fill guard2: 0x%08X (corrupted=%s)\n",
                cb1.guard2, corrupted ? "YES" : "no");

    // Test safe version
    ControlBlock cb2;
    safe_fill(cb2, 100);
    assert(cb2.guard1 == 0xAAAAAAAA);
    assert(cb2.guard2 == 0xBBBBBBBB);
    assert(cb2.data[0] == 100);
    assert(cb2.data[3] == 103);
}

}  // namespace stack_corruption

// ========================================================================
// Main
// ========================================================================

int main() {
    std::printf("=== Dangling Pointer Corruption ===\n");
    corruption::demonstrate_corruption();

    std::printf("\n=== Fixed Version ===\n");
    corruption::demonstrate_fix();

    std::printf("\n=== Stack Buffer Overflow ===\n");
    stack_corruption::test();

    std::printf("\nex07_watchpoint_corruption: ALL TESTS PASSED\n");
    std::printf("\nGDB Practice:\n");
    std::printf("  gdb ./ex07_watchpoint_corruption\n");
    std::printf("  (gdb) break stack_corruption::buggy_fill\n");
    std::printf("  (gdb) run\n");
    std::printf("  (gdb) print cb.guard2\n");
    std::printf("  (gdb) watch cb.guard2\n");
    std::printf("  (gdb) continue\n");
    std::printf("  -> GDB breaks exactly when guard2 is overwritten!\n");

    return 0;
}
