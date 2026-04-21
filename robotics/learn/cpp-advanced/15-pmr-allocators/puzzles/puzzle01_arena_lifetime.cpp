// =============================================================================
// Puzzle 01: Arena Lifetime — Dangling Pointers After Resource Destruction
// =============================================================================
// THE TRAP: When a memory resource is destroyed, all memory it managed becomes
//           invalid. But there's NO error, NO exception, NO sanitizer warning
//           (in most cases). Just silent corruption.
//
// CHALLENGE: Design a "safe arena" that detects when objects outlive the arena.
//
// Compile: g++ -std=c++2a -O2 -Wall -Wextra -o puzzle01 puzzle01_arena_lifetime.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Part 1: Demonstrating the dangling pointer problem
// =============================================================================
void part1_the_problem() {
    std::cout << "=== Part 1: The Dangling Pointer Problem ===\n\n";

    int* dangling = nullptr;

    {
        // Arena with its own buffer
        alignas(std::max_align_t) char buffer[1024];
        std::pmr::monotonic_buffer_resource arena(
            buffer, sizeof(buffer), std::pmr::null_memory_resource());

        std::pmr::vector<int> v(&arena);
        v.push_back(42);
        v.push_back(99);

        dangling = v.data();  // Save a pointer to arena-managed memory

        std::printf("  Inside scope: v[0]=%d, v[1]=%d (at %p)\n",
                    v[0], v[1], static_cast<void*>(dangling));

        // v is destroyed here, but arena still owns the memory
        // (monotonic doesn't actually free on deallocate)
    }
    // Arena is destroyed here! buffer[] is gone (stack unwound).
    // 'dangling' now points to freed stack memory.

    // THIS IS UNDEFINED BEHAVIOR:
    // std::printf("  After scope: *dangling=%d\n", *dangling);
    // Might print 42, might print garbage, might crash.
    // On most systems it "works" — the stack hasn't been overwritten yet.
    // That's what makes this bug so insidious.

    std::cout << "  After scope: 'dangling' points to destroyed arena memory.\n";
    std::cout << "  Accessing it is UB. It may 'work' (reading stale stack data)\n";
    std::cout << "  or silently corrupt data. No crash, no warning.\n\n";
}

// =============================================================================
// Part 2: The same bug with heap-backed arena (even more subtle)
// =============================================================================
void part2_heap_arena() {
    std::cout << "=== Part 2: Heap-Backed Arena (More Subtle) ===\n\n";

    std::pmr::vector<int>* leaked = nullptr;

    {
        // Arena backed by new_delete_resource
        std::pmr::monotonic_buffer_resource arena(std::pmr::new_delete_resource());

        // Allocate a vector on the heap, but its CONTENTS are in the arena
        leaked = new std::pmr::vector<int>(&arena);
        leaked->push_back(1);
        leaked->push_back(2);
        leaked->push_back(3);

        std::printf("  Inside scope: leaked->size()=%zu, (*leaked)[0]=%d\n",
                    leaked->size(), (*leaked)[0]);
    }
    // Arena destroyed! The vector object (leaked) is on the heap and alive,
    // but its internal buffer was in the arena → dangling.

    std::cout << "  After arena destroyed:\n";
    std::cout << "  'leaked' still exists on the heap, but its data buffer is gone.\n";
    std::cout << "  leaked->size() might return garbage. Accessing elements is UB.\n";

    // Clean up the vector object (but NOT its contents — those are already gone)
    // This delete will try to deallocate from the destroyed arena → UB
    // delete leaked;  // DON'T DO THIS — the arena is gone
    // We intentionally leak the vector object to avoid UB in the destructor.
    (void)leaked;

    std::cout << "  We can't even safely delete the vector — its destructor\n";
    std::cout << "  would try to deallocate from the dead arena.\n\n";
}

// =============================================================================
// CHALLENGE: SafeArena — asserts that all allocations are returned before death
// =============================================================================
class SafeArena : public std::pmr::memory_resource {
public:
    SafeArena(void* buffer, size_t size)
        : buffer_(static_cast<char*>(buffer))
        , size_(size)
        , offset_(0)
        , live_allocations_(0) {}

    ~SafeArena() {
        // THE SAFETY CHECK: if any allocations are still live, we have a bug.
        if (live_allocations_ != 0) {
            // In production, you might log + abort instead of fprintf
            std::fprintf(stderr,
                "\n  *** SafeArena VIOLATION: destroyed with %d live allocations! ***\n"
                "  *** This means objects are using memory that no longer exists. ***\n\n",
                live_allocations_);
            // In debug builds: assert(false);
            // In production: abort() or throw
        } else {
            std::printf("  [SafeArena] Clean shutdown — 0 live allocations. ✓\n");
        }
    }

    void release() {
        // Only safe to call when no live allocations exist!
        if (live_allocations_ != 0) {
            std::fprintf(stderr,
                "  *** SafeArena::release() called with %d live allocations! ***\n",
                live_allocations_);
        }
        offset_ = 0;
        live_allocations_ = 0;
    }

    int live_allocations() const { return live_allocations_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned + bytes > size_) {
            throw std::bad_alloc();
        }
        void* p = buffer_ + aligned;
        offset_ = aligned + bytes;
        ++live_allocations_;
        return p;
    }

    void do_deallocate(void* /*p*/, size_t /*bytes*/, size_t /*alignment*/) override {
        // Monotonic-style: don't actually free, but track the count
        --live_allocations_;
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    char*  buffer_;
    size_t size_;
    size_t offset_;
    int    live_allocations_;
};

// =============================================================================
// Part 3: SafeArena in action
// =============================================================================
void part3_safe_arena() {
    std::cout << "=== Part 3: SafeArena — Catches Lifetime Bugs ===\n\n";

    // --- Case A: Correct usage — all objects destroyed before arena ---
    std::cout << "  Case A: Correct usage\n";
    {
        alignas(std::max_align_t) char buf[4096];
        SafeArena arena(buf, sizeof(buf));

        {
            std::pmr::vector<int> v(&arena);
            v.push_back(1);
            v.push_back(2);
            v.push_back(3);
            std::printf("    v.size()=%zu, live_allocs=%d\n",
                        v.size(), arena.live_allocations());
        }  // v destroyed → deallocate called → live count decremented

        std::printf("    After v destroyed: live_allocs=%d\n", arena.live_allocations());
    }  // arena destroyed → checks live_allocations == 0 → OK!

    std::cout << "\n  Case B: BUG — object outlives arena\n";
    // We can't easily demonstrate this without UB, but let's show the concept:
    {
        alignas(std::max_align_t) char buf[4096];
        SafeArena arena(buf, sizeof(buf));

        // Simulate: allocate manually, "forget" to deallocate
        void* p1 = arena.allocate(64);
        void* p2 = arena.allocate(128);
        (void)p1;

        // Return one, "leak" the other
        arena.deallocate(p2, 128);

        std::printf("    Allocated 2, returned 1: live_allocs=%d\n",
                    arena.live_allocations());
    }  // arena destroyed with 1 live allocation → prints violation!

    std::cout << "\n";
}

// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Puzzle 01: Arena Lifetime                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    part1_the_problem();
    part2_heap_arena();
    part3_safe_arena();

    std::cout << "=== Key Lessons ===\n";
    std::cout << "  1. Destroying an arena invalidates ALL memory it managed\n";
    std::cout << "  2. There's NO automatic detection — it's UB, not an error\n";
    std::cout << "  3. The bug often 'works' (stale data still readable) → very sneaky\n";
    std::cout << "  4. SafeArena pattern: track live allocations, assert on destruction\n";
    std::cout << "  5. Rule: arena lifetime must OUTLIVE all objects using it\n";
    std::cout << "  6. Declare arena BEFORE containers, destroy containers FIRST\n";

    return 0;
}
