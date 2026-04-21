// =============================================================================
// Exercise 03: Custom Memory Resources
// =============================================================================
// Topics:
//   - Writing custom std::pmr::memory_resource subclasses
//   - TrackingResource — wraps another resource, logs allocations/deallocations
//   - StackResource — fixed stack buffer, no fallback
//   - BumpAllocator — like monotonic but with explicit reset()
//   - Nested PMR container propagation: pmr::vector<pmr::string>
//
// Compile: g++ -std=c++2a -O2 -Wall -Wextra -o ex03 ex03_custom_memory_resource.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory_resource>
#include <string>
#include <vector>

// =============================================================================
// Custom Resource 1: TrackingResource
// Wraps an upstream resource and logs every allocation/deallocation.
// Invaluable for debugging where memory is going.
// =============================================================================
class TrackingResource : public std::pmr::memory_resource {
public:
    explicit TrackingResource(std::pmr::memory_resource* upstream,
                               const char* name = "tracker")
        : upstream_(upstream), name_(name) {}

    // --- Reporting ---
    void report(const char* label = "") const {
        std::printf("  [%s%s%s] allocs=%d deallocs=%d alive=%d "
                    "bytes_alloc=%zu bytes_dealloc=%zu current=%zu peak=%zu\n",
                    name_, (*label ? " " : ""), label,
                    alloc_count_, dealloc_count_, alloc_count_ - dealloc_count_,
                    total_allocated_, total_deallocated_,
                    total_allocated_ - total_deallocated_, peak_);
    }

    size_t current_bytes() const { return total_allocated_ - total_deallocated_; }
    size_t peak_bytes() const { return peak_; }
    int    alloc_count() const { return alloc_count_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        if (verbose_) {
            std::printf("    [%s] allocate(%zu, align=%zu)\n",
                        name_, bytes, alignment);
        }
        void* p = upstream_->allocate(bytes, alignment);
        ++alloc_count_;
        total_allocated_ += bytes;
        size_t current = total_allocated_ - total_deallocated_;
        if (current > peak_) peak_ = current;
        return p;
    }

    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        if (verbose_) {
            std::printf("    [%s] deallocate(%p, %zu, align=%zu)\n",
                        name_, p, bytes, alignment);
        }
        upstream_->deallocate(p, bytes, alignment);
        ++dealloc_count_;
        total_deallocated_ += bytes;
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        // Two TrackingResources are equal only if they're the same instance.
        // This is the typical implementation for stateful resources.
        return this == &other;
    }

private:
    std::pmr::memory_resource* upstream_;
    const char* name_;
    int alloc_count_ = 0;
    int dealloc_count_ = 0;
    size_t total_allocated_ = 0;
    size_t total_deallocated_ = 0;
    size_t peak_ = 0;

public:
    bool verbose_ = false;
};

// =============================================================================
// Custom Resource 2: StackResource
// Allocates from a fixed stack buffer. No fallback. Throws when full.
// This is the simplest possible allocator — good for understanding alignment.
// =============================================================================
class StackResource : public std::pmr::memory_resource {
public:
    StackResource(void* buffer, size_t size)
        : buffer_(static_cast<char*>(buffer))
        , size_(size)
        , offset_(0) {}

    size_t used() const { return offset_; }
    size_t remaining() const { return size_ - offset_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        // Align the current offset up to the requested alignment.
        // This is the key part most people get wrong!
        size_t aligned_offset = align_up(offset_, alignment);

        if (aligned_offset + bytes > size_) {
            throw std::bad_alloc();  // No fallback — hard failure
        }

        void* result = buffer_ + aligned_offset;
        offset_ = aligned_offset + bytes;
        return result;
    }

    void do_deallocate(void* /*p*/, size_t /*bytes*/, size_t /*alignment*/) override {
        // Stack allocator: deallocation is a no-op.
        // Memory is only reclaimed when the StackResource itself is destroyed.
        // (Like monotonic, but simpler — no upstream, no release())
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    static size_t align_up(size_t offset, size_t alignment) {
        // Round up to next multiple of alignment.
        // alignment must be a power of 2.
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    char*  buffer_;
    size_t size_;
    size_t offset_;
};

// =============================================================================
// Custom Resource 3: BumpAllocator
// Like monotonic_buffer_resource but with an explicit reset() for frame reuse.
// This is the classic "arena allocator" pattern used in game engines.
// =============================================================================
class BumpAllocator : public std::pmr::memory_resource {
public:
    BumpAllocator(void* buffer, size_t size)
        : buffer_(static_cast<char*>(buffer))
        , size_(size)
        , offset_(0)
        , alloc_count_(0)
        , high_water_(0) {}

    // Reset the allocator — all previous allocations are invalidated!
    // Caller must ensure no live objects reference the old memory.
    void reset() {
        if (offset_ > high_water_) high_water_ = offset_;
        offset_ = 0;
        alloc_count_ = 0;
    }

    size_t used() const { return offset_; }
    size_t remaining() const { return size_ - offset_; }
    size_t high_water_mark() const { return high_water_ > offset_ ? high_water_ : offset_; }
    int    alloc_count() const { return alloc_count_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        size_t aligned = align_up(offset_, alignment);
        if (aligned + bytes > size_) {
            throw std::bad_alloc();
        }
        void* result = buffer_ + aligned;
        offset_ = aligned + bytes;
        ++alloc_count_;
        return result;
    }

    void do_deallocate(void* /*p*/, size_t /*bytes*/, size_t /*alignment*/) override {
        // No-op. Memory reclaimed only via reset().
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    static size_t align_up(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    char*  buffer_;
    size_t size_;
    size_t offset_;
    int    alloc_count_;
    size_t high_water_;
};

// =============================================================================
// Part 1: TrackingResource — observe allocation patterns
// =============================================================================
void part1_tracking() {
    std::cout << "=== Part 1: TrackingResource ===\n\n";

    TrackingResource tracker(std::pmr::new_delete_resource(), "heap-tracker");

    {
        std::pmr::vector<int> v(&tracker);
        v.reserve(100);
        for (int i = 0; i < 100; ++i) v.push_back(i);
        tracker.report("after 100 ints");
    }
    tracker.report("after vector destroyed");
    std::cout << "\n";

    // Verbose mode — see every allocation
    tracker.verbose_ = true;
    std::cout << "  Verbose mode — watch pmr::string allocations:\n";
    {
        std::pmr::string s("Hello, PMR world! This is long enough to avoid SSO.", &tracker);
    }
    tracker.verbose_ = false;
    std::cout << "\n";
}

// =============================================================================
// Part 2: StackResource — fixed buffer, no fallback
// =============================================================================
void part2_stack_resource() {
    std::cout << "=== Part 2: StackResource ===\n\n";

    alignas(std::max_align_t) char buffer[512];
    StackResource stack(buffer, sizeof(buffer));

    // Use with pmr::vector
    std::pmr::vector<int> v(&stack);
    for (int i = 0; i < 20; ++i) v.push_back(i);
    std::printf("  Vector of 20 ints: used %zu / %zu bytes\n",
                stack.used(), stack.used() + stack.remaining());

    // Use with pmr::string
    std::pmr::string s("allocated from stack buffer!", &stack);
    std::printf("  After string: used %zu / 512 bytes\n", stack.used());

    // Demonstrate overflow
    try {
        std::pmr::vector<double> big(&stack);
        for (int i = 0; i < 1000; ++i) big.push_back(static_cast<double>(i));
        std::cout << "  ERROR: should have thrown\n";
    } catch (const std::bad_alloc&) {
        std::printf("  Overflow caught! Used %zu / 512 bytes when it failed.\n\n",
                    stack.used());
    }
}

// =============================================================================
// Part 3: BumpAllocator — reset for frame reuse
// =============================================================================
void part3_bump_allocator() {
    std::cout << "=== Part 3: BumpAllocator with reset() ===\n\n";

    constexpr size_t BUF_SIZE = 8192;
    alignas(std::max_align_t) char buffer[BUF_SIZE];
    BumpAllocator bump(buffer, BUF_SIZE);

    // Simulate 5 "frames" — each frame allocates, then resets
    for (int frame = 0; frame < 5; ++frame) {
        // --- Frame work: create temp containers ---
        {
            std::pmr::vector<int> sensor_data(&bump);
            sensor_data.reserve(50);
            for (int i = 0; i < 50; ++i) {
                sensor_data.push_back(i * 10 + frame);
            }

            std::pmr::vector<double> output(&bump);
            output.reserve(50);
            for (int i = 0; i < 50; ++i) {
                output.push_back(sensor_data[static_cast<size_t>(i)] * 0.1);
            }
        }
        // Containers destroyed here, but bump doesn't free (no-op deallocate)

        std::printf("  Frame %d: used %4zu bytes, %d allocs\n",
                    frame, bump.used(), bump.alloc_count());

        // Reset for next frame — O(1), just moves pointer back to 0
        bump.reset();
    }

    std::printf("  High-water mark: %zu / %zu bytes\n\n",
                bump.high_water_mark(), BUF_SIZE);
}

// =============================================================================
// Part 4: Nested PMR container propagation
// =============================================================================
void part4_nested_propagation() {
    std::cout << "=== Part 4: Nested PMR Container Propagation ===\n\n";

    TrackingResource tracker(std::pmr::new_delete_resource(), "nested");

    // pmr::vector<pmr::string> — BOTH use the tracker's memory
    std::cout << "  Creating pmr::vector<pmr::string> with shared resource:\n";
    tracker.verbose_ = true;
    {
        std::pmr::vector<std::pmr::string> names(&tracker);
        names.reserve(3);  // pre-alloc vector storage

        // Each emplace_back creates a pmr::string that inherits the resource
        names.emplace_back("Alice");    // Short string → may use SSO (no alloc)
        names.emplace_back("Bob");
        names.emplace_back("This is a long string that definitely exceeds SSO");
    }
    tracker.verbose_ = false;
    tracker.report("after destruction");

    std::cout << "\n  Key insight: The inner pmr::string used the SAME resource\n";
    std::cout << "  as the outer vector. This is allocator propagation in action.\n";
    std::cout << "  (Short strings may use SSO and not allocate at all.)\n\n";

    // Verify propagation explicitly
    {
        std::pmr::monotonic_buffer_resource mono(std::pmr::new_delete_resource());
        std::pmr::vector<std::pmr::string> v(&mono);
        v.emplace_back("test");

        // The string's allocator should reference the same resource as the vector
        auto* vec_resource = v.get_allocator().resource();
        auto* str_resource = v[0].get_allocator().resource();

        std::printf("  vector resource: %p\n", static_cast<void*>(vec_resource));
        std::printf("  string resource: %p\n", static_cast<void*>(str_resource));
        std::printf("  Same? %s ✓\n\n",
                    (vec_resource == str_resource) ? "YES" : "NO (BUG!)");
    }
}

// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Exercise 03: Custom Memory Resources            ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    part1_tracking();
    part2_stack_resource();
    part3_bump_allocator();
    part4_nested_propagation();

    std::cout << "=== Summary ===\n";
    std::cout << "  - Override do_allocate, do_deallocate, do_is_equal\n";
    std::cout << "  - TrackingResource: debugging tool, wraps any upstream\n";
    std::cout << "  - StackResource: simplest allocator, fixed buffer, throws on full\n";
    std::cout << "  - BumpAllocator: arena with reset(), perfect for frame-based reuse\n";
    std::cout << "  - pmr::vector<pmr::string> propagates the resource to inner strings\n";

    return 0;
}
