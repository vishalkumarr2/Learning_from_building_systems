// =============================================================================
// Puzzle 02: Allocator Propagation Trap
// =============================================================================
// THE TRAP: pmr::vector<std::string> vs pmr::vector<pmr::string>
//           The first one silently heap-allocates for inner strings.
//           There's no compiler warning, no runtime error. Just wasted
//           determinism in your "zero-allocation" hot path.
//
// CHALLENGE: Write a compile-time check (type trait or concept) that catches
//            non-PMR inner types in PMR containers.
//
// Compile: g++ -std=c++2a -O2 -Wall -Wextra -o puzzle02 puzzle02_propagation_trap.cpp
// =============================================================================

#include <cstdio>
#include <iostream>
#include <map>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <vector>

// =============================================================================
// Tracking resource to prove where allocations go
// =============================================================================
class TrackingResource : public std::pmr::memory_resource {
public:
    explicit TrackingResource(std::pmr::memory_resource* upstream,
                               const char* name)
        : upstream_(upstream), name_(name) {}

    void report() const {
        std::printf("  [%-15s] allocs=%3d bytes=%6zu\n",
                    name_, count_, bytes_);
    }
    void reset() { count_ = 0; bytes_ = 0; }
    int count() const { return count_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        ++count_;
        bytes_ += bytes;
        return upstream_->allocate(bytes, alignment);
    }
    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        upstream_->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    std::pmr::memory_resource* upstream_;
    const char* name_;
    int count_ = 0;
    size_t bytes_ = 0;
};

// =============================================================================
// Part 1: Demonstrate the propagation difference
// =============================================================================
void part1_demonstrate() {
    std::cout << "=== Part 1: The Propagation Difference ===\n\n";

    TrackingResource custom(std::pmr::new_delete_resource(), "custom-arena");
    TrackingResource heap(std::pmr::new_delete_resource(), "default-heap");
    auto* old = std::pmr::set_default_resource(&heap);

    // Long strings to avoid SSO (Short String Optimization)
    const char* long_str = "This_is_a_very_long_string_that_exceeds_SSO_threshold_for_sure!!";

    // --- BAD: pmr::vector<std::string> ---
    std::cout << "  pmr::vector<std::string>  (BAD — inner heap-allocates):\n";
    custom.reset();
    heap.reset();
    {
        std::pmr::vector<std::string> v(&custom);
        v.reserve(5);
        for (int i = 0; i < 5; ++i) {
            v.emplace_back(long_str);
        }
    }
    custom.report();
    heap.report();
    std::cout << "  ^^^ Strings went to default heap! Vector storage to custom.\n\n";

    // --- GOOD: pmr::vector<pmr::string> ---
    std::cout << "  pmr::vector<pmr::string>  (GOOD — all through custom):\n";
    custom.reset();
    heap.reset();
    {
        std::pmr::vector<std::pmr::string> v(&custom);
        v.reserve(5);
        for (int i = 0; i < 5; ++i) {
            v.emplace_back(long_str);
        }
    }
    custom.report();
    heap.report();
    std::cout << "  ^^^ Everything through custom. Zero heap.\n\n";

    std::pmr::set_default_resource(old);
}

// =============================================================================
// Part 2: More subtle cases
// =============================================================================
void part2_subtle_cases() {
    std::cout << "=== Part 2: More Subtle Propagation Traps ===\n\n";

    TrackingResource custom(std::pmr::new_delete_resource(), "custom");
    TrackingResource heap(std::pmr::new_delete_resource(), "heap");
    auto* old = std::pmr::set_default_resource(&heap);

    const char* long_key = "sensor_front_lidar_long_name_to_exceed_sso_____";
    const char* long_val = "value_that_is_also_very_long_to_exceed_sso______";

    // Case A: pmr::map<std::string, std::string> — keys and values heap-allocate!
    std::cout << "  Case A: pmr::map<std::string, std::string>\n";
    custom.reset();
    heap.reset();
    {
        std::pmr::map<std::string, std::string> m(&custom);
        m[long_key] = long_val;
    }
    custom.report();
    heap.report();
    std::cout << "  ^^^ Map nodes in custom, but key/value strings on heap!\n\n";

    // Case B: pmr::map<pmr::string, pmr::string> — everything propagated
    std::cout << "  Case B: pmr::map<pmr::string, pmr::string>\n";
    custom.reset();
    heap.reset();
    {
        std::pmr::map<std::pmr::string, std::pmr::string> m(&custom);
        m.emplace(std::pmr::string(long_key, &custom),
                  std::pmr::string(long_val, &custom));
    }
    custom.report();
    heap.report();
    std::cout << "  ^^^ All memory through custom.\n\n";

    // Case C: pmr::vector<std::vector<int>> — inner vectors heap-allocate
    std::cout << "  Case C: pmr::vector<std::vector<int>>\n";
    custom.reset();
    heap.reset();
    {
        std::pmr::vector<std::vector<int>> v(&custom);
        v.reserve(3);
        for (int i = 0; i < 3; ++i) {
            v.emplace_back(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        }
    }
    custom.report();
    heap.report();
    std::cout << "  ^^^ Outer vector in custom, inner vectors on heap!\n\n";

    std::pmr::set_default_resource(old);
}

// =============================================================================
// CHALLENGE: Compile-time detection of non-PMR inner types
// =============================================================================

// Type trait: does a type use pmr::polymorphic_allocator?
// We check if the type has an allocator_type that is polymorphic_allocator.
template <typename T, typename = void>
struct uses_pmr_allocator : std::false_type {};

template <typename T>
struct uses_pmr_allocator<T, std::void_t<typename T::allocator_type>>
    : std::bool_constant<
          std::is_same_v<
              typename T::allocator_type,
              std::pmr::polymorphic_allocator<typename T::value_type>
          >
      > {};

// Type trait: does a type have no allocator (e.g., int, double)?
// Non-container types are fine — they don't allocate.
template <typename T, typename = void>
struct has_allocator : std::false_type {};

template <typename T>
struct has_allocator<T, std::void_t<typename T::allocator_type>>
    : std::true_type {};

// The safety check: if T has an allocator, it MUST be a PMR allocator.
// If T has no allocator (primitive, POD), it's fine.
template <typename T>
struct is_pmr_safe
    : std::bool_constant<!has_allocator<T>::value || uses_pmr_allocator<T>::value> {};

// Helper for readable static_asserts
template <typename Container>
void assert_pmr_propagation() {
    using ValueType = typename Container::value_type;
    static_assert(
        is_pmr_safe<ValueType>::value,
        "PMR PROPAGATION BUG: Inner container type uses a non-PMR allocator! "
        "Use pmr::string instead of std::string, pmr::vector instead of std::vector, etc."
    );
}

// =============================================================================
// Part 3: Compile-time checks in action
// =============================================================================
void part3_compile_time_checks() {
    std::cout << "=== Part 3: Compile-Time Propagation Checks ===\n\n";

    // These pass:
    assert_pmr_propagation<std::pmr::vector<int>>();           // int has no allocator → OK
    assert_pmr_propagation<std::pmr::vector<double>>();        // double has no allocator → OK
    assert_pmr_propagation<std::pmr::vector<std::pmr::string>>();  // pmr::string → OK

    std::cout << "  pmr::vector<int>            → safe (int has no allocator) ✓\n";
    std::cout << "  pmr::vector<double>         → safe (double has no allocator) ✓\n";
    std::cout << "  pmr::vector<pmr::string>    → safe (pmr allocator) ✓\n";

    // This would FAIL to compile (uncomment to test):
    // assert_pmr_propagation<std::pmr::vector<std::string>>();
    // error: static assertion failed: PMR PROPAGATION BUG: Inner container type
    //        uses a non-PMR allocator!

    std::cout << "  pmr::vector<std::string>    → CAUGHT AT COMPILE TIME ✗\n";
    std::cout << "    (Uncomment the static_assert line to see the error)\n\n";

    // Verify the trait itself
    std::cout << "  Type trait results:\n";
    std::printf("    is_pmr_safe<int>:              %s\n",
                is_pmr_safe<int>::value ? "true (OK — no allocator)" : "false");
    std::printf("    is_pmr_safe<std::string>:      %s\n",
                is_pmr_safe<std::string>::value ? "true" : "false (CAUGHT — non-PMR)");
    std::printf("    is_pmr_safe<std::pmr::string>: %s\n",
                is_pmr_safe<std::pmr::string>::value ? "true (OK — uses PMR)" : "false");
    std::printf("    is_pmr_safe<std::vector<int>>: %s\n",
                is_pmr_safe<std::vector<int>>::value ? "true" : "false (CAUGHT — non-PMR)");
    std::printf("    is_pmr_safe<std::pmr::vector<int>>: %s\n",
                is_pmr_safe<std::pmr::vector<int>>::value ? "true (OK — uses PMR)" : "false");
    std::cout << "\n";
}

// =============================================================================
// Part 4: A safe PMR vector wrapper that enforces propagation
// =============================================================================
template <typename T>
class safe_pmr_vector : public std::pmr::vector<T> {
    static_assert(
        is_pmr_safe<T>::value,
        "safe_pmr_vector<T>: T uses a non-PMR allocator! "
        "Use pmr types (e.g., pmr::string instead of std::string)."
    );

    using Base = std::pmr::vector<T>;
public:
    using Base::Base;  // Inherit constructors
};

void part4_safe_wrapper() {
    std::cout << "=== Part 4: Safe PMR Vector Wrapper ===\n\n";

    TrackingResource alloc(std::pmr::new_delete_resource(), "safe-arena");

    // This compiles fine — pmr::string is safe
    safe_pmr_vector<std::pmr::string> good(&alloc);
    good.emplace_back("safely propagated!");
    std::cout << "  safe_pmr_vector<pmr::string>: \"" << good[0] << "\" ✓\n";

    // This compiles fine — int has no allocator
    safe_pmr_vector<int> also_good(&alloc);
    also_good.push_back(42);
    std::cout << "  safe_pmr_vector<int>: " << also_good[0] << " ✓\n";

    // This would NOT compile (uncomment to test):
    // safe_pmr_vector<std::string> bad(&alloc);
    // error: static assertion failed: T uses a non-PMR allocator!

    std::cout << "  safe_pmr_vector<std::string>: COMPILE ERROR (caught!) ✗\n\n";
}

// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Puzzle 02: Allocator Propagation Trap           ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    part1_demonstrate();
    part2_subtle_cases();
    part3_compile_time_checks();
    part4_safe_wrapper();

    std::cout << "=== Key Lessons ===\n";
    std::cout << "  1. pmr::vector<std::string> is a SILENT BUG — no warning\n";
    std::cout << "  2. EVERY nested container must be a pmr:: type\n";
    std::cout << "  3. Use is_pmr_safe<T> trait to catch this at compile time\n";
    std::cout << "  4. The safe_pmr_vector wrapper enforces propagation\n";
    std::cout << "  5. Rule: if it's in a PMR container and it allocates, it must be PMR\n";

    return 0;
}
