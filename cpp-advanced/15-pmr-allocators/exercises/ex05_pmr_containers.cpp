// =============================================================================
// Exercise 05: PMR Container Compatibility
// =============================================================================
// Topics:
//   - pmr::vector<pmr::string> — allocator propagation
//   - Nested container propagation: pmr::vector<pmr::vector<int>>
//   - Verifying propagation with polymorphic_allocator::resource()
//   - The subtle bug: pmr::vector<std::string> (inner doesn't propagate!)
//   - Move semantics with different memory resources
//
// Compile: g++ -std=c++2a -O2 -Wall -Wextra -o ex05 ex05_pmr_containers.cpp
// =============================================================================

#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory_resource>
#include <string>
#include <vector>

// =============================================================================
// TrackingResource — reused from ex03 for observing allocations
// =============================================================================
class TrackingResource : public std::pmr::memory_resource {
public:
    explicit TrackingResource(std::pmr::memory_resource* upstream,
                               const char* name = "tracker")
        : upstream_(upstream), name_(name) {}

    void report(const char* label = "") const {
        std::printf("  [%s%s%s] allocs=%d deallocs=%d bytes=%zu\n",
                    name_, (*label ? " " : ""), label,
                    alloc_count_, dealloc_count_, total_allocated_);
    }

    void reset_counters() {
        alloc_count_ = dealloc_count_ = 0;
        total_allocated_ = total_deallocated_ = 0;
    }

    int alloc_count() const { return alloc_count_; }
    size_t total_allocated() const { return total_allocated_; }

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        void* p = upstream_->allocate(bytes, alignment);
        ++alloc_count_;
        total_allocated_ += bytes;
        return p;
    }

    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        upstream_->deallocate(p, bytes, alignment);
        ++dealloc_count_;
        total_deallocated_ += bytes;
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    std::pmr::memory_resource* upstream_;
    const char* name_;
    int alloc_count_ = 0;
    int dealloc_count_ = 0;
    size_t total_allocated_ = 0;
    size_t total_deallocated_ = 0;
};

// =============================================================================
// Part 1: pmr::vector<pmr::string> — full propagation
// =============================================================================
void part1_string_propagation() {
    std::cout << "=== Part 1: pmr::vector<pmr::string> — Full Propagation ===\n\n";

    TrackingResource custom_alloc(std::pmr::new_delete_resource(), "custom");

    {
        std::pmr::vector<std::pmr::string> names(&custom_alloc);
        names.reserve(4);

        // Long strings (exceed SSO threshold ~15-22 chars) to force allocation
        names.emplace_back("Alexander_Hamilton_was_here_________________");
        names.emplace_back("Benjamin_Franklin_was_also_here______________");
        names.emplace_back("Catherine_the_Great_approves_this_message____");

        // Verify: every string's allocator points to our custom resource
        std::cout << "  Checking allocator propagation:\n";
        for (size_t i = 0; i < names.size(); ++i) {
            auto* res = names[i].get_allocator().resource();
            std::printf("    names[%zu].resource() = %p (custom=%p) → %s\n",
                        i, static_cast<void*>(res),
                        static_cast<void*>(&custom_alloc),
                        (res == &custom_alloc) ? "PROPAGATED ✓" : "WRONG ✗");
        }
    }
    custom_alloc.report("after destruction");
    std::cout << "  All memory (vector + strings) went through custom allocator.\n\n";
}

// =============================================================================
// Part 2: THE SUBTLE BUG — pmr::vector<std::string>
// =============================================================================
void part2_the_bug() {
    std::cout << "=== Part 2: THE BUG — pmr::vector<std::string> ===\n\n";

    TrackingResource custom_alloc(std::pmr::new_delete_resource(), "custom");
    TrackingResource heap_tracker(std::pmr::new_delete_resource(), "default-heap");

    // Temporarily set global default to our heap_tracker so we can see
    // what goes through the default allocator
    auto* old_default = std::pmr::set_default_resource(&heap_tracker);

    std::cout << "  --- WRONG: pmr::vector<std::string> ---\n";
    {
        // Vector uses custom allocator, but std::string uses DEFAULT allocator!
        std::pmr::vector<std::string> wrong(&custom_alloc);
        wrong.reserve(3);
        wrong.emplace_back("This_long_string_goes_to_HEAP_not_custom_alloc!!");
        wrong.emplace_back("Another_long_string_also_heap_allocated_silently");
        wrong.emplace_back("Third_string_same_problem_no_warning_no_error___");
    }
    custom_alloc.report("vector storage");
    heap_tracker.report("string heap!!");
    std::cout << "  ^^^ The strings silently heap-allocated!\n";
    std::cout << "  In RT, this breaks your zero-allocation guarantee.\n\n";

    custom_alloc.reset_counters();
    heap_tracker.reset_counters();

    std::cout << "  --- RIGHT: pmr::vector<pmr::string> ---\n";
    {
        std::pmr::vector<std::pmr::string> right(&custom_alloc);
        right.reserve(3);
        right.emplace_back("This_long_string_goes_to_CUSTOM_allocator!!!!!");
        right.emplace_back("Another_long_string_also_custom_allocated______");
        right.emplace_back("Third_string_same_resource_everything_correct__");
    }
    custom_alloc.report("all memory");
    heap_tracker.report("heap (should be 0)");
    std::cout << "  ^^^ ALL memory through custom allocator. Zero heap.\n\n";

    // Restore default
    std::pmr::set_default_resource(old_default);
}

// =============================================================================
// Part 3: Nested container propagation
// =============================================================================
void part3_nested_containers() {
    std::cout << "=== Part 3: Nested Container Propagation ===\n\n";

    TrackingResource alloc(std::pmr::new_delete_resource(), "arena");

    // pmr::vector<pmr::vector<int>> — both levels use the same resource
    {
        std::pmr::vector<std::pmr::vector<int>> matrix(&alloc);
        matrix.reserve(3);

        for (int row = 0; row < 3; ++row) {
            matrix.emplace_back();  // Inner vector gets the resource from outer
            matrix.back().reserve(10);
            for (int col = 0; col < 10; ++col) {
                matrix.back().push_back(row * 10 + col);
            }
        }

        // Verify all inner vectors use the same resource
        std::cout << "  pmr::vector<pmr::vector<int>> propagation check:\n";
        auto* outer_res = matrix.get_allocator().resource();
        for (size_t i = 0; i < matrix.size(); ++i) {
            auto* inner_res = matrix[i].get_allocator().resource();
            std::printf("    matrix[%zu].resource() → %s\n", i,
                        (inner_res == outer_res) ? "same as outer ✓" : "DIFFERENT ✗");
        }
    }
    alloc.report("all memory");
    std::cout << "\n";

    // pmr::map with pmr::string keys and pmr::vector values
    alloc.reset_counters();
    {
        std::pmr::map<std::pmr::string, std::pmr::vector<int>> index(&alloc);

        // Build an index of sensor readings by name
        index["front_lidar_long_name_to_avoid_sso______"].push_back(100);
        index["front_lidar_long_name_to_avoid_sso______"].push_back(200);
        index["rear_lidar_long_name_to_avoid_sso_______"].push_back(300);
        index["imu_sensor_long_name_to_avoid_sso_______"].push_back(400);

        // All keys (strings) and values (vectors) should use the same resource
        auto* map_res = index.get_allocator().resource();
        std::cout << "  pmr::map<pmr::string, pmr::vector<int>> check:\n";
        for (const auto& [key, val] : index) {
            auto* key_res = key.get_allocator().resource();
            auto* val_res = val.get_allocator().resource();
            std::printf("    key '%.*s...': key=%s val=%s\n",
                        12, key.c_str(),
                        (key_res == map_res) ? "✓" : "✗",
                        (val_res == map_res) ? "✓" : "✗");
        }
    }
    alloc.report("map+strings+vectors");
    std::cout << "\n";
}

// =============================================================================
// Part 4: Move semantics with different resources
// =============================================================================
void part4_move_semantics() {
    std::cout << "=== Part 4: Move Semantics with Different Resources ===\n\n";

    TrackingResource alloc_a(std::pmr::new_delete_resource(), "resource-A");
    TrackingResource alloc_b(std::pmr::new_delete_resource(), "resource-B");

    // --- Same resource: move is O(1) (pointer transfer) ---
    std::cout << "  Case 1: Move between containers with SAME resource\n";
    {
        std::pmr::vector<int> v1({1, 2, 3, 4, 5}, &alloc_a);
        alloc_a.reset_counters();

        std::pmr::vector<int> v2(std::move(v1), &alloc_a);
        // Same resource → v1's buffer is transferred to v2 (no copy)

        alloc_a.report("after same-resource move");
        std::printf("    v1.size()=%zu v2.size()=%zu\n", v1.size(), v2.size());
        std::cout << "    → O(1) move, no allocations. ✓\n\n";
    }

    alloc_a.reset_counters();
    alloc_b.reset_counters();

    // --- Different resources: move becomes a COPY! ---
    std::cout << "  Case 2: Move between containers with DIFFERENT resources\n";
    {
        std::pmr::vector<int> v1({1, 2, 3, 4, 5}, &alloc_a);
        alloc_a.reset_counters();

        // Different resource → can't transfer ownership → must COPY elements
        std::pmr::vector<int> v2(std::move(v1), &alloc_b);

        alloc_a.report("source (should be zero new allocs)");
        alloc_b.report("dest   (had to allocate!)");
        std::printf("    v1.size()=%zu v2.size()=%zu\n", v1.size(), v2.size());
        std::cout << "    → Move with different resource = COPY! This can surprise you.\n";
        std::cout << "    → In RT: avoid moving between resources, or ensure same resource.\n\n";
    }

    // --- Move assignment: same-resource vs different-resource ---
    alloc_a.reset_counters();
    alloc_b.reset_counters();

    std::cout << "  Case 3: Move assignment with different resources\n";
    {
        std::pmr::vector<int> v1({10, 20, 30}, &alloc_a);
        std::pmr::vector<int> v2(&alloc_b);

        alloc_a.reset_counters();
        alloc_b.reset_counters();

        v2 = std::move(v1);
        // PMR allocator does NOT propagate on move assignment (POCMA = false)
        // So v2 still uses alloc_b → must copy elements from alloc_a to alloc_b

        alloc_a.report("source");
        alloc_b.report("dest");
        std::printf("    v2 resource: %s\n",
                    (v2.get_allocator().resource() == &alloc_b) ? "still B ✓" : "changed!");
        std::cout << "    → PMR allocator does NOT propagate on move-assign.\n";
        std::cout << "    → v2 keeps its original resource, copies data into it.\n\n";
    }
}

// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Exercise 05: PMR Container Compatibility        ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    part1_string_propagation();
    part2_the_bug();
    part3_nested_containers();
    part4_move_semantics();

    std::cout << "=== Summary ===\n";
    std::cout << "  - ALWAYS use pmr::string, pmr::vector, etc. for inner containers\n";
    std::cout << "  - pmr::vector<std::string> is a SILENT BUG — inner heap-allocates\n";
    std::cout << "  - Verify propagation with .get_allocator().resource()\n";
    std::cout << "  - Move with same resource = O(1). Different resource = COPY.\n";
    std::cout << "  - PMR allocator does not propagate on move/copy assignment\n";

    return 0;
}
