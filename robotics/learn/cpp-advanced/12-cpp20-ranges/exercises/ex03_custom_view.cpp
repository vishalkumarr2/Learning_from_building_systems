// Module 12, Exercise 3: Writing a Custom View
// Compiler: GCC 12+ or Clang 14+ with -std=c++20
//
// Implements a `stride_view` that takes every Nth element from a range.
// C++23 adds std::views::stride — this is how you'd write it from scratch.
//
// Demonstrates:
//   - view_interface CRTP base
//   - Custom iterator implementation
//   - Composability with pipe syntax
//   - Working with both containers and other views

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ranges>
#include <vector>

namespace ranges = std::ranges;
namespace views = std::views;

// ============================================================
// stride_view: take every Nth element
// ============================================================

template <ranges::input_range V>
    requires ranges::view<V>
class stride_view : public ranges::view_interface<stride_view<V>> {
private:
    V base_;
    std::size_t stride_;

    // --- Iterator ---
    class iterator {
    public:
        // Iterator traits (required for ranges concepts)
        using iterator_category = std::input_iterator_tag;
        using value_type = ranges::range_value_t<V>;
        using difference_type = ranges::range_difference_t<V>;
        using pointer = void;
        using reference = ranges::range_reference_t<V>;

    private:
        ranges::iterator_t<V> current_;
        ranges::sentinel_t<V> end_;
        std::size_t stride_;

    public:
        iterator() = default;

        iterator(ranges::iterator_t<V> current, ranges::sentinel_t<V> end, std::size_t stride)
            : current_(std::move(current)), end_(std::move(end)), stride_(stride) {}

        reference operator*() const { return *current_; }

        iterator& operator++() {
            // Advance by stride, but don't go past end
            for (std::size_t i = 0; i < stride_ && current_ != end_; ++i) {
                ++current_;
            }
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const iterator& other) const {
            return current_ == other.current_;
        }

        // Sentinel comparison
        friend bool operator==(const iterator& it, ranges::sentinel_t<V> sent) {
            return it.current_ == sent;
        }
    };

    // --- Sentinel (wraps the underlying sentinel) ---
    class sentinel {
        ranges::sentinel_t<V> end_;
    public:
        sentinel() = default;
        explicit sentinel(ranges::sentinel_t<V> end) : end_(std::move(end)) {}

        friend bool operator==(const iterator& it, const sentinel& s) {
            // Use the iterator's comparison with the underlying sentinel
            return it == s.end_;
        }
    };

public:
    stride_view() = default;

    stride_view(V base, std::size_t stride)
        : base_(std::move(base)), stride_(stride) {}

    auto begin() {
        return iterator(ranges::begin(base_), ranges::end(base_), stride_);
    }

    auto end() {
        return sentinel(ranges::end(base_));
    }
};

// Deduction guide: stride_view(R, size_t) → stride_view<views::all_t<R>>
template <class R>
stride_view(R&&, std::size_t) -> stride_view<views::all_t<R>>;

// ============================================================
// Range adaptor closure for pipe syntax: rng | stride(N)
// ============================================================

struct stride_adaptor {
    std::size_t n;

    // Called as: some_range | stride(3)
    template <ranges::viewable_range R>
    friend auto operator|(R&& r, stride_adaptor self) {
        return stride_view(std::forward<R>(r), self.n);
    }
};

// Factory function
inline auto stride(std::size_t n) {
    return stride_adaptor{n};
}

// ============================================================
// Tests
// ============================================================

void test_basic() {
    std::cout << "=== Test 1: Basic stride over vector ===\n";

    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    std::cout << "Every 3rd element: ";
    for (int x : stride_view(data, 3)) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    // Expected: 0 3 6 9

    std::cout << "Every 2nd element: ";
    for (int x : stride_view(data, 2)) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    // Expected: 0 2 4 6 8 10

    std::cout << "Every 1st (identity): ";
    for (int x : stride_view(data, 1)) {
        std::cout << x << " ";
    }
    std::cout << "\n\n";
    // Expected: 0 1 2 3 4 5 6 7 8 9 10 11
}

void test_pipe_syntax() {
    std::cout << "=== Test 2: Pipe syntax ===\n";

    std::vector<int> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // Use our pipe adaptor
    std::cout << "data | stride(3): ";
    for (int x : data | stride(3)) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    // Expected: 10 40 70 100

    std::cout << "\n";
}

void test_composition() {
    std::cout << "=== Test 3: Composing stride with standard views ===\n";

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Chain: filter evens → stride(2) → transform to string
    // Evens: 2, 4, 6, 8, 10, 12
    // Stride(2): 2, 6, 10
    // Transform: square
    std::cout << "filter(even) | stride(2) | transform(square): ";
    auto pipeline = data
        | views::filter([](int x) { return x % 2 == 0; })
        | stride(2)
        | views::transform([](int x) { return x * x; });

    for (int x : pipeline) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    // Expected: 4 36 100

    std::cout << "\n";
}

void test_sensor_downsampling() {
    std::cout << "=== Test 4: Real-World — Sensor Downsampling ===\n";

    // Simulate 100Hz sensor data — we want 10Hz (every 10th sample)
    std::vector<double> raw_sensor(100);
    for (std::size_t i = 0; i < raw_sensor.size(); ++i) {
        raw_sensor[i] = static_cast<double>(i) * 0.37;  // some sensor value
    }

    auto downsampled = raw_sensor | stride(10);

    std::cout << "100Hz → 10Hz (every 10th): ";
    int count = 0;
    for (double val : downsampled) {
        std::cout << std::fixed;
        std::cout.precision(1);
        std::cout << val << " ";
        ++count;
    }
    std::cout << "\n(" << count << " samples from " << raw_sensor.size() << ")\n\n";
}

void test_with_iota() {
    std::cout << "=== Test 5: Stride over iota (infinite range) ===\n";

    // Every 7th number from 0..., take first 8
    auto every_7th = views::iota(0) | stride(7) | views::take(8);

    std::cout << "Every 7th natural number (first 8): ";
    for (int x : every_7th) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    // Expected: 0 7 14 21 28 35 42 49

    std::cout << "\n";
}

int main() {
    test_basic();
    test_pipe_syntax();
    test_composition();
    test_sensor_downsampling();
    test_with_iota();
    return 0;
}
