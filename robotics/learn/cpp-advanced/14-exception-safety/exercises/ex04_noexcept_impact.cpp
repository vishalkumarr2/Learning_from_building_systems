// =============================================================================
// Exercise 04: The Real Impact of noexcept
// =============================================================================
// Demonstrates that noexcept on move constructors is NOT just documentation —
// it changes runtime behavior. std::vector copies instead of moves when the
// move constructor isn't noexcept, causing massive performance differences.
//
// Compile: g++ -std=c++2a -Wall -Wextra -O2 -o ex04 ex04_noexcept_impact.cpp
// =============================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <type_traits>
#include <utility>
#include <cstring>

// =============================================================================
// Two widget classes: identical except for noexcept on move ctor
// =============================================================================

// Widget WITH noexcept move — vector will MOVE during reallocation
class WidgetNoexcept {
    int* data_;
    size_t size_;

    static int move_count;
    static int copy_count;

public:
    explicit WidgetNoexcept(size_t n = 100)
        : data_(new int[n])
        , size_(n)
    {
        std::memset(data_, 0, n * sizeof(int));
    }

    ~WidgetNoexcept() { delete[] data_; }

    // Copy constructor — expensive!
    WidgetNoexcept(const WidgetNoexcept& other)
        : data_(new int[other.size_])
        , size_(other.size_)
    {
        std::memcpy(data_, other.data_, size_ * sizeof(int));
        ++copy_count;
    }

    // Move constructor — NOEXCEPT! Vector will use this.
    WidgetNoexcept(WidgetNoexcept&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        ++move_count;
    }

    WidgetNoexcept& operator=(WidgetNoexcept other) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        return *this;
    }

    static void reset_counts() { move_count = 0; copy_count = 0; }
    static int moves() { return move_count; }
    static int copies() { return copy_count; }
};

int WidgetNoexcept::move_count = 0;
int WidgetNoexcept::copy_count = 0;

// Widget WITHOUT noexcept move — vector will COPY during reallocation!
class WidgetMayThrow {
    int* data_;
    size_t size_;

    static int move_count;
    static int copy_count;

public:
    explicit WidgetMayThrow(size_t n = 100)
        : data_(new int[n])
        , size_(n)
    {
        std::memset(data_, 0, n * sizeof(int));
    }

    ~WidgetMayThrow() { delete[] data_; }

    // Copy constructor — expensive!
    WidgetMayThrow(const WidgetMayThrow& other)
        : data_(new int[other.size_])
        , size_(other.size_)
    {
        std::memcpy(data_, other.data_, size_ * sizeof(int));
        ++copy_count;
    }

    // Move constructor — NOT noexcept! Vector will AVOID this during reallocation.
    // Why? If a move throws halfway through reallocation, the old buffer is
    // already partially destroyed. There's no way to recover.
    // So vector plays it safe and copies instead.
    WidgetMayThrow(WidgetMayThrow&& other)  // <-- no noexcept!
        : data_(other.data_)
        , size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        ++move_count;
    }

    WidgetMayThrow& operator=(WidgetMayThrow other) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        return *this;
    }

    static void reset_counts() { move_count = 0; copy_count = 0; }
    static int moves() { return move_count; }
    static int copies() { return copy_count; }
};

int WidgetMayThrow::move_count = 0;
int WidgetMayThrow::copy_count = 0;

// =============================================================================
// Compile-time checks
// =============================================================================
void show_type_traits() {
    std::cout << "\n=== Compile-Time Type Traits ===" << std::endl;

    std::cout << "  WidgetNoexcept:" << std::endl;
    std::cout << "    is_nothrow_move_constructible: "
              << std::is_nothrow_move_constructible_v<WidgetNoexcept> << std::endl;
    std::cout << "    noexcept(move ctor): "
              << noexcept(WidgetNoexcept(std::declval<WidgetNoexcept&&>()))
              << std::endl;

    std::cout << "  WidgetMayThrow:" << std::endl;
    std::cout << "    is_nothrow_move_constructible: "
              << std::is_nothrow_move_constructible_v<WidgetMayThrow> << std::endl;
    std::cout << "    noexcept(move ctor): "
              << noexcept(WidgetMayThrow(std::declval<WidgetMayThrow&&>()))
              << std::endl;
}

// =============================================================================
// Benchmark: push_back with noexcept move vs without
// =============================================================================
void benchmark() {
    std::cout << "\n=== Benchmark: vector::push_back ===" << std::endl;
    constexpr int N = 100000;

    // --- With noexcept move ---
    WidgetNoexcept::reset_counts();
    auto start1 = std::chrono::steady_clock::now();
    {
        std::vector<WidgetNoexcept> vec;
        for (int i = 0; i < N; ++i)
            vec.push_back(WidgetNoexcept(100));
    }
    auto end1 = std::chrono::steady_clock::now();
    auto ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();

    std::cout << "\n  WidgetNoexcept (move is noexcept):" << std::endl;
    std::cout << "    Time: " << ms1 << " ms" << std::endl;
    std::cout << "    Moves: " << WidgetNoexcept::moves() << std::endl;
    std::cout << "    Copies: " << WidgetNoexcept::copies() << std::endl;

    // --- Without noexcept move ---
    WidgetMayThrow::reset_counts();
    auto start2 = std::chrono::steady_clock::now();
    {
        std::vector<WidgetMayThrow> vec;
        for (int i = 0; i < N; ++i)
            vec.push_back(WidgetMayThrow(100));
    }
    auto end2 = std::chrono::steady_clock::now();
    auto ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();

    std::cout << "\n  WidgetMayThrow (move is NOT noexcept):" << std::endl;
    std::cout << "    Time: " << ms2 << " ms" << std::endl;
    std::cout << "    Moves: " << WidgetMayThrow::moves() << std::endl;
    std::cout << "    Copies: " << WidgetMayThrow::copies()
              << "  <-- COPIES during reallocation!" << std::endl;

    if (ms2 > 0 && ms1 > 0) {
        std::cout << "\n  Slowdown factor: ~" << ms2 / std::max(ms1, 1L) << "x" << std::endl;
    }
    std::cout << "\n  Key: Without noexcept on move, vector COPIES all elements"
              << std::endl;
    std::cout << "  during reallocation to maintain the strong guarantee." << std::endl;
}

// =============================================================================
// Demonstrate std::move_if_noexcept
// =============================================================================
void demo_move_if_noexcept() {
    std::cout << "\n=== std::move_if_noexcept ===" << std::endl;

    // move_if_noexcept returns T&& if T is nothrow-move-constructible,
    // otherwise returns const T& (forcing a copy).

    WidgetNoexcept w1(10);
    WidgetMayThrow w2(10);

    // For noexcept-movable type: returns rvalue reference (will move)
    auto&& ref1 = std::move_if_noexcept(w1);
    std::cout << "  WidgetNoexcept: move_if_noexcept returns "
              << (std::is_rvalue_reference_v<decltype(ref1)> ? "rvalue ref (MOVE)" : "lvalue ref (COPY)")
              << std::endl;
    // Avoid unused-variable warning — just checking the type
    (void)ref1;

    // For may-throw type: returns const lvalue reference (will copy)
    auto&& ref2 = std::move_if_noexcept(w2);
    std::cout << "  WidgetMayThrow: move_if_noexcept returns "
              << (std::is_rvalue_reference_v<decltype(ref2)> ? "rvalue ref (MOVE)" : "lvalue ref (COPY)")
              << std::endl;
    (void)ref2;

    std::cout << "\n  This is EXACTLY what std::vector uses internally during"
              << std::endl;
    std::cout << "  reallocation to decide whether to move or copy elements."
              << std::endl;
}

// =============================================================================
// Demonstrate conditional noexcept
// =============================================================================
namespace conditional_noexcept {

// A wrapper whose noexcept-ness depends on the wrapped type
template<typename T>
class Wrapper {
    T value_;
public:
    explicit Wrapper(T val) : value_(std::move(val)) {}

    // Conditional noexcept: this is noexcept IFF T's move ctor is noexcept
    Wrapper(Wrapper&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
        : value_(std::move(other.value_))
    {}

    // The noexcept OPERATOR (not specifier) tests an expression at compile time
    // noexcept(expr) → returns bool: true if expr is noexcept
    void do_thing() noexcept(noexcept(value_.do_something())) {
        value_.do_something();
    }

    const T& get() const { return value_; }
};

void demo() {
    std::cout << "\n=== Conditional noexcept ===" << std::endl;

    // Wrapper<WidgetNoexcept> move ctor should be noexcept
    std::cout << "  Wrapper<WidgetNoexcept> move is noexcept: "
              << std::is_nothrow_move_constructible_v<Wrapper<WidgetNoexcept>>
              << std::endl;

    // Wrapper<WidgetMayThrow> move ctor should NOT be noexcept
    std::cout << "  Wrapper<WidgetMayThrow> move is noexcept: "
              << std::is_nothrow_move_constructible_v<Wrapper<WidgetMayThrow>>
              << std::endl;

    std::cout << "\n  Pattern: noexcept(std::is_nothrow_move_constructible_v<T>)"
              << std::endl;
    std::cout << "  Propagates noexcept-ness from inner types to wrappers."
              << std::endl;
}

} // namespace conditional_noexcept

// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " The Real Impact of noexcept" << std::endl;
    std::cout << "============================================" << std::endl;

    show_type_traits();
    benchmark();
    demo_move_if_noexcept();
    conditional_noexcept::demo();

    std::cout << "\n============================================" << std::endl;
    std::cout << " Rule: ALWAYS mark move ctors/assignment noexcept" << std::endl;
    std::cout << " if they genuinely cannot throw. Forgetting this" << std::endl;
    std::cout << " silently degrades vector performance." << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
