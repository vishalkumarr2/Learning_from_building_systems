// =============================================================================
// Exercise 02: Copy-and-Swap Idiom Deep Dive
// =============================================================================
// Shows why naive assignment is NOT exception-safe, then demonstrates the
// copy-and-swap pattern that provides the strong guarantee.
// Also includes a throwing allocator to prove correctness under failure.
//
// Compile: g++ -std=c++2a -Wall -Wextra -o ex02 ex02_copy_and_swap.cpp
// =============================================================================

#include <iostream>
#include <algorithm>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <cassert>
#include <memory>

// =============================================================================
// Utility: An allocator that throws after N allocations (for testing)
// =============================================================================
static int g_alloc_count = 0;
static int g_alloc_limit = -1;  // -1 = no limit

void* throwing_alloc(size_t bytes) {
    if (g_alloc_limit >= 0 && g_alloc_count >= g_alloc_limit) {
        throw std::bad_alloc();
    }
    ++g_alloc_count;
    return ::operator new(bytes);
}

void reset_alloc_counter(int limit = -1) {
    g_alloc_count = 0;
    g_alloc_limit = limit;
}

// =============================================================================
// VERSION 1: NAIVE ASSIGNMENT (WRONG — not exception-safe)
// =============================================================================
namespace naive {

class DynamicArray {
    int* data_;
    size_t size_;

public:
    explicit DynamicArray(size_t n, int val = 0)
        : data_(static_cast<int*>(throwing_alloc(n * sizeof(int))))
        , size_(n)
    {
        std::fill(data_, data_ + size_, val);
    }

    ~DynamicArray() { ::operator delete(data_); }

    // Copy constructor
    DynamicArray(const DynamicArray& other)
        : data_(static_cast<int*>(throwing_alloc(other.size_ * sizeof(int))))
        , size_(other.size_)
    {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // WRONG: Naive assignment operator — NOT exception-safe!
    //
    // Problem: If throwing_alloc fails, we've already deleted old data.
    // The object is now in a corrupted state: data_ is dangling, size_ is old.
    // We null out data_ after delete to avoid double-free in this demo,
    // but in real code the object would be in an unusable state.
    DynamicArray& operator=(const DynamicArray& other) {
        if (this == &other) return *this;

        ::operator delete(data_);   // Step 1: DELETE OLD DATA (point of no return!)
        data_ = nullptr;            // Prevent double-free (demo safety net only)

        // Step 2: Allocate new — if this THROWS, data_ is null!
        // The object is now in an invalid state: old data was freed,
        // but we have no new buffer. The array is empty/broken.
        data_ = static_cast<int*>(throwing_alloc(other.size_ * sizeof(int)));

        size_ = other.size_;
        std::copy(other.data_, other.data_ + size_, data_);
        return *this;
    }

    size_t size() const { return size_; }
    int operator[](size_t i) const { return data_[i]; }
    int& operator[](size_t i) { return data_[i]; }

    void print(const char* label) const {
        std::cout << "  " << label << " [size=" << size_ << "]: ";
        for (size_t i = 0; i < size_ && i < 8; ++i)
            std::cout << data_[i] << " ";
        if (size_ > 8) std::cout << "...";
        std::cout << std::endl;
    }
};

void demo() {
    std::cout << "\n=== NAIVE ASSIGNMENT (WRONG) ===" << std::endl;

    reset_alloc_counter();
    DynamicArray a(5, 42);
    DynamicArray b(3, 99);
    a.print("a (before)");
    b.print("b (before)");

    // Force the allocation inside operator= to fail
    reset_alloc_counter(0);  // next alloc throws
    try {
        a = b;  // DANGEROUS: deletes a's data, then throws during alloc
    } catch (const std::bad_alloc&) {
        std::cout << "  CAUGHT bad_alloc during assignment!" << std::endl;
        // a is now CORRUPTED — data_ was freed, no new buffer allocated.
        // We can't even safely print 'a' or call its destructor properly.
        // In real code this would be undefined behavior.
        std::cout << "  a is now in a CORRUPTED state (dangling pointer)!" << std::endl;
        std::cout << "  (We skip printing 'a' to avoid UB)" << std::endl;
    }
    reset_alloc_counter();
}

} // namespace naive

// =============================================================================
// VERSION 2: COPY-AND-SWAP (CORRECT — strong guarantee)
// =============================================================================
namespace copy_and_swap {

class DynamicArray {
    int* data_;
    size_t size_;

public:
    explicit DynamicArray(size_t n, int val = 0)
        : data_(static_cast<int*>(throwing_alloc(n * sizeof(int))))
        , size_(n)
    {
        std::fill(data_, data_ + size_, val);
    }

    ~DynamicArray() { ::operator delete(data_); }

    // Copy constructor — may throw (allocates)
    DynamicArray(const DynamicArray& other)
        : data_(static_cast<int*>(throwing_alloc(other.size_ * sizeof(int))))
        , size_(other.size_)
    {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // Move constructor — must be noexcept for container efficiency
    DynamicArray(DynamicArray&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // THE KEY: noexcept swap — this is what makes copy-and-swap work
    friend void swap(DynamicArray& a, DynamicArray& b) noexcept {
        using std::swap;
        swap(a.data_, b.data_);   // pointer swap: noexcept
        swap(a.size_, b.size_);   // size_t swap: noexcept
    }

    // Copy-and-swap assignment operator — STRONG GUARANTEE
    //
    // How it works:
    // 1. The parameter 'other' is passed BY VALUE → the copy happens HERE
    //    (in the caller's context, before any state is modified)
    // 2. If the copy throws (bad_alloc), 'this' is completely untouched
    // 3. If copy succeeds, we swap — which is noexcept
    // 4. The old data (now in 'other') is destroyed when 'other' goes out of scope
    //
    // This handles BOTH copy-assignment AND move-assignment:
    //  - copy assignment: copy ctor called for 'other' parameter
    //  - move assignment: move ctor called for 'other' parameter (no allocation!)
    DynamicArray& operator=(DynamicArray other) {  // NOTE: pass by value!
        swap(*this, other);  // noexcept — always succeeds
        return *this;
        // 'other' (holding our old data) is destroyed here
    }

    size_t size() const { return size_; }
    int operator[](size_t i) const { return data_[i]; }
    int& operator[](size_t i) { return data_[i]; }

    void print(const char* label) const {
        std::cout << "  " << label << " [size=" << size_ << "]: ";
        if (!data_) { std::cout << "(null)" << std::endl; return; }
        for (size_t i = 0; i < size_ && i < 8; ++i)
            std::cout << data_[i] << " ";
        if (size_ > 8) std::cout << "...";
        std::cout << std::endl;
    }
};

void demo() {
    std::cout << "\n=== COPY-AND-SWAP (CORRECT — strong guarantee) ===" << std::endl;

    reset_alloc_counter();
    DynamicArray a(5, 42);
    DynamicArray b(3, 99);
    a.print("a (before)");
    b.print("b (before)");

    // Force the copy (inside operator= parameter) to fail
    reset_alloc_counter(0);  // next alloc throws
    try {
        a = b;  // Copy ctor for parameter throws BEFORE swap happens
    } catch (const std::bad_alloc&) {
        std::cout << "  CAUGHT bad_alloc during assignment!" << std::endl;
        // a is UNTOUCHED — strong guarantee!
        a.print("a (after failed assign)");
        std::cout << "  a is still valid! Strong guarantee held." << std::endl;
    }

    // Successful copy assignment
    reset_alloc_counter();
    a = b;
    a.print("a (after successful assign)");
    assert(a.size() == 3 && a[0] == 99);

    // Move assignment (no allocation needed — just swaps)
    a = DynamicArray(4, 77);
    a.print("a (after move assign)");
    assert(a.size() == 4 && a[0] == 77);

    std::cout << "  All assignments exception-safe!" << std::endl;
    reset_alloc_counter();
}

} // namespace copy_and_swap

// =============================================================================
// VERSION 3: PIMPL + COPY-AND-SWAP for complex objects
// =============================================================================
// When a class has many members, pimpl makes swap trivially noexcept
// (just swap one pointer).
namespace pimpl_variant {

class Widget {
    struct Impl {
        std::string name;
        int* data;
        size_t size;

        Impl(const std::string& n, size_t s)
            : name(n)
            , data(new int[s])
            , size(s)
        {
            std::fill(data, data + size, 0);
        }

        ~Impl() { delete[] data; }

        Impl(const Impl& other)
            : name(other.name)
            , data(new int[other.size])
            , size(other.size)
        {
            std::copy(other.data, other.data + size, data);
        }

        // No copy assignment needed — we use the outer class's copy-and-swap
    };

    std::unique_ptr<Impl> pimpl_;

public:
    Widget(const std::string& name, size_t size)
        : pimpl_(std::make_unique<Impl>(name, size))
    {}

    // Copy ctor: may throw (allocates a new Impl)
    Widget(const Widget& other)
        : pimpl_(std::make_unique<Impl>(*other.pimpl_))
    {}

    // Move ctor: noexcept (just moves the pointer)
    Widget(Widget&& other) noexcept = default;

    // Swap: noexcept — just swap one pointer!
    friend void swap(Widget& a, Widget& b) noexcept {
        a.pimpl_.swap(b.pimpl_);
    }

    // Copy-and-swap assignment — same pattern
    Widget& operator=(Widget other) {
        swap(*this, other);
        return *this;
    }

    void print(const char* label) const {
        std::cout << "  " << label << " [" << pimpl_->name
                  << ", size=" << pimpl_->size << "]" << std::endl;
    }
};

void demo() {
    std::cout << "\n=== PIMPL + COPY-AND-SWAP ===" << std::endl;

    Widget w1("alpha", 10);
    Widget w2("beta", 20);
    w1.print("w1");
    w2.print("w2");

    w1 = w2;
    w1.print("w1 (after w1=w2)");

    // Move assignment
    w1 = Widget("gamma", 5);
    w1.print("w1 (after move)");

    std::cout << "  Pimpl makes swap trivially noexcept (single pointer swap)."
              << std::endl;
}

} // namespace pimpl_variant

// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Copy-and-Swap Idiom Deep Dive" << std::endl;
    std::cout << "============================================" << std::endl;

    naive::demo();
    copy_and_swap::demo();
    pimpl_variant::demo();

    std::cout << "\n============================================" << std::endl;
    std::cout << " Key Insight:" << std::endl;
    std::cout << "  Naive assignment: delete first, then allocate → UNSAFE" << std::endl;
    std::cout << "  Copy-and-swap:    copy first (may throw), then swap → SAFE" << std::endl;
    std::cout << "  Pimpl variant:    reduces swap to single pointer exchange" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
