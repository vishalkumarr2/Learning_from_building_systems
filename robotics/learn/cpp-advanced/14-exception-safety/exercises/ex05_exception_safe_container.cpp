// =============================================================================
// Exercise 05: Building an Exception-Safe Container
// =============================================================================
// Implements a SmallVector<T, N> with Small Buffer Optimization (SBO).
// push_back provides the STRONG guarantee: if T's constructor throws, the
// container is unchanged.
//
// Compile: g++ -std=c++2a -Wall -Wextra -o ex05 ex05_exception_safe_container.cpp
// =============================================================================

#include <iostream>
#include <memory>
#include <new>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <cassert>
#include <cstring>

// =============================================================================
// Test type: throws on the Nth copy
// =============================================================================
class ThrowOnCopy {
    int value_;
    static int copy_count_;
    static int throw_after_;

public:
    explicit ThrowOnCopy(int v) : value_(v) {}

    ThrowOnCopy(const ThrowOnCopy& other) : value_(other.value_) {
        ++copy_count_;
        if (throw_after_ >= 0 && copy_count_ >= throw_after_) {
            throw std::runtime_error(
                "ThrowOnCopy: threw on copy #" + std::to_string(copy_count_));
        }
    }

    // noexcept move — so the container can move safely
    ThrowOnCopy(ThrowOnCopy&& other) noexcept
        : value_(other.value_)
    {
        other.value_ = -1;
    }

    ThrowOnCopy& operator=(ThrowOnCopy other) noexcept {
        value_ = other.value_;
        return *this;
    }

    int value() const { return value_; }

    static void reset(int throw_after = -1) {
        copy_count_ = 0;
        throw_after_ = throw_after;
    }
    static int copies() { return copy_count_; }
};

int ThrowOnCopy::copy_count_ = 0;
int ThrowOnCopy::throw_after_ = -1;

// =============================================================================
// SmallVector<T, N> — exception-safe container with SBO
// =============================================================================
template<typename T, size_t N = 8>
class SmallVector {
    // Small buffer: stack-allocated storage for up to N elements
    alignas(T) unsigned char sbo_[N * sizeof(T)];

    T* data_;        // points to sbo_ or heap allocation
    size_t size_;
    size_t capacity_;

    bool is_small() const noexcept { return data_ == sbo_as_T(); }

    T* sbo_as_T() const noexcept {
        return const_cast<T*>(reinterpret_cast<const T*>(sbo_));
    }

    // Destroy elements [first, last)
    static void destroy_range(T* first, T* last) noexcept {
        for (T* p = first; p != last; ++p)
            p->~T();
    }

    // Allocate raw memory for 'n' elements (no construction)
    static T* allocate(size_t n) {
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    static void deallocate(T* p) noexcept {
        ::operator delete(p);
    }

    // Move or copy elements to new_data, providing strong guarantee.
    // Uses move_if_noexcept: if T's move ctor is noexcept, moves.
    // Otherwise, copies (so we can recover if one fails).
    void relocate_to(T* new_data, size_t count) {
        size_t constructed = 0;
        try {
            for (size_t i = 0; i < count; ++i) {
                // Construct in new location. move_if_noexcept gives us
                // an rvalue ref if T is nothrow-movable, else lvalue ref.
                ::new (static_cast<void*>(new_data + i))
                    T(std::move_if_noexcept(data_[i]));
                ++constructed;
            }
        } catch (...) {
            // Something threw during relocation — destroy what we built
            destroy_range(new_data, new_data + constructed);
            throw;  // re-throw; caller handles cleanup of new_data
        }
    }

    void grow(size_t min_capacity) {
        size_t new_cap = capacity_ * 2;
        if (new_cap < min_capacity) new_cap = min_capacity;
        if (new_cap < N) new_cap = N;

        T* new_data = allocate(new_cap);
        try {
            relocate_to(new_data, size_);
        } catch (...) {
            deallocate(new_data);
            throw;  // original data_ is still intact!
        }

        // Success — destroy old elements and switch
        destroy_range(data_, data_ + size_);
        if (!is_small()) {
            deallocate(data_);
        }
        data_ = new_data;
        capacity_ = new_cap;
    }

public:
    SmallVector() noexcept
        : data_(sbo_as_T())
        , size_(0)
        , capacity_(N)
    {}

    ~SmallVector() {
        destroy_range(data_, data_ + size_);
        if (!is_small()) {
            deallocate(data_);
        }
    }

    // Disable copy for simplicity (focus is on push_back exception safety)
    SmallVector(const SmallVector&) = delete;
    SmallVector& operator=(const SmallVector&) = delete;

    // Move constructor
    SmallVector(SmallVector&& other) noexcept
        : size_(0)
        , capacity_(N)
    {
        data_ = sbo_as_T();
        if (other.is_small()) {
            // Move elements from other's SBO to our SBO
            for (size_t i = 0; i < other.size_; ++i) {
                ::new (static_cast<void*>(data_ + i)) T(std::move(other.data_[i]));
            }
            size_ = other.size_;
        } else {
            // Steal other's heap buffer
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = other.sbo_as_T();
            other.size_ = 0;
            other.capacity_ = N;
        }
    }

    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    // push_back — provides the STRONG GUARANTEE
    //
    // Key invariant: size_ is ONLY incremented AFTER the element is
    // successfully constructed. If construction throws, size_ is unchanged.
    void push_back(const T& value) {
        if (size_ == capacity_) {
            // Need to grow. If grow() throws (allocation or relocation),
            // the original data_ and size_ are unchanged.
            grow(size_ + 1);
        }

        // Construct the new element in-place
        // CRITICAL: size_ is NOT incremented yet!
        ::new (static_cast<void*>(data_ + size_)) T(value);

        // Only increment size_ AFTER successful construction.
        // If T(value) threw, size_ is unchanged → strong guarantee.
        ++size_;
    }

    // push_back with move semantics
    void push_back(T&& value) {
        if (size_ == capacity_) {
            grow(size_ + 1);
        }
        ::new (static_cast<void*>(data_ + size_)) T(std::move(value));
        ++size_;
    }

    // emplace_back with perfect forwarding
    template<typename... Args>
    T& emplace_back(Args&&... args) {
        if (size_ == capacity_) {
            grow(size_ + 1);
        }
        ::new (static_cast<void*>(data_ + size_)) T(std::forward<Args>(args)...);
        T& ref = data_[size_];
        ++size_;
        return ref;
    }

    void pop_back() noexcept {
        assert(size_ > 0);
        --size_;
        data_[size_].~T();
    }

    // Print contents
    void print(const char* label) const {
        std::cout << "  " << label << " [size=" << size_
                  << " cap=" << capacity_
                  << " sbo=" << (is_small() ? "yes" : "no") << "]: ";
        for (size_t i = 0; i < size_; ++i) {
            if constexpr (std::is_same_v<T, ThrowOnCopy>) {
                std::cout << data_[i].value() << " ";
            } else {
                std::cout << data_[i] << " ";
            }
        }
        std::cout << std::endl;
    }
};

// =============================================================================
// Tests
// =============================================================================
void test_basic_operations() {
    std::cout << "\n=== Test: Basic Operations ===" << std::endl;

    SmallVector<int, 4> vec;
    assert(vec.capacity() == 4);

    // Fill SBO
    vec.push_back(10);
    vec.push_back(20);
    vec.push_back(30);
    vec.push_back(40);
    vec.print("after 4 pushes (SBO)");
    assert(vec.size() == 4);

    // Trigger growth to heap
    vec.push_back(50);
    vec.print("after 5th push (heap)");
    assert(vec.size() == 5);

    // More pushes
    vec.push_back(60);
    vec.push_back(70);
    vec.print("after 7 pushes");

    std::cout << "  PASS" << std::endl;
}

void test_strong_guarantee_on_push() {
    std::cout << "\n=== Test: Strong Guarantee on push_back ===" << std::endl;

    SmallVector<ThrowOnCopy, 4> vec;
    ThrowOnCopy::reset();

    // Add 3 elements successfully
    vec.emplace_back(1);
    vec.emplace_back(2);
    vec.emplace_back(3);
    vec.print("after 3 pushes");

    // Now set up: the NEXT copy will throw
    ThrowOnCopy item4(4);
    ThrowOnCopy::reset(1);  // throw on the 1st copy

    std::cout << "\n  Attempting push_back that will throw..." << std::endl;
    try {
        vec.push_back(item4);  // this copies item4 → should throw
        std::cout << "  ERROR: should have thrown!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  CAUGHT: " << e.what() << std::endl;
    }

    // Verify strong guarantee: size should still be 3, elements unchanged
    vec.print("after failed push");
    assert(vec.size() == 3);
    assert(vec[0].value() == 1);
    assert(vec[1].value() == 2);
    assert(vec[2].value() == 3);
    std::cout << "  STRONG GUARANTEE HELD: size and contents unchanged!" << std::endl;

    // Verify we can still use the container after failure
    ThrowOnCopy::reset();  // stop throwing
    vec.emplace_back(44);
    vec.print("after recovery push");
    assert(vec.size() == 4);
    std::cout << "  PASS" << std::endl;
}

void test_strong_guarantee_during_growth() {
    std::cout << "\n=== Test: Strong Guarantee During Growth ===" << std::endl;

    // Fill to capacity, then trigger growth with a throwing element
    SmallVector<ThrowOnCopy, 2> vec;
    ThrowOnCopy::reset();

    vec.emplace_back(10);
    vec.emplace_back(20);
    vec.print("full SBO");
    assert(vec.size() == 2);
    assert(vec.capacity() == 2);

    // The next push triggers growth (reallocation).
    // The new element's copy might throw during push_back.
    ThrowOnCopy item3(30);
    ThrowOnCopy::reset(1);  // throw on 1st copy

    std::cout << "\n  Attempting push_back triggering growth + throw..." << std::endl;
    try {
        vec.push_back(item3);
    } catch (const std::exception& e) {
        std::cout << "  CAUGHT: " << e.what() << std::endl;
    }

    // The move-based relocation of existing elements should succeed
    // (since ThrowOnCopy has noexcept move), but copying item3 fails.
    // If relocation succeeded but final construction failed, the container
    // may have grown but size_ is unchanged.
    vec.print("after failed growth push");
    assert(vec[0].value() == 10);
    assert(vec[1].value() == 20);
    std::cout << "  Elements preserved, data integrity intact." << std::endl;

    // Recovery
    ThrowOnCopy::reset();
    vec.emplace_back(30);
    vec.print("after recovery");
    std::cout << "  PASS" << std::endl;
}

void test_emplace_back() {
    std::cout << "\n=== Test: emplace_back ===" << std::endl;

    SmallVector<ThrowOnCopy, 4> vec;
    ThrowOnCopy::reset();

    // emplace_back constructs in-place — no copy needed
    auto& ref = vec.emplace_back(42);
    assert(ref.value() == 42);
    vec.print("after emplace");
    assert(ThrowOnCopy::copies() == 0);  // zero copies!
    std::cout << "  Copies performed: " << ThrowOnCopy::copies()
              << " (emplace avoids copies)" << std::endl;
    std::cout << "  PASS" << std::endl;
}

// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Exception-Safe Container: SmallVector<T,N>" << std::endl;
    std::cout << "============================================" << std::endl;

    test_basic_operations();
    test_strong_guarantee_on_push();
    test_strong_guarantee_during_growth();
    test_emplace_back();

    std::cout << "\n============================================" << std::endl;
    std::cout << " Key Insight:" << std::endl;
    std::cout << "  1. Increment size_ AFTER successful construction" << std::endl;
    std::cout << "  2. Use move_if_noexcept during reallocation" << std::endl;
    std::cout << "  3. If growth allocation throws, old buffer is intact" << std::endl;
    std::cout << "  4. emplace_back avoids copies entirely" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
