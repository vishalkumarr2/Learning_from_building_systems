// Module 16: Sanitizer Workshop
// Exercise 01: AddressSanitizer Bug Hunt
//
// This file contains 8 intentional memory bugs. Each is isolated in its own
// function. Compile with ASan and run each one to see the error report:
//
//   g++ -std=c++2a -fsanitize=address -fno-omit-frame-pointer -g -O1
//       ex01_asan_bugs.cpp -o ex01_asan
//   ./ex01_asan <bug_number>    (1-8)
//
// ASAN_OPTIONS="detect_stack_use_after_return=1:detect_leaks=1" ./ex01_asan 6

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ============================================================
// Bug 1: Heap-buffer-overflow
// ASan reports: "heap-buffer-overflow on address ..."
// Writing 1 byte past the end of a 10-byte heap allocation.
// ============================================================
__attribute__((noinline))
void bug_heap_overflow() {
    char* buf = new char[10];
    std::memset(buf, 'A', 10);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
    // BUG: writing at index 10, which is one past the end
    volatile char* p = buf;
    p[10] = 'X';
#pragma GCC diagnostic pop

    std::cout << "bug_heap_overflow: wrote past heap buffer\n";
    delete[] buf;
}

// ============================================================
// Bug 2: Stack-buffer-overflow
// ASan reports: "stack-buffer-overflow on address ..."
// Writing past the end of a stack-allocated array.
// ============================================================
__attribute__((noinline))
void bug_stack_overflow() {
    volatile int arr[5] = {1, 2, 3, 4, 5};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
    // BUG: index 5 is out of bounds (valid: 0-4)
    // Use a runtime-computed index to avoid compiler warnings
    volatile int idx = 5;
    arr[idx] = 999;
#pragma GCC diagnostic pop

    std::cout << "bug_stack_overflow: wrote past stack array, arr[5]=" << arr[5] << "\n";
}

// ============================================================
// Bug 3: Use-after-free
// ASan reports: "heap-use-after-free on address ..."
// Accessing memory after it has been freed.
// ============================================================
__attribute__((noinline))
void bug_use_after_free() {
    int* p = new int[5];
    p[0] = 42;
    delete[] p;

    // BUG: reading freed memory
    volatile int val = p[0];
    std::cout << "bug_use_after_free: read " << val << " from freed memory\n";
}

// ============================================================
// Bug 4: Use-after-scope
// ASan reports: "stack-use-after-scope on address ..."
// A pointer to a local variable is used after the variable's scope ends.
// ============================================================
__attribute__((noinline))
int* get_scoped_ptr() {
    int local = 42;
    volatile int* p = &local;
    return const_cast<int*>(const_cast<volatile int*>(p));
}

__attribute__((noinline))
void bug_use_after_scope() {
    int* dangling = get_scoped_ptr();
    // BUG: 'local' is out of scope, pointer is dangling
    volatile int val = *dangling;
    std::cout << "bug_use_after_scope: read " << val << " from dead scope\n";
}

// ============================================================
// Bug 5: Double-free
// ASan reports: "attempting double-free on address ..."
// Freeing the same pointer twice.
// ============================================================
__attribute__((noinline))
void bug_double_free() {
    volatile int* p = new int(123);
    delete const_cast<int*>(const_cast<volatile int*>(p));

    // BUG: freeing the same pointer again
    delete const_cast<int*>(const_cast<volatile int*>(p));

    std::cout << "bug_double_free: freed same pointer twice\n";
}

// ============================================================
// Bug 6: Heap-use-after-return (stack use after return)
// Requires: ASAN_OPTIONS=detect_stack_use_after_return=1
// ASan reports: "stack-use-after-return" or "use-after-scope"
// Returning address of a stack variable.
// ============================================================
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
__attribute__((noinline))
int* create_stack_int() {
    int value = 0xDEAD;
    // BUG: returning address of local stack variable
    return &value;
}
#pragma GCC diagnostic pop

__attribute__((noinline))
void bug_heap_use_after_return() {
    int* p = create_stack_int();
    // BUG: reading from returned stack address
    volatile int val = *p;
    std::cout << "bug_heap_use_after_return: read " << val << " from dead stack frame\n";
}

// ============================================================
// Bug 7: Memory leak
// ASan (with LSan) reports: "detected memory leaks"
// Allocating memory without ever freeing it.
// ============================================================
__attribute__((noinline))
void bug_memory_leak() {
    // BUG: allocated but never freed
    volatile int* leak1 = new int[100];
    leak1[0] = 42;  // Use it to prevent optimization

    volatile char* leak2 = new char[256];
    leak2[0] = 'X';

    std::cout << "bug_memory_leak: leaked 100 ints + 256 chars\n";
    // No delete — LSan will report this at program exit
}

// ============================================================
// Bug 8: Container overflow (std::vector out-of-bounds via pointer)
// ASan with container overflow detection reports the out-of-bounds.
// This demonstrates that ASan can catch OOB access on vector internal storage.
// ============================================================
__attribute__((noinline))
void bug_container_overflow() {
    std::vector<int> vec = {10, 20, 30, 40, 50};

    // Get raw pointer to internal storage
    int* raw = vec.data();

    // BUG: accessing past the vector's size via raw pointer arithmetic
    // The vector has 5 elements, but capacity might be larger.
    // Force a small vector and access well past its allocation.
    volatile int val = raw[100];  // Way past the end
    std::cout << "bug_container_overflow: read vec[100] = " << val << "\n";
}

// ============================================================
// Main: select bug by command-line argument
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bug_number 1-8>\n";
        std::cerr << "  1: heap-buffer-overflow\n";
        std::cerr << "  2: stack-buffer-overflow\n";
        std::cerr << "  3: use-after-free\n";
        std::cerr << "  4: use-after-scope\n";
        std::cerr << "  5: double-free\n";
        std::cerr << "  6: heap-use-after-return (needs ASAN_OPTIONS=detect_stack_use_after_return=1)\n";
        std::cerr << "  7: memory-leak\n";
        std::cerr << "  8: container-overflow\n";
        return 1;
    }

    int bug = std::atoi(argv[1]);

    switch (bug) {
        case 1: bug_heap_overflow(); break;
        case 2: bug_stack_overflow(); break;
        case 3: bug_use_after_free(); break;
        case 4: bug_use_after_scope(); break;
        case 5: bug_double_free(); break;
        case 6: bug_heap_use_after_return(); break;
        case 7: bug_memory_leak(); break;
        case 8: bug_container_overflow(); break;
        default:
            std::cerr << "Invalid bug number: " << bug << " (valid: 1-8)\n";
            return 1;
    }

    return 0;
}
