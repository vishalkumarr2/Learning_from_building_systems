// ex06_compare_exchange.cpp — Lock-free stack with compare_exchange_weak
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 ex06_compare_exchange.cpp -o ex06
//
// Topics:
//   1. Lock-free stack (Treiber stack) using CAS
//   2. Stress test with contention statistics
//   3. ABA problem explanation and tagged-pointer solution

#include <atomic>
#include <thread>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <optional>

// ============================================================================
// Part 1: Basic Lock-Free Stack (has ABA problem — we discuss it below)
// ============================================================================

template <typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(const T& d) : data(d), next(nullptr) {}
    };

    std::atomic<Node*> head_{nullptr};

    // Contention counters
    std::atomic<uint64_t> push_cas_failures_{0};
    std::atomic<uint64_t> pop_cas_failures_{0};
    std::atomic<uint64_t> push_count_{0};
    std::atomic<uint64_t> pop_count_{0};

public:
    ~LockFreeStack() {
        // Drain remaining nodes
        Node* n = head_.load(std::memory_order_relaxed);
        while (n) {
            Node* next = n->next;
            delete n;
            n = next;
        }
    }

    void push(const T& value) {
        Node* new_node = new Node(value);
        new_node->next = head_.load(std::memory_order_relaxed);

        // CAS loop: try to set head to new_node.
        // compare_exchange_weak may fail spuriously (returns false even if
        // head == expected). That's OK — we just retry. Weak is faster in
        // loops because it doesn't need to handle spurious failures internally.
        while (!head_.compare_exchange_weak(
                   new_node->next,    // expected (updated on failure)
                   new_node,          // desired
                   std::memory_order_release,   // success: publish the node
                   std::memory_order_relaxed))  // failure: just retry
        {
            push_cas_failures_.fetch_add(1, std::memory_order_relaxed);
        }
        push_count_.fetch_add(1, std::memory_order_relaxed);
    }

    std::optional<T> pop() {
        Node* old_head = head_.load(std::memory_order_acquire);

        while (old_head) {
            // Read old_head->next BEFORE the CAS. If CAS succeeds, old_head
            // is ours and we can safely read it. If it fails, old_head is
            // updated to the current head and we retry.
            //
            // WARNING: This is where ABA can bite us. See Part 3 below.
            Node* next = old_head->next;

            if (head_.compare_exchange_weak(
                    old_head,     // expected (updated on failure)
                    next,         // desired
                    std::memory_order_acquire,
                    std::memory_order_relaxed))
            {
                T result = old_head->data;
                delete old_head;
                pop_count_.fetch_add(1, std::memory_order_relaxed);
                return result;
            }
            pop_cas_failures_.fetch_add(1, std::memory_order_relaxed);
        }
        return std::nullopt;  // stack was empty
    }

    void print_stats(const char* label) const {
        uint64_t pushes = push_count_.load(std::memory_order_relaxed);
        uint64_t pops = pop_count_.load(std::memory_order_relaxed);
        uint64_t push_fail = push_cas_failures_.load(std::memory_order_relaxed);
        uint64_t pop_fail = pop_cas_failures_.load(std::memory_order_relaxed);

        std::cout << "  " << label << ":\n";
        std::cout << "    pushes: " << pushes
                  << "  (CAS failures: " << push_fail;
        if (pushes > 0)
            std::cout << ", retry rate: "
                      << (100.0 * static_cast<double>(push_fail) / static_cast<double>(pushes)) << "%";
        std::cout << ")\n";
        std::cout << "    pops:   " << pops
                  << "  (CAS failures: " << pop_fail;
        if (pops > 0)
            std::cout << ", retry rate: "
                      << (100.0 * static_cast<double>(pop_fail) / static_cast<double>(pops)) << "%";
        std::cout << ")\n";
    }
};

// ============================================================================
// Part 2: Stress test
// ============================================================================

void stress_test_basic() {
    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 100'000;

    LockFreeStack<int> stack;
    std::atomic<int> total_pushed{0};
    std::atomic<int> total_popped{0};

    // Phase 1: all threads push
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&, t]() {
                int base = t * OPS_PER_THREAD;
                for (int i = 0; i < OPS_PER_THREAD; ++i) {
                    stack.push(base + i);
                    total_pushed.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // Phase 2: all threads pop
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&]() {
                int count = 0;
                while (count < OPS_PER_THREAD) {
                    if (stack.pop().has_value()) {
                        total_popped.fetch_add(1, std::memory_order_relaxed);
                        ++count;
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    int pushed = total_pushed.load();
    int popped = total_popped.load();
    std::cout << "  pushed: " << pushed << ", popped: " << popped;
    if (pushed == popped) {
        std::cout << " ✓ balanced\n";
    } else {
        std::cout << " ✗ MISMATCH\n";
    }
    stack.print_stats("Basic stack");
}

// ============================================================================
// Part 3: ABA Problem — Explanation
// ============================================================================
//
// The ABA problem occurs when:
//   1. Thread A reads head_ → Node X (with X->next = Y)
//   2. Thread A is preempted
//   3. Thread B pops X (head_ → Y), pops Y (head_ → Z), pushes X back (head_ → X→Z)
//   4. Thread A resumes, CAS(expected=X, desired=Y) SUCCEEDS
//      because head_ is X again — but X->next is now Z, not Y!
//      We just set head_ to Y, which was already freed. Use-after-free!
//
// Specific interleaving:
//   Initial: head → [A] → [B] → [C] → null
//
//   T1: old_head = A, next = B        (reads head=A, A.next=B)
//   T1: <preempted>
//
//   T2: pop() → removes A             head → [B] → [C] → null
//   T2: pop() → removes B             head → [C] → null
//   T2: delete A, delete B
//   T2: push(new node at same address as A)
//       Allocator returns old A's memory!
//       head → [A'] → [C] → null      (A' has same address as old A)
//
//   T1: <resumes>
//   T1: CAS(expected=A, desired=B)     head_ is A'... same address!
//       CAS succeeds! head → [B]       BUT B WAS FREED! Crash/corruption.
//
// The fundamental issue: CAS compares the pointer VALUE (address), not the
// identity. Same address ≠ same object after free+realloc.

// ============================================================================
// Part 4: Tagged Pointer Stack — Solves ABA
// ============================================================================
// Pack a monotonic counter into the upper bits of the pointer.
// Every CAS increments the counter, so even if the address is reused,
// the tag differs and CAS fails correctly.
//
// On x86-64, virtual addresses use only 48 bits (or 57 with 5-level paging).
// We use the upper 16 bits as a generation counter. This gives 65535 ABA
// cycles before wrap-around (practically sufficient).

template <typename T>
class TaggedLockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(const T& d) : data(d), next(nullptr) {}
    };

    // Tagged pointer: upper 16 bits = tag, lower 48 bits = pointer
    struct TaggedPtr {
        uintptr_t value;

        TaggedPtr() : value(0) {}
        TaggedPtr(Node* ptr, uint16_t tag) {
            value = (static_cast<uintptr_t>(tag) << 48)
                  | (reinterpret_cast<uintptr_t>(ptr) & 0x0000FFFFFFFFFFFF);
        }

        Node* ptr() const {
            // Sign-extend if needed for canonical addresses
            uintptr_t raw = value & 0x0000FFFFFFFFFFFF;
            // If bit 47 is set, sign-extend to make it canonical
            if (raw & (1ULL << 47)) {
                raw |= 0xFFFF000000000000;
            }
            return reinterpret_cast<Node*>(raw);
        }

        uint16_t tag() const {
            return static_cast<uint16_t>(value >> 48);
        }

        bool operator==(const TaggedPtr& other) const { return value == other.value; }
    };

    std::atomic<uintptr_t> head_{0}; // stores TaggedPtr.value

    std::atomic<uint64_t> push_cas_failures_{0};
    std::atomic<uint64_t> pop_cas_failures_{0};
    std::atomic<uint64_t> push_count_{0};
    std::atomic<uint64_t> pop_count_{0};

    TaggedPtr load_head(std::memory_order order) const {
        TaggedPtr tp;
        tp.value = head_.load(order);
        return tp;
    }

    bool cas_head(TaggedPtr& expected, TaggedPtr desired,
                  std::memory_order success, std::memory_order failure) {
        return head_.compare_exchange_weak(expected.value, desired.value,
                                           success, failure);
    }

public:
    ~TaggedLockFreeStack() {
        TaggedPtr cur = load_head(std::memory_order_relaxed);
        Node* n = cur.ptr();
        while (n) {
            Node* next = n->next;
            delete n;
            n = next;
        }
    }

    void push(const T& value) {
        Node* new_node = new Node(value);
        TaggedPtr old_head = load_head(std::memory_order_relaxed);

        TaggedPtr new_head;
        do {
            new_node->next = old_head.ptr();
            // Increment tag on every push to prevent ABA
            new_head = TaggedPtr(new_node, static_cast<uint16_t>(old_head.tag() + 1));
        } while (!cas_head(old_head, new_head,
                           std::memory_order_release,
                           std::memory_order_relaxed) &&
                 (push_cas_failures_.fetch_add(1, std::memory_order_relaxed), true));

        push_count_.fetch_add(1, std::memory_order_relaxed);
    }

    std::optional<T> pop() {
        TaggedPtr old_head = load_head(std::memory_order_acquire);

        while (old_head.ptr()) {
            Node* next = old_head.ptr()->next;
            // Increment tag — even if the same address appears later, the tag
            // will differ, and CAS will correctly fail.
            TaggedPtr new_head(next, static_cast<uint16_t>(old_head.tag() + 1));

            if (cas_head(old_head, new_head,
                         std::memory_order_acquire,
                         std::memory_order_relaxed)) {
                T result = old_head.ptr()->data;
                delete old_head.ptr();
                pop_count_.fetch_add(1, std::memory_order_relaxed);
                return result;
            }
            pop_cas_failures_.fetch_add(1, std::memory_order_relaxed);
        }
        return std::nullopt;
    }

    void print_stats(const char* label) const {
        uint64_t pushes = push_count_.load(std::memory_order_relaxed);
        uint64_t pops = pop_count_.load(std::memory_order_relaxed);
        uint64_t push_fail = push_cas_failures_.load(std::memory_order_relaxed);
        uint64_t pop_fail = pop_cas_failures_.load(std::memory_order_relaxed);

        std::cout << "  " << label << ":\n";
        std::cout << "    pushes: " << pushes
                  << "  (CAS failures: " << push_fail;
        if (pushes > 0)
            std::cout << ", retry rate: "
                      << (100.0 * static_cast<double>(push_fail) / static_cast<double>(pushes)) << "%";
        std::cout << ")\n";
        std::cout << "    pops:   " << pops
                  << "  (CAS failures: " << pop_fail;
        if (pops > 0)
            std::cout << ", retry rate: "
                      << (100.0 * static_cast<double>(pop_fail) / static_cast<double>(pops)) << "%";
        std::cout << ")\n";
    }
};

// ============================================================================
// Stress test for tagged stack
// ============================================================================

void stress_test_tagged() {
    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 100'000;

    TaggedLockFreeStack<int> stack;
    std::atomic<int> total_pushed{0};
    std::atomic<int> total_popped{0};

    // Phase 1: concurrent push
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&, t]() {
                int base = t * OPS_PER_THREAD;
                for (int i = 0; i < OPS_PER_THREAD; ++i) {
                    stack.push(base + i);
                    total_pushed.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // Phase 2: concurrent pop
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&]() {
                int count = 0;
                while (count < OPS_PER_THREAD) {
                    if (stack.pop().has_value()) {
                        total_popped.fetch_add(1, std::memory_order_relaxed);
                        ++count;
                    }
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    int pushed = total_pushed.load();
    int popped = total_popped.load();
    std::cout << "  pushed: " << pushed << ", popped: " << popped;
    if (pushed == popped) {
        std::cout << " ✓ balanced\n";
    } else {
        std::cout << " ✗ MISMATCH\n";
    }
    stack.print_stats("Tagged stack (ABA-safe)");
}

// ============================================================================
// Interleaved stress test — push and pop simultaneously (max contention)
// ============================================================================

template <typename Stack>
void interleaved_stress(Stack& stack, const char* label) {
    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 100'000;

    std::atomic<uint64_t> total_pushed{0};
    std::atomic<uint64_t> total_popped{0};

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                stack.push(t * OPS_PER_THREAD + i);
                total_pushed.fetch_add(1, std::memory_order_relaxed);

                if (stack.pop().has_value()) {
                    total_popped.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& t : threads) t.join();

    auto elapsed = std::chrono::steady_clock::now() - start;
    double ms = std::chrono::duration<double, std::milli>(elapsed).count();

    uint64_t pushed = total_pushed.load();
    uint64_t popped = total_popped.load();
    std::cout << "  " << label << ": "
              << pushed << " push, " << popped << " pop in "
              << ms << " ms\n";
    stack.print_stats(label);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Exercise 06: Lock-Free Stack with CAS ===\n\n";

    std::cout << "[Part 1] Basic Treiber stack stress test\n";
    stress_test_basic();

    std::cout << "\n[Part 2] Tagged pointer stack stress test (ABA-safe)\n";
    stress_test_tagged();

    std::cout << "\n[Part 3] Interleaved push/pop (maximum contention)\n";
    std::cout << "  (Skipping basic stack — concurrent push+pop triggers ABA/double-free.\n";
    std::cout << "   This is EXPECTED: the basic Treiber stack without ABA protection is\n";
    std::cout << "   unsafe when freed nodes are reallocated and reused by concurrent ops.)\n\n";
    {
        TaggedLockFreeStack<int> tagged;
        interleaved_stress(tagged, "Tagged stack (ABA-safe)");
    }

    std::cout << "\n--- ABA Problem Summary ---\n";
    std::cout << "The basic stack is vulnerable to ABA: if a popped node's address\n";
    std::cout << "is reused (common with malloc/new), CAS may succeed incorrectly.\n";
    std::cout << "The tagged pointer adds a generation counter in the upper 16 bits,\n";
    std::cout << "so even if the address matches, the tag won't → CAS fails correctly.\n";
    std::cout << "\nAlternatives to tagged pointers:\n";
    std::cout << "  - Hazard pointers (safe memory reclamation)\n";
    std::cout << "  - Epoch-based reclamation (RCU-style)\n";
    std::cout << "  - std::atomic<std::shared_ptr<T>> (C++20, not lock-free on all platforms)\n";

    return 0;
}
