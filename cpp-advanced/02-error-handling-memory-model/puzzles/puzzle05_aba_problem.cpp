// puzzle05_aba_problem.cpp
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 puzzle05_aba_problem.cpp -o puzzle05
//
// PUZZLE: Watch the ABA problem corrupt a lock-free stack.
//
// Three threads, one shared stack (A → B → C):
//   Thread 1: starts pop, reads head=A, A->next=B, then gets preempted
//   Thread 2: pops A, pops B, pushes A back (A->next is now C, not B!)
//   Thread 1: resumes, CAS(head, A, B) succeeds (head is still A), sets head=B
//             But B was already popped! Stack is corrupted.
//
// This demo uses manual synchronization to force the interleaving.

#include <atomic>
#include <iostream>
#include <thread>
#include <cassert>
#include <chrono>
#include <vector>

using namespace std::chrono_literals;

// ═══════════════════════════════════════════════════════
// Part 1: Demonstrating the ABA bug
// ═══════════════════════════════════════════════════════

namespace aba_bug {

struct Node {
    int value;
    Node* next;
    Node(int v, Node* n = nullptr) : value(v), next(n) {}
};

std::atomic<Node*> head{nullptr};

// Barriers to control interleaving
std::atomic<int> step{0};

void push(Node* node) {
    Node* old_head = head.load(std::memory_order_relaxed);
    do {
        node->next = old_head;
    } while (!head.compare_exchange_weak(old_head, node,
                std::memory_order_release, std::memory_order_relaxed));
}

// Pop WITHOUT ABA protection — the buggy version
Node* pop_buggy() {
    Node* old_head = head.load(std::memory_order_acquire);
    if (!old_head) return nullptr;
    Node* next = old_head->next;
    // Note: in the real bug, a preemption happens between reading next
    // and executing the CAS. We simulate this with step synchronization.
    while (!head.compare_exchange_weak(old_head, next,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
        if (!old_head) return nullptr;
        next = old_head->next;
    }
    return old_head;
}

void demonstrate() {
    std::cout << "=== Part 1: ABA Bug Demonstration ===\n\n";

    // Build stack: head → A(1) → B(2) → C(3)
    Node* C = new Node(3);
    Node* B = new Node(2, C);
    Node* A = new Node(1, B);
    head.store(A, std::memory_order_release);

    std::cout << "Initial stack: A(1) → B(2) → C(3)\n";
    std::cout << "A=" << A << ", B=" << B << ", C=" << C << "\n\n";

    // Thread 1: starts pop, reads head=A, next=B, then waits
    std::thread t1([&]() {
        Node* old_head = head.load(std::memory_order_acquire);
        Node* next = old_head->next;  // next = B
        std::cout << "[T1] Read head=" << old_head->value
                  << " (addr=" << old_head << "), next=" << next->value
                  << " (addr=" << next << ")\n";

        // ── T1 gets preempted here ──
        step.store(1, std::memory_order_release);  // Signal T2 to go
        while (step.load(std::memory_order_acquire) < 2) {}  // Wait for T2

        // Resume: CAS(head, A, B)
        // head is A again (T2 pushed it back), so CAS succeeds!
        bool ok = head.compare_exchange_strong(old_head, next,
                    std::memory_order_acq_rel, std::memory_order_acquire);
        std::cout << "[T1] CAS(head, A, B) = " << (ok ? "SUCCESS" : "FAIL") << "\n";
        std::cout << "[T1] Set head = B (addr=" << next << ")\n";
    });

    // Thread 2: pops A, pops B, rewires A->next=C, pushes A back
    std::thread t2([&]() {
        while (step.load(std::memory_order_acquire) < 1) {}  // Wait for T1 to read

        // Pop A
        Node* popped_a = pop_buggy();
        std::cout << "[T2] Popped A(" << popped_a->value << ")\n";
        // Pop B
        Node* popped_b = pop_buggy();
        std::cout << "[T2] Popped B(" << popped_b->value << ")\n";
        // Stack is now: head → C(3)

        // Rewire A->next to C (simulating reuse or modification)
        popped_a->next = C;
        std::cout << "[T2] Set A->next = C (was B)\n";

        // Push A back
        push(popped_a);
        std::cout << "[T2] Pushed A back. Stack: A(1) → C(3)\n";
        std::cout << "[T2] B(" << popped_b->value << ") at addr=" << popped_b
                  << " is now free/dangling\n";

        step.store(2, std::memory_order_release);  // Signal T1 to resume
    });

    t1.join();
    t2.join();

    // Inspect the damage
    std::cout << "\n[RESULT] Stack after T1's CAS:\n";
    std::cout << "  head = " << head.load() << " (B, value=" << head.load()->value << ")\n";
    std::cout << "  head->next = " << head.load()->next
              << " (value=" << head.load()->next->value << ")\n\n";

    std::cout << "★ BUG: head points to B, but B was already popped!\n";
    std::cout << "  B->next still points to C only because we didn't free B.\n";
    std::cout << "  In a real allocator, B's memory would be reused → CORRUPTION.\n";
    std::cout << "  C is skipped — lost from the stack forever (memory leak).\n";
    std::cout << "  The correct stack should have been: A(1) → C(3)\n\n";

    // Cleanup (we own all nodes, none were freed)
    delete A;
    delete B;
    delete C;
    head.store(nullptr);
}

} // namespace aba_bug

// ═══════════════════════════════════════════════════════
// Part 2: Tagged pointer fix
// ═══════════════════════════════════════════════════════
// Pack tag + pointer into a single 64-bit word.
// x86-64 uses only 48 bits for virtual addresses, leaving 16 bits for a tag.
// This fits in a normal atomic<uint64_t> — no 128-bit CAS needed.

namespace tagged_fix {

struct Node {
    int value;
    Node* next;
    Node(int v, Node* n = nullptr) : value(v), next(n) {}
};

// 16-bit tag in the top bits, 48-bit pointer in the bottom bits
static constexpr uint64_t PTR_MASK = 0x0000'FFFF'FFFF'FFFF;
static constexpr int TAG_SHIFT = 48;

uint64_t pack(Node* ptr, uint16_t tag) {
    return (static_cast<uint64_t>(tag) << TAG_SHIFT)
         | (reinterpret_cast<uint64_t>(ptr) & PTR_MASK);
}
Node* get_ptr(uint64_t packed) {
    return reinterpret_cast<Node*>(packed & PTR_MASK);
}
uint16_t get_tag(uint64_t packed) {
    return static_cast<uint16_t>(packed >> TAG_SHIFT);
}

std::atomic<uint64_t> head{0};

void push(Node* node) {
    uint64_t old_head = head.load(std::memory_order_relaxed);
    uint64_t new_head;
    do {
        node->next = get_ptr(old_head);
        new_head = pack(node, get_tag(old_head) + 1);
    } while (!head.compare_exchange_weak(old_head, new_head,
                std::memory_order_release, std::memory_order_relaxed));
}

Node* pop() {
    uint64_t old_head = head.load(std::memory_order_acquire);
    uint64_t new_head;
    do {
        Node* p = get_ptr(old_head);
        if (!p) return nullptr;
        new_head = pack(p->next, get_tag(old_head) + 1);
    } while (!head.compare_exchange_weak(old_head, new_head,
                std::memory_order_acq_rel, std::memory_order_acquire));
    return get_ptr(old_head);
}

void demonstrate() {
    std::cout << "=== Part 2: Tagged Pointer Fix (packed 64-bit) ===\n\n";

    std::cout << "atomic<uint64_t> is lock-free: "
              << std::boolalpha << head.is_lock_free() << "\n\n";

    Node* C = new Node(3);
    Node* B = new Node(2, C);
    Node* A = new Node(1, B);

    head.store(pack(A, 0), std::memory_order_release);
    std::cout << "Initial stack: A(1) → B(2) → C(3)\n";

    std::atomic<int> step{0};

    std::thread t1([&]() {
        uint64_t old_head = head.load(std::memory_order_acquire);
        Node* old_ptr = get_ptr(old_head);
        uint16_t old_tag = get_tag(old_head);
        Node* next = old_ptr->next;
        std::cout << "[T1] Read head=A (tag=" << old_tag << "), next=B\n";

        step.store(1, std::memory_order_release);
        while (step.load(std::memory_order_acquire) < 2) {}

        // Try CAS: expected {A, tag=0} but head is now {A, tag=3}!
        uint64_t new_head = pack(next, old_tag + 1);
        bool ok = head.compare_exchange_strong(old_head, new_head,
                    std::memory_order_acq_rel, std::memory_order_acquire);
        std::cout << "[T1] CAS(head, {A,tag=0}, {B,tag=1}) = "
                  << (ok ? "SUCCESS" : "FAIL") << "\n";
        if (!ok) {
            std::cout << "[T1] CAS failed because tag changed! (head tag="
                      << get_tag(old_head) << ")\n";
            std::cout << "[T1] ABA detected and prevented! ✓\n";
        }
    });

    std::thread t2([&]() {
        while (step.load(std::memory_order_acquire) < 1) {}

        Node* a = pop();
        std::cout << "[T2] Popped " << a->value << " (tag now: "
                  << get_tag(head.load()) << ")\n";
        Node* b = pop();
        std::cout << "[T2] Popped " << b->value << " (tag now: "
                  << get_tag(head.load()) << ")\n";

        a->next = C;
        push(a);
        std::cout << "[T2] Pushed A back (tag now: "
                  << get_tag(head.load()) << ")\n";
        std::cout << "[T2] Stack: A(1) → C(3)\n";

        step.store(2, std::memory_order_release);
    });

    t1.join();
    t2.join();

    std::cout << "\n[RESULT] Stack is intact: ";
    Node* curr = get_ptr(head.load());
    while (curr) {
        std::cout << curr->value;
        curr = curr->next;
        if (curr) std::cout << " → ";
    }
    std::cout << "\n";

    delete A;
    delete B;
    delete C;
}

} // namespace tagged_fix

int main() {
    aba_bug::demonstrate();
    std::cout << "─────────────────────────────────────────\n\n";
    tagged_fix::demonstrate();
}

/*
 * ═══════════════════════════════════════════════════════
 * ANSWER
 * ═══════════════════════════════════════════════════════
 *
 * THE ABA PROBLEM:
 *
 * compare_exchange (CAS) checks if the value is EQUAL to expected,
 * but "equal" doesn't mean "unchanged". The value A can leave,
 * other things happen, and A comes back — CAS sees A and succeeds,
 * but the world has changed underneath.
 *
 * In a lock-free stack:
 *   T1 reads head=A, A->next=B
 *   T1 gets preempted
 *   T2 pops A, pops B, pushes A (now A->next=C, not B!)
 *   T1 resumes: CAS(head, A, B) succeeds → head=B (WRONG!)
 *     - B was already freed/popped
 *     - C is orphaned (memory leak)
 *     - Accessing B->next is use-after-free
 *
 * FIXES:
 *
 * 1. Tagged pointers (shown above):
 *    Pack a monotonic counter with the pointer. Every modification
 *    bumps the counter. CAS checks both pointer AND counter, so even
 *    if the pointer matches, the counter won't.
 *    Techniques:
 *    - Pack into 64 bits: 16-bit tag + 48-bit pointer (x86-64 uses 48-bit VA)
 *      Fits in a regular atomic<uint64_t> — no special linking needed.
 *    - 128-bit CAS (cmpxchg16b): full 64-bit pointer + 64-bit counter,
 *      requires -latomic and may not be lock-free on all platforms.
 *
 * 2. Hazard pointers:
 *    Each thread publishes which nodes it's currently reading.
 *    A node can't be reused/freed until no thread's hazard pointer
 *    references it. More complex but doesn't waste counter bits.
 *
 * 3. Epoch-based reclamation (crossbeam in Rust):
 *    Nodes are retired but not freed until all threads have advanced
 *    past the epoch in which the node was retired.
 *
 * 4. RCU (Read-Copy-Update):
 *    Similar to epoch but with quiescent state detection.
 *    Used heavily in the Linux kernel.
 *
 * PRACTICAL RULE:
 *   Never use raw CAS on pointers in a lock-free data structure
 *   without ABA protection. The simplest fix is tagged pointers
 *   if your platform supports double-width CAS.
 */
