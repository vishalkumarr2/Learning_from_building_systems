// Module 16: Sanitizer Workshop
// Exercise 03: ThreadSanitizer Bug Hunt
//
// This file contains 5 intentional data races / threading bugs.
// Compile with TSan and run each one:
//
//   g++ -std=c++2a -fsanitize=thread -g -O1 ex03_tsan_bugs.cpp -o ex03_tsan -pthread
//   TSAN_OPTIONS="second_deadlock_stack=1" ./ex03_tsan <bug_number>
//
// Each function demonstrates a different category of threading error.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ============================================================
// Bug 1: Unprotected shared counter
// TSan reports: "data race on address ..."
// Two threads increment the same integer without synchronization.
// ============================================================

// Shared state with no protection
struct SharedCounter {
    int value = 0;  // NOT atomic, NOT mutex-protected
};

__attribute__((noinline))
void bug_unprotected_counter() {
    SharedCounter counter;
    constexpr int iterations = 100000;

    auto increment = [&counter, iterations]() {
        for (int i = 0; i < iterations; ++i) {
            // BUG: unsynchronized read-modify-write from multiple threads
            ++counter.value;
        }
    };

    std::thread t1(increment);
    std::thread t2(increment);
    t1.join();
    t2.join();

    // Expected: 200000, actual: undefined (torn reads/writes)
    std::cout << "bug_unprotected_counter: expected 200000, got "
              << counter.value << "\n";
}

// ============================================================
// Bug 2: Vector race
// TSan reports: "data race on address ..."
// Two threads push to the same std::vector without locking.
// std::vector is not thread-safe for concurrent modification.
// ============================================================
__attribute__((noinline))
void bug_vector_race() {
    std::vector<int> shared_vec;

    auto pusher = [&shared_vec](int start) {
        for (int i = start; i < start + 100; ++i) {
            // BUG: concurrent push_back causes data race on vector internals
            // (size, capacity, data pointer can all be torn)
            shared_vec.push_back(i);
        }
    };

    std::thread t1(pusher, 0);
    std::thread t2(pusher, 1000);
    t1.join();
    t2.join();

    std::cout << "bug_vector_race: vector size = " << shared_vec.size()
              << " (expected 200, likely corrupted)\n";
}

// ============================================================
// Bug 3: Bitfield race
// TSan reports: "data race on address ..."
// Adjacent bitfield members share the same memory location.
// Modifying different bitfield members from different threads is a data race!
// This is a subtle bug — many developers think "different fields = safe".
// ============================================================
struct StatusFlags {
    unsigned int ready : 1;
    unsigned int error : 1;
    unsigned int busy  : 1;
    unsigned int count : 5;
};

__attribute__((noinline))
void bug_bitfield_race() {
    StatusFlags flags{};

    // BUG: Even though t1 writes 'ready' and t2 writes 'error',
    // they are in the SAME byte. This is a data race per C++ standard.
    // The compiler may load-modify-store the entire word for each bitfield write.
    auto writer1 = [&flags]() {
        for (int i = 0; i < 10000; ++i) {
            flags.ready = 1;
            flags.count = (i & 0x1F);
        }
    };

    auto writer2 = [&flags]() {
        for (int i = 0; i < 10000; ++i) {
            flags.error = 1;
            flags.busy = (i & 1);
        }
    };

    std::thread t1(writer1);
    std::thread t2(writer2);
    t1.join();
    t2.join();

    std::cout << "bug_bitfield_race: ready=" << flags.ready
              << " error=" << flags.error
              << " busy=" << flags.busy << "\n";
}

// ============================================================
// Bug 4: "Benign" race — still flagged by TSan
// TSan reports: "data race on address ..."
// A "benign" race: one thread sets a boolean flag, another reads it.
// Many developers argue this is "fine in practice" on x86, but:
//   - It's formally UB per the C++ memory model
//   - TSan correctly flags it
//   - On non-x86 architectures, it genuinely isn't safe
//   - The compiler can optimize assuming no races (reorder, cache in register)
//
// Lesson: there is NO benign data race in C++. Use std::atomic<bool>.
// ============================================================
__attribute__((noinline))
void bug_benign_race() {
    bool stop_flag = false;  // Should be std::atomic<bool>
    int work_done = 0;

    auto worker = [&stop_flag, &work_done]() {
        while (!stop_flag) {  // BUG: reading non-atomic bool set by another thread
            ++work_done;
        }
    };

    std::thread t(worker);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stop_flag = true;  // BUG: writing non-atomic bool read by another thread

    t.join();
    std::cout << "bug_benign_race: work_done = " << work_done
              << " (race on stop_flag)\n";
}

// ============================================================
// Bug 5: Lock-order inversion (potential deadlock)
// TSan reports: "lock-order-inversion (potential deadlock)"
// Classic ABBA deadlock pattern: Thread 1 locks A then B,
// Thread 2 locks B then A. If both reach the first lock simultaneously,
// they deadlock waiting for each other.
// ============================================================
__attribute__((noinline))
void bug_lock_order_inversion() {
    std::mutex mutex_a;
    std::mutex mutex_b;

    auto thread1_work = [&mutex_a, &mutex_b]() {
        for (int i = 0; i < 100; ++i) {
            // Lock order: A then B
            std::lock_guard<std::mutex> lock_a(mutex_a);
            std::lock_guard<std::mutex> lock_b(mutex_b);
            // ... critical section ...
        }
    };

    auto thread2_work = [&mutex_a, &mutex_b]() {
        for (int i = 0; i < 100; ++i) {
            // BUG: Lock order: B then A (inverted!)
            std::lock_guard<std::mutex> lock_b(mutex_b);
            std::lock_guard<std::mutex> lock_a(mutex_a);
            // ... critical section ...
        }
    };

    std::thread t1(thread1_work);
    std::thread t2(thread2_work);
    t1.join();
    t2.join();

    std::cout << "bug_lock_order_inversion: completed without actual deadlock "
              << "(but TSan detected the potential)\n";
}

// ============================================================
// Main: select bug by command-line argument
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bug_number 1-5>\n";
        std::cerr << "  1: unprotected shared counter (classic data race)\n";
        std::cerr << "  2: vector race (concurrent push_back)\n";
        std::cerr << "  3: bitfield race (adjacent fields = same memory word)\n";
        std::cerr << "  4: 'benign' race (bool flag — still UB!)\n";
        std::cerr << "  5: lock-order inversion (ABBA deadlock pattern)\n";
        return 1;
    }

    int bug = std::atoi(argv[1]);

    switch (bug) {
        case 1: bug_unprotected_counter(); break;
        case 2: bug_vector_race(); break;
        case 3: bug_bitfield_race(); break;
        case 4: bug_benign_race(); break;
        case 5: bug_lock_order_inversion(); break;
        default:
            std::cerr << "Invalid bug number: " << bug << " (valid: 1-5)\n";
            return 1;
    }

    return 0;
}
