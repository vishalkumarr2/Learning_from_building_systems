// Module 16: Sanitizer Workshop
// Exercise 04: Sanitizer CI — Realistic Mini-Project
//
// A MessageQueue<T> class implementing a bounded producer-consumer queue.
// This code APPEARS to work correctly most of the time, but contains:
//   1. One memory bug (caught by ASan)
//   2. One undefined behavior (caught by UBSan)
//   3. One data race (caught by TSan)
//
// Task:
//   1. Build and run without sanitizers — observe it "works"
//   2. Run under ASan:  find the memory bug
//   3. Run under UBSan: find the undefined behavior
//   4. Run under TSan:  find the data race
//   5. Fix all bugs (see #ifdef FIXED version at bottom)
//   6. Verify all 3 sanitizers pass
//
// Build commands:
//   g++ -std=c++2a -g -O1 ex04_sanitizer_ci.cpp -o ex04 -pthread
//   g++ -std=c++2a -fsanitize=address -fno-omit-frame-pointer -g ex04_sanitizer_ci.cpp -o ex04_asan -pthread
//   g++ -std=c++2a -fsanitize=undefined -g ex04_sanitizer_ci.cpp -o ex04_ubsan -pthread
//   g++ -std=c++2a -fsanitize=thread -g ex04_sanitizer_ci.cpp -o ex04_tsan -pthread
//
// To build the fixed version:
//   g++ -std=c++2a -DFIXED -fsanitize=address,undefined -fno-omit-frame-pointer -g ex04_sanitizer_ci.cpp -o ex04_fixed_au -pthread
//   g++ -std=c++2a -DFIXED -fsanitize=thread -g ex04_sanitizer_ci.cpp -o ex04_fixed_t -pthread

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#ifndef FIXED
// ================================================================
// BUGGY VERSION — contains 3 bugs for sanitizers to find
// ================================================================

template <typename T, std::size_t Capacity = 16>
class MessageQueue {
public:
    MessageQueue() = default;

    // Producer: push a message into the queue
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() { return count_ < Capacity; });

        // BUG #3 (TSan): stats_total_pushed_ is accessed without the lock held
        // elsewhere (in get_stats()), creating a data race.
        buffer_[write_idx_] = item;
        write_idx_ = (write_idx_ + 1) % Capacity;
        ++count_;
        ++stats_total_pushed_;

        lock.unlock();
        cv_not_empty_.notify_one();
    }

    // Consumer: pop a message from the queue
    std::optional<T> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!cv_not_empty_.wait_for(lock, timeout,
                                     [this]() { return count_ > 0; })) {
            return std::nullopt;
        }

        T item = buffer_[read_idx_];
        read_idx_ = (read_idx_ + 1) % Capacity;

        // BUG #2 (UBSan): signed integer overflow when count_ is 0
        // The decrement of count_ is fine here, but compute_priority() below
        // triggers UB with the message value.
        --count_;

        lock.unlock();
        cv_not_full_.notify_one();
        return item;
    }

    // Get statistics — called from monitoring thread
    std::size_t get_stats() const {
        // BUG #3 (TSan): reading stats_total_pushed_ without holding mutex_
        // This races with push() which writes stats_total_pushed_ under the lock.
        // A monitoring thread calling get_stats() concurrently with push() = race.
        return stats_total_pushed_;
    }

private:
    std::array<T, Capacity> buffer_{};
    std::size_t read_idx_ = 0;
    std::size_t write_idx_ = 0;
    std::size_t count_ = 0;
    std::size_t stats_total_pushed_ = 0;

    std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
};

// Message type used in the queue
struct SensorMessage {
    int sensor_id;
    int reading;
    int priority;
};

// BUG #2 (UBSan): This function has signed overflow
// When reading values approach INT_MAX, the multiplication overflows.
__attribute__((noinline))
int compute_priority(int reading) {
    volatile int scale = 1000000;
    // BUG: if reading > ~2147, this overflows (INT_MAX / 1000000 ≈ 2147)
    int priority = reading * scale;
    return priority;
}

// BUG #1 (ASan): use-after-free in the consumer
__attribute__((noinline))
void process_messages(MessageQueue<SensorMessage>& queue,
                      std::atomic<bool>& running) {
    // BUG #1: we allocate a "processing buffer" and free it too early
    int* processing_buffer = new int[8];

    while (running.load()) {
        auto msg = queue.pop(std::chrono::milliseconds(50));
        if (!msg.has_value()) continue;

        // Process the message
        processing_buffer[msg->sensor_id % 8] = msg->reading;

        if (msg->reading > 2000) {
            // BUG #1 (ASan): free the buffer, then continue the loop
            // which will access it again on next iteration
            delete[] processing_buffer;
            processing_buffer = nullptr;

            // Oops — next iteration will deref nullptr or use-after-free
            // depending on timing. In practice this only triggers when
            // a "high reading" message is followed by another message.
        }

        // BUG #2 (UBSan): compute priority with potentially large values
        compute_priority(msg->reading);
    }

    delete[] processing_buffer;  // May double-free if already deleted above
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    std::cout << "=== MessageQueue CI Test ===\n";

    MessageQueue<SensorMessage> queue;
    std::atomic<bool> running{true};

    // Producer thread: generate sensor messages
    std::thread producer([&queue, &running]() {
        for (int i = 0; i < 100 && running.load(); ++i) {
            SensorMessage msg{};
            msg.sensor_id = i % 4;
            msg.reading = i * 50;  // Some will exceed 2000, triggering bugs
            msg.priority = 0;
            queue.push(msg);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Consumer thread
    std::thread consumer([&queue, &running]() {
        process_messages(queue, running);
    });

    // Monitor thread: periodically reads stats (causes TSan race)
    std::thread monitor([&queue, &running]() {
        while (running.load()) {
            auto stats = queue.get_stats();  // BUG #3: data race here
            (void)stats;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    producer.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    running.store(false);
    consumer.join();
    monitor.join();

    std::cout << "Total messages pushed: " << queue.get_stats() << "\n";
    std::cout << "=== Done ===\n";
    return 0;
}

#else
// ================================================================
// FIXED VERSION — all 3 bugs corrected
// ================================================================

template <typename T, std::size_t Capacity = 16>
class MessageQueue {
public:
    MessageQueue() = default;

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() { return count_ < Capacity; });

        buffer_[write_idx_] = item;
        write_idx_ = (write_idx_ + 1) % Capacity;
        ++count_;
        // FIX #3: stats is now atomic, no race with get_stats()
        stats_total_pushed_.fetch_add(1, std::memory_order_relaxed);

        lock.unlock();
        cv_not_empty_.notify_one();
    }

    std::optional<T> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!cv_not_empty_.wait_for(lock, timeout,
                                     [this]() { return count_ > 0; })) {
            return std::nullopt;
        }

        T item = buffer_[read_idx_];
        read_idx_ = (read_idx_ + 1) % Capacity;
        --count_;

        lock.unlock();
        cv_not_full_.notify_one();
        return item;
    }

    // FIX #3: stats_total_pushed_ is now atomic — no lock needed for reads
    std::size_t get_stats() const {
        return stats_total_pushed_.load(std::memory_order_relaxed);
    }

private:
    std::array<T, Capacity> buffer_{};
    std::size_t read_idx_ = 0;
    std::size_t write_idx_ = 0;
    std::size_t count_ = 0;
    std::atomic<std::size_t> stats_total_pushed_{0};  // FIX #3: atomic

    std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
};

struct SensorMessage {
    int sensor_id;
    int reading;
    int priority;
};

// FIX #2: use safe arithmetic that doesn't overflow
__attribute__((noinline))
int compute_priority(int reading) {
    // Clamp reading to prevent overflow in multiplication
    constexpr int max_safe = 2000;
    int clamped = (reading > max_safe) ? max_safe : reading;
    int priority = clamped * 1000;  // Safe: 2000 * 1000 < INT_MAX
    return priority;
}

__attribute__((noinline))
void process_messages(MessageQueue<SensorMessage>& queue,
                      std::atomic<bool>& running) {
    int processing_buffer[8] = {};  // FIX #1: stack buffer, no dynamic alloc

    while (running.load()) {
        auto msg = queue.pop(std::chrono::milliseconds(50));
        if (!msg.has_value()) continue;

        // FIX #1: no delete/re-alloc — always safe to access
        processing_buffer[msg->sensor_id % 8] = msg->reading;
        compute_priority(msg->reading);
    }
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    std::cout << "=== MessageQueue CI Test (FIXED) ===\n";

    MessageQueue<SensorMessage> queue;
    std::atomic<bool> running{true};

    std::thread producer([&queue, &running]() {
        for (int i = 0; i < 100 && running.load(); ++i) {
            SensorMessage msg{};
            msg.sensor_id = i % 4;
            msg.reading = i * 50;
            msg.priority = 0;
            queue.push(msg);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::thread consumer([&queue, &running]() {
        process_messages(queue, running);
    });

    std::thread monitor([&queue, &running]() {
        while (running.load()) {
            auto stats = queue.get_stats();  // Safe: atomic read
            (void)stats;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    producer.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    running.store(false);
    consumer.join();
    monitor.join();

    std::cout << "Total messages pushed: " << queue.get_stats() << "\n";
    std::cout << "=== Done (all sanitizers should pass) ===\n";
    return 0;
}

#endif  // FIXED
