// =============================================================================
// Exercise 04: Lock-Free Triple Buffer
// =============================================================================
// Implement a lock-free triple buffer for producer-consumer communication
// between tasks running at different rates.
//
// Problem:
//   - Sensor fusion runs at 100Hz producing state estimates
//   - Controller reads at 50Hz (every other cycle)
//   - Motion planner reads at 10Hz (every 10th cycle)
//   - No locks allowed (deadline violations, priority inversion)
//
// Triple-buffer protocol:
//   - 3 slots: [writing] [clean] [reading]
//   - Writer always writes to the "writing" slot, then swaps with "clean"
//   - Reader always reads from the "reading" slot, swaps with "clean" first
//   - No blocking: writer never waits for reader, reader never waits for writer
//   - Reader always gets the latest COMPLETE data (may skip intermediate)
//
// Production relevance:
//   - AUTOSAR Adaptive uses this for inter-process communication
//   - Lock-free buffers prevent priority inversion (Mars Pathfinder incident!)
//   - Critical for multi-rate sensor fusion architectures
// =============================================================================

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

// ============================================================================
// Triple Buffer (Lock-Free, Single Producer, Single Consumer)
// ============================================================================
template <typename T>
class TripleBuffer {
    // Three data slots
    std::array<T, 3> buffer_{};

    // Atomic index control
    // Bits: [writer_idx:2][clean_idx:2][reader_idx:2][new_data:1]
    //
    // Simpler encoding: use 3 atomic indices + a flag
    std::atomic<uint8_t> write_idx_{0};
    std::atomic<uint8_t> clean_idx_{1};
    std::atomic<uint8_t> read_idx_{2};
    std::atomic<bool> new_data_{false};

    uint64_t write_count_{0};
    uint64_t read_count_{0};
    uint64_t skip_count_{0};

public:
    // Writer: store new value, mark as available
    void write(T const& value) {
        uint8_t wi = write_idx_.load(std::memory_order_relaxed);
        buffer_[wi] = value;

        // Swap write slot with clean slot
        uint8_t ci = clean_idx_.exchange(wi, std::memory_order_acq_rel);
        write_idx_.store(ci, std::memory_order_relaxed);
        new_data_.store(true, std::memory_order_release);
        ++write_count_;
    }

    // Reader: get latest value (may skip intermediate writes)
    // Returns true if new data was available
    bool read(T& out) {
        if (!new_data_.load(std::memory_order_acquire)) {
            // No new data — reader gets stale value
            uint8_t ri = read_idx_.load(std::memory_order_relaxed);
            out = buffer_[ri];
            return false;
        }

        // Swap read slot with clean slot (which has latest data)
        new_data_.store(false, std::memory_order_relaxed);
        uint8_t ri = read_idx_.load(std::memory_order_relaxed);
        uint8_t ci = clean_idx_.exchange(ri, std::memory_order_acq_rel);
        read_idx_.store(ci, std::memory_order_relaxed);

        out = buffer_[ci];
        ++read_count_;
        return true;
    }

    // Statistics
    uint64_t writes() const { return write_count_; }
    uint64_t reads() const { return read_count_; }
};

// ============================================================================
// Sample data type (vehicle state estimate)
// ============================================================================
struct VehicleState {
    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;
    double speed = 0.0;
    uint64_t sequence = 0;
    uint64_t timestamp_us = 0;
};

// ============================================================================
// Self-Test
// ============================================================================
void test_basic_write_read() {
    std::cout << "--- Test: Basic write → read ---\n";
    TripleBuffer<VehicleState> buf;

    VehicleState state{1.0, 2.0, 0.5, 10.0, 1, 1000};
    buf.write(state);

    VehicleState out;
    bool got_new = buf.read(out);
    assert(got_new);
    assert(out.x == 1.0);
    assert(out.y == 2.0);
    assert(out.sequence == 1);
    std::cout << "  PASS\n";
}

void test_latest_value_wins() {
    std::cout << "--- Test: Latest value wins ---\n";
    TripleBuffer<VehicleState> buf;

    // Write 3 values before reading
    for (uint64_t i = 1; i <= 3; ++i) {
        VehicleState s{};
        s.sequence = i;
        s.x = static_cast<double>(i);
        buf.write(s);
    }

    VehicleState out;
    bool got_new = buf.read(out);
    assert(got_new);
    // Should get the latest (seq 3), skipping 1 and 2
    assert(out.sequence == 3);
    assert(out.x == 3.0);
    std::cout << "  PASS: Got sequence " << out.sequence << "\n";
}

void test_no_new_data() {
    std::cout << "--- Test: No new data returns false ---\n";
    TripleBuffer<VehicleState> buf;

    VehicleState state{5.0, 6.0, 0.0, 0.0, 42, 0};
    buf.write(state);

    VehicleState out;
    buf.read(out);  // consume
    bool got_new = buf.read(out);  // no new data
    assert(!got_new);
    // Still returns last known value
    std::cout << "  PASS: read returned false (no new data)\n";
}

void test_multiple_write_read_cycles() {
    std::cout << "--- Test: Multiple write/read cycles ---\n";
    TripleBuffer<VehicleState> buf;

    for (uint64_t i = 0; i < 100; ++i) {
        VehicleState s{};
        s.sequence = i;
        s.x = static_cast<double>(i);
        buf.write(s);

        VehicleState out;
        bool got_new = buf.read(out);
        assert(got_new);
        assert(out.sequence == i);
    }
    assert(buf.writes() == 100);
    assert(buf.reads() == 100);
    std::cout << "  PASS: 100 synchronized cycles\n";
}

void test_concurrent_multi_rate() {
    std::cout << "--- Test: Concurrent multi-rate ---\n";

    TripleBuffer<VehicleState> buf;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> last_read_seq{0};
    std::atomic<uint64_t> reader_count{0};
    std::atomic<bool> monotonic_ok{true};

    // Producer at ~1kHz
    std::thread producer([&]() {
        uint64_t seq = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            VehicleState s{};
            s.sequence = ++seq;
            s.x = static_cast<double>(seq);
            buf.write(s);
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    });

    // Consumer at ~200Hz
    std::thread consumer([&]() {
        uint64_t prev_seq = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            VehicleState out;
            if (buf.read(out)) {
                // Sequence must be monotonically increasing
                if (out.sequence < prev_seq) {
                    monotonic_ok.store(false, std::memory_order_relaxed);
                }
                prev_seq = out.sequence;
                reader_count.fetch_add(1, std::memory_order_relaxed);
                last_read_seq.store(out.sequence,
                                    std::memory_order_relaxed);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop.store(true);

    producer.join();
    consumer.join();

    assert(monotonic_ok.load());
    uint64_t reads = reader_count.load();
    uint64_t writes = buf.writes();
    assert(reads > 0);
    assert(writes > reads);  // Writer faster than reader → skips expected

    std::cout << "  PASS: " << writes << " writes, " << reads
              << " reads, monotonic sequence preserved\n";
}

void test_data_integrity() {
    std::cout << "--- Test: Data integrity under stress ---\n";

    struct BigData {
        std::array<uint64_t, 64> values{};
        uint64_t checksum = 0;
    };

    TripleBuffer<BigData> buf;
    std::atomic<bool> stop{false};
    std::atomic<bool> integrity_ok{true};

    // Writer fills all fields with same value, checksum = value * 64
    std::thread writer([&]() {
        uint64_t v = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            BigData d;
            d.values.fill(++v);
            d.checksum = v * 64;
            buf.write(d);
        }
    });

    // Reader verifies all fields match checksum
    std::thread reader([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            BigData d;
            if (buf.read(d)) {
                uint64_t expected = d.values[0];
                for (auto val : d.values) {
                    if (val != expected) {
                        integrity_ok.store(false);
                        return;
                    }
                }
                if (d.checksum != expected * 64) {
                    integrity_ok.store(false);
                    return;
                }
            }
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop.store(true);
    writer.join();
    reader.join();

    assert(integrity_ok.load());
    std::cout << "  PASS: No torn reads detected\n";
}

// ============================================================================
int main() {
    std::cout << "=== Lock-Free Triple Buffer Exercise ===\n\n";

    test_basic_write_read();
    test_latest_value_wins();
    test_no_new_data();
    test_multiple_write_read_cycles();
    test_concurrent_multi_rate();
    test_data_integrity();

    std::cout << "\n=== ALL TRIPLE BUFFER TESTS PASSED ===\n";
    return 0;
}
