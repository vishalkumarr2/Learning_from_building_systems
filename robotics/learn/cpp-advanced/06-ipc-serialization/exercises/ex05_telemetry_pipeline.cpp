// ex05_telemetry_pipeline.cpp — Week 6 mini-project: two-thread telemetry pipeline
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -lrt -ldl ex05_telemetry_pipeline.cpp -o ex05
//
// Architecture:
//   RT thread (producer) ---> SPSC ring buffer ---> Logger thread (consumer) ---> binary file
//
// In production, use shared memory between separate processes so the logger
// crash doesn't take down the RT controller. The SPSC queue would live in
// a mmap'd region, and the consumer would be a separate process.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// ---------- Telemetry record — fixed-size, trivially copyable ----------

struct TelemetryRecord {
    uint64_t timestamp_ns;
    uint32_t sensor_id;
    float values[6];  // accel_x/y/z, gyro_x/y/z
};

static_assert(std::is_trivially_copyable_v<TelemetryRecord>,
              "TelemetryRecord must be trivially copyable for binary I/O and shared memory");

// ---------- Lock-free SPSC ring buffer (power-of-2 capacity) ----------
// Single-Producer Single-Consumer — no locks needed, just atomic head/tail.
// Producer writes at head, consumer reads at tail.
// Capacity must be power of 2 so we can use bitwise AND instead of modulo.

template <typename T, std::size_t N>
class SPSCQueue {
    static_assert((N & (N - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    alignas(64) std::atomic<std::size_t> head_{0};  // written by producer
    alignas(64) std::atomic<std::size_t> tail_{0};  // written by consumer
    T buffer_[N];

    static constexpr std::size_t kMask = N - 1;

public:
    // Returns false if full (producer side)
    bool try_push(const T& item) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t next_h = (h + 1) & kMask;

        // If next_head == tail, the buffer is full
        if (next_h == tail_.load(std::memory_order_acquire)) {
            return false;  // full — drop or back-pressure
        }

        buffer_[h] = item;
        head_.store(next_h, std::memory_order_release);
        return true;
    }

    // Returns false if empty (consumer side)
    bool try_pop(T& item) {
        const std::size_t t = tail_.load(std::memory_order_relaxed);

        // If tail == head, the buffer is empty
        if (t == head_.load(std::memory_order_acquire)) {
            return false;  // empty
        }

        item = buffer_[t];
        tail_.store((t + 1) & kMask, std::memory_order_release);
        return true;
    }

    std::size_t capacity() const { return N - 1; }  // usable slots = N-1
};

// ---------- Configuration ----------

static constexpr std::size_t QUEUE_SIZE = 8192;       // power of 2
static constexpr int SAMPLE_RATE_HZ = 1000;
static constexpr int DURATION_SEC = 3;
static constexpr int TOTAL_SAMPLES = SAMPLE_RATE_HZ * DURATION_SEC;
static constexpr const char* OUTPUT_FILE = "/tmp/telemetry_pipeline.bin";

// ---------- Clock utility ----------

static uint64_t now_ns() {
    auto tp = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count());
}

// ---------- Producer: simulated 1kHz IMU ----------

static void producer_thread(SPSCQueue<TelemetryRecord, QUEUE_SIZE>& queue,
                            std::atomic<bool>& done,
                            std::atomic<uint64_t>& dropped) {
    const auto period = std::chrono::microseconds(1000000 / SAMPLE_RATE_HZ);
    auto next_wake = std::chrono::steady_clock::now();

    for (int i = 0; i < TOTAL_SAMPLES; ++i) {
        double t = static_cast<double>(i) / SAMPLE_RATE_HZ;

        TelemetryRecord rec{};
        rec.timestamp_ns = now_ns();
        rec.sensor_id = 1;

        // Simulated IMU: sin waves at different frequencies + small noise
        rec.values[0] = static_cast<float>(std::sin(2.0 * M_PI * 1.0 * t) + 0.01 * (i % 7));   // accel_x
        rec.values[1] = static_cast<float>(std::cos(2.0 * M_PI * 0.5 * t) + 0.01 * (i % 11));   // accel_y
        rec.values[2] = static_cast<float>(9.81 + 0.05 * std::sin(2.0 * M_PI * 0.1 * t));        // accel_z
        rec.values[3] = static_cast<float>(0.1 * std::sin(2.0 * M_PI * 2.0 * t));                // gyro_x
        rec.values[4] = static_cast<float>(0.2 * std::cos(2.0 * M_PI * 1.5 * t));                // gyro_y
        rec.values[5] = static_cast<float>(0.05 * std::sin(2.0 * M_PI * 3.0 * t));               // gyro_z

        if (!queue.try_push(rec)) {
            dropped.fetch_add(1, std::memory_order_relaxed);
        }

        next_wake += period;
        std::this_thread::sleep_until(next_wake);
    }

    done.store(true, std::memory_order_release);
}

// ---------- Consumer: logger thread ----------

static void consumer_thread(SPSCQueue<TelemetryRecord, QUEUE_SIZE>& queue,
                            std::atomic<bool>& done,
                            uint64_t& records_written,
                            uint64_t& max_latency_ns,
                            uint64_t& total_latency_ns) {
    FILE* fp = std::fopen(OUTPUT_FILE, "wb");
    if (!fp) {
        std::perror("fopen");
        return;
    }

    TelemetryRecord rec{};
    records_written = 0;
    max_latency_ns = 0;
    total_latency_ns = 0;

    while (true) {
        if (queue.try_pop(rec)) {
            // Measure queue latency (time from push to pop)
            uint64_t lat = now_ns() - rec.timestamp_ns;
            total_latency_ns += lat;
            if (lat > max_latency_ns) {
                max_latency_ns = lat;
            }

            std::fwrite(&rec, sizeof(rec), 1, fp);
            ++records_written;
        } else if (done.load(std::memory_order_acquire)) {
            // Drain remaining after producer signals done
            while (queue.try_pop(rec)) {
                uint64_t lat = now_ns() - rec.timestamp_ns;
                total_latency_ns += lat;
                if (lat > max_latency_ns) max_latency_ns = lat;
                std::fwrite(&rec, sizeof(rec), 1, fp);
                ++records_written;
            }
            break;
        } else {
            // Spin-wait with yield (in production: use futex or eventfd)
            std::this_thread::yield();
        }
    }

    std::fclose(fp);
}

// ---------- Verification: read back and check ----------

static bool verify_output(uint64_t expected_records) {
    FILE* fp = std::fopen(OUTPUT_FILE, "rb");
    if (!fp) {
        std::perror("fopen for verify");
        return false;
    }

    // Get file size
    std::fseek(fp, 0, SEEK_END);
    long file_size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);

    std::size_t record_count = static_cast<std::size_t>(file_size) / sizeof(TelemetryRecord);
    bool size_ok = (static_cast<std::size_t>(file_size) % sizeof(TelemetryRecord)) == 0;

    std::printf("\n--- Verification ---\n");
    std::printf("  File: %s\n", OUTPUT_FILE);
    std::printf("  File size: %ld bytes\n", file_size);
    std::printf("  Record size: %zu bytes\n", sizeof(TelemetryRecord));
    std::printf("  Records in file: %zu\n", record_count);
    std::printf("  Expected: %lu\n", expected_records);
    std::printf("  Aligned: %s\n", size_ok ? "YES" : "NO (CORRUPT!)");

    // Read a few records and check monotonic timestamps
    bool monotonic = true;
    uint64_t prev_ts = 0;
    TelemetryRecord rec{};
    std::size_t checked = 0;

    while (std::fread(&rec, sizeof(rec), 1, fp) == 1) {
        if (rec.timestamp_ns < prev_ts) {
            monotonic = false;
        }
        prev_ts = rec.timestamp_ns;
        ++checked;

        // Sanity check: accel_z should be near 9.81
        if (rec.values[2] < 9.0f || rec.values[2] > 10.5f) {
            std::printf("  WARNING: accel_z=%.3f out of range at record %zu\n",
                        rec.values[2], checked);
        }
    }

    std::printf("  Timestamps monotonic: %s\n", monotonic ? "YES" : "NO");
    std::printf("  All records readable: %s\n",
                checked == record_count ? "YES" : "NO");

    std::fclose(fp);
    return size_ok && monotonic && (record_count == expected_records);
}

// ---------- main ----------

int main() {
    std::printf("=== Telemetry Pipeline ===\n");
    std::printf("  Sample rate: %d Hz\n", SAMPLE_RATE_HZ);
    std::printf("  Duration: %d s\n", DURATION_SEC);
    std::printf("  Expected samples: %d\n", TOTAL_SAMPLES);
    std::printf("  Queue capacity: %zu slots\n", QUEUE_SIZE - 1);
    std::printf("  Record size: %zu bytes\n\n", sizeof(TelemetryRecord));

    SPSCQueue<TelemetryRecord, QUEUE_SIZE> queue;
    std::atomic<bool> done{false};
    std::atomic<uint64_t> dropped{0};
    uint64_t records_written = 0;
    uint64_t max_latency_ns = 0;
    uint64_t total_latency_ns = 0;

    auto t_start = std::chrono::steady_clock::now();

    std::thread consumer(consumer_thread, std::ref(queue), std::ref(done),
                         std::ref(records_written), std::ref(max_latency_ns),
                         std::ref(total_latency_ns));
    std::thread producer(producer_thread, std::ref(queue), std::ref(done),
                         std::ref(dropped));

    producer.join();
    consumer.join();

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double throughput_mbs = (records_written * sizeof(TelemetryRecord)) / (elapsed_s * 1024.0 * 1024.0);
    double avg_latency_ns = records_written > 0
        ? static_cast<double>(total_latency_ns) / records_written : 0.0;

    std::printf("--- Results ---\n");
    std::printf("  Records written: %lu / %d\n", records_written, TOTAL_SAMPLES);
    std::printf("  Dropped (queue full): %lu\n", dropped.load());
    std::printf("  Elapsed: %.3f s\n", elapsed_s);
    std::printf("  Throughput: %.2f MB/s\n", throughput_mbs);
    std::printf("  Avg queue latency: %.0f ns\n", avg_latency_ns);
    std::printf("  Max queue latency: %lu ns (%.3f ms)\n",
                max_latency_ns, max_latency_ns / 1e6);

    bool ok = verify_output(records_written);
    std::printf("\n%s\n", ok ? "PASS — all checks passed" : "FAIL — verification errors");

    // Cleanup
    std::remove(OUTPUT_FILE);

    // In production, use shared memory between separate processes:
    //   Producer (RT controller): writes TelemetryRecord into shm ring buffer
    //   Consumer (logger process): reads from shm, writes to disk
    //   Benefit: logger crash doesn't affect RT process

    return ok ? 0 : 1;
}
