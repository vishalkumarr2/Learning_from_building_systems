// ex01_shared_memory.cpp — Shared memory IPC with seqlock pattern
//
// Usage:
//   ./ex01 producer   # run first in one terminal
//   ./ex01 consumer   # run in another terminal
//
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -lrt ex01_shared_memory.cpp -o ex01

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

// ─── Shared data structure placed in shared memory ──────
static constexpr std::size_t kValueCount = 128;

struct SharedData {
    // Seqlock sequence: odd = write in progress, even = consistent
    alignas(64) std::atomic<uint64_t> sequence{0};

    // Timestamp (ns since epoch) written by producer
    alignas(64) std::atomic<int64_t> write_timestamp_ns{0};

    // Payload
    double values[kValueCount];

    // Padding to fill a reasonable page
    char _pad[4096 - sizeof(std::atomic<uint64_t>) * 2
              - sizeof(double) * kValueCount];
};

static constexpr const char* kShmName = "/week06_ex01_shm";
static constexpr std::size_t kShmSize = sizeof(SharedData);
static_assert(std::is_standard_layout_v<SharedData>);

// ─── Helpers ────────────────────────────────────────────
static int64_t now_ns() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

// ─── Producer ───────────────────────────────────────────
static void run_producer() {
    // Create shared memory
    int fd = shm_open(kShmName, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); std::exit(1); }

    if (ftruncate(fd, static_cast<off_t>(kShmSize)) < 0) {
        perror("ftruncate"); std::exit(1);
    }

    void* ptr = mmap(nullptr, kShmSize, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); std::exit(1); }
    close(fd);

    // Construct SharedData at the mapped address
    auto* data = new (ptr) SharedData{};

    std::printf("Producer: writing at 1 kHz for 3 seconds...\n");
    std::printf("  shm name: %s  size: %zu bytes\n", kShmName, kShmSize);

    const auto start = std::chrono::steady_clock::now();
    uint64_t msgs = 0;

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > std::chrono::seconds(3)) break;

        // Seqlock: increment to odd (write-in-progress)
        uint64_t seq = data->sequence.load(std::memory_order_relaxed);
        data->sequence.store(seq + 1, std::memory_order_release);

        // Write payload: sine wave indexed by message count
        for (std::size_t i = 0; i < kValueCount; ++i) {
            data->values[i] = std::sin(static_cast<double>(msgs * kValueCount + i) * 0.001);
        }

        // Stamp the write time
        data->write_timestamp_ns.store(now_ns(), std::memory_order_relaxed);

        // Seqlock: increment to even (write complete)
        data->sequence.store(seq + 2, std::memory_order_release);

        ++msgs;
        std::this_thread::sleep_for(std::chrono::microseconds(1000)); // ~1 kHz
    }

    std::printf("Producer: wrote %lu messages. Waiting for consumer to finish...\n", msgs);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    munmap(ptr, kShmSize);
    shm_unlink(kShmName);
    std::printf("Producer: cleaned up shared memory.\n");
}

// ─── Consumer ───────────────────────────────────────────
static void run_consumer() {
    // Open existing shared memory
    int fd = shm_open(kShmName, O_RDONLY, 0);
    if (fd < 0) {
        std::fprintf(stderr, "Consumer: shm_open failed — start the producer first.\n");
        std::exit(1);
    }

    void* ptr = mmap(nullptr, kShmSize, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); std::exit(1); }
    close(fd);

    const auto* data = reinterpret_cast<const SharedData*>(ptr);

    std::printf("Consumer: reading for 3 seconds...\n");

    const auto start = std::chrono::steady_clock::now();
    uint64_t reads = 0;
    uint64_t torn_reads = 0;
    int64_t total_latency_ns = 0;
    int64_t min_latency_ns = INT64_MAX;
    int64_t max_latency_ns = 0;

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > std::chrono::seconds(3)) break;

        // Seqlock read
        uint64_t s1{}, s2{};
        double local_values[kValueCount];
        int64_t write_ts{};
        int retries = 0;

        do {
            s1 = data->sequence.load(std::memory_order_acquire);

            // Copy payload while possibly inconsistent
            std::memcpy(local_values, data->values, sizeof(local_values));
            write_ts = data->write_timestamp_ns.load(std::memory_order_relaxed);

            s2 = data->sequence.load(std::memory_order_acquire);

            if (s1 != s2 || (s1 & 1)) {
                ++torn_reads;
                ++retries;
            }
        } while (s1 != s2 || (s1 & 1));

        // Measure latency if we have a valid timestamp
        if (write_ts > 0) {
            int64_t latency = now_ns() - write_ts;
            if (latency > 0 && latency < 1'000'000'000) { // sanity: < 1s
                total_latency_ns += latency;
                if (latency < min_latency_ns) min_latency_ns = latency;
                if (latency > max_latency_ns) max_latency_ns = latency;
            }
        }

        // Simple validation: check that values are in [-1, 1] (sine wave)
        bool valid = true;
        for (std::size_t i = 0; i < kValueCount; ++i) {
            if (local_values[i] < -1.01 || local_values[i] > 1.01) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            std::printf("Consumer: WARNING — invalid values detected at read %lu (seq %lu)\n",
                        reads, s1);
        }

        ++reads;

        // Don't spin at 100% CPU — small yield
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // ─── Report ─────────────────────────────────────────
    std::printf("\n=== Consumer Report ===\n");
    std::printf("  Total reads:     %lu\n", reads);
    std::printf("  Torn reads:      %lu (retried successfully)\n", torn_reads);
    if (reads > 0 && total_latency_ns > 0) {
        std::printf("  Avg latency:     %ld ns\n", total_latency_ns / static_cast<int64_t>(reads));
        std::printf("  Min latency:     %ld ns\n", min_latency_ns);
        std::printf("  Max latency:     %ld ns\n", max_latency_ns);
    }
    std::printf("  Seqlock consistent reads demonstrate zero-kernel-crossing IPC.\n");

    munmap(const_cast<void*>(ptr), kShmSize);
}

// ─── Standalone demo (single process, both roles) ───────
static void run_demo() {
    std::printf("=== Single-Process Demo (producer + consumer in threads) ===\n\n");

    int fd = shm_open(kShmName, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); std::exit(1); }
    ftruncate(fd, static_cast<off_t>(kShmSize));

    void* ptr = mmap(nullptr, kShmSize, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    close(fd);
    auto* data = new (ptr) SharedData{};

    std::atomic<bool> done{false};
    std::atomic<uint64_t> producer_count{0};

    // Producer thread
    auto producer = std::thread([&] {
        for (uint64_t i = 0; i < 5000; ++i) {
            uint64_t seq = data->sequence.load(std::memory_order_relaxed);
            data->sequence.store(seq + 1, std::memory_order_release);

            for (std::size_t j = 0; j < kValueCount; ++j)
                data->values[j] = std::sin(static_cast<double>(i * kValueCount + j) * 0.001);
            data->write_timestamp_ns.store(now_ns(), std::memory_order_relaxed);

            data->sequence.store(seq + 2, std::memory_order_release);
            ++producer_count;

            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        done.store(true, std::memory_order_release);
    });

    // Consumer thread
    uint64_t reads = 0, torn = 0;
    int64_t total_lat = 0, min_lat = INT64_MAX, max_lat = 0;

    auto consumer = std::thread([&] {
        while (!done.load(std::memory_order_acquire) || reads < producer_count.load(std::memory_order_acquire)) {
            uint64_t s1, s2;
            double local[kValueCount];
            int64_t wts;

            do {
                s1 = data->sequence.load(std::memory_order_acquire);
                std::memcpy(local, data->values, sizeof(local));
                wts = data->write_timestamp_ns.load(std::memory_order_relaxed);
                s2 = data->sequence.load(std::memory_order_acquire);
                if (s1 != s2 || (s1 & 1)) ++torn;
            } while (s1 != s2 || (s1 & 1));

            if (wts > 0) {
                int64_t lat = now_ns() - wts;
                if (lat > 0 && lat < 1'000'000'000) {
                    total_lat += lat;
                    if (lat < min_lat) min_lat = lat;
                    if (lat > max_lat) max_lat = lat;
                }
            }
            ++reads;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    producer.join();
    consumer.join();

    std::printf("Producer wrote: %lu messages\n", producer_count.load());
    std::printf("Consumer read:  %lu times  (torn retries: %lu)\n", reads, torn);
    if (reads > 0 && total_lat > 0) {
        std::printf("Latency — avg: %ld ns, min: %ld ns, max: %ld ns\n",
                    total_lat / static_cast<int64_t>(reads), min_lat, max_lat);
    }

    munmap(ptr, kShmSize);
    shm_unlink(kShmName);
    std::printf("\nDone. To test cross-process: ./ex01 producer & ./ex01 consumer\n");
}

// ─── Main ───────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        run_demo();
        return 0;
    }

    std::string mode = argv[1];
    if (mode == "producer") {
        run_producer();
    } else if (mode == "consumer") {
        run_consumer();
    } else {
        std::fprintf(stderr, "Usage: %s [producer|consumer]\n", argv[0]);
        std::fprintf(stderr, "  No argument = single-process demo\n");
        return 1;
    }
    return 0;
}
