// ============================================================================
// puzzle02_asan_vs_tsan.cpp — Passes ASan, Fails TSan
//
// This program has a DATA RACE on a plain (non-atomic) integer. Two threads
// read/write the same variable without synchronization.
//
// - ASan (AddressSanitizer) does NOT detect this because there is no
//   heap corruption, buffer overflow, or use-after-free. ASan instruments
//   memory ACCESS BOUNDS, not synchronization.
//
// - TSan (ThreadSanitizer) WILL detect this because it tracks
//   happens-before relationships. Two unsynchronized accesses to the same
//   memory location where at least one is a write = data race = UB.
//
// YOUR TASK:
//   1. Build with ASan and verify it passes:
//        g++ -std=c++20 -O1 -g -fsanitize=address puzzle02_asan_vs_tsan.cpp \
//            -pthread -o puzzle02_asan
//        ./puzzle02_asan    # no errors
//
//   2. Build with TSan and verify it fails:
//        g++ -std=c++20 -O1 -g -fsanitize=thread puzzle02_asan_vs_tsan.cpp \
//            -pthread -o puzzle02_tsan
//        ./puzzle02_tsan    # TSan reports data race
//
//   3. Fix the race (use std::atomic<int> or a mutex).
//
//   4. Explain: Why does the program "work" despite the UB?
//      (On most x86 hardware, aligned int reads/writes are atomic at the
//       machine level — but the C++ memory model says it's UB, and the
//       compiler is free to optimize based on that assumption.)
//
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Shared state — NOT atomic, NOT protected by mutex
// This is "just an int" — no heap corruption, so ASan is blind to it.
// ---------------------------------------------------------------------------
static int g_counter = 0;        // DATA RACE: unsynchronized read/write
static int g_snapshot = 0;       // DATA RACE: read in main, written in thread

// Use volatile to prevent the compiler from optimizing the loop away,
// but volatile does NOT provide synchronization (common misconception).
static volatile bool g_running = true;

// ---------------------------------------------------------------------------
// Writer thread: Increment counter in a tight loop
// ---------------------------------------------------------------------------
static void writer_thread(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        g_counter += 1;           // WRITE without synchronization
    }
    g_snapshot = g_counter;       // Another unsynchronized WRITE
}

// ---------------------------------------------------------------------------
// Reader thread: Read counter periodically
// ---------------------------------------------------------------------------
static void reader_thread(std::vector<int>& samples) {
    while (g_running) {
        int val = g_counter;      // READ without synchronization
        samples.push_back(val);

        // Small delay to avoid spinning too fast
        for (volatile int i = 0; i < 100; ++i) {}
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("=== puzzle02: ASan vs TSan ===\n");
    printf("This program has a data race on g_counter.\n");
    printf("ASan won't catch it. TSan will.\n\n");

    constexpr int kIterations = 1'000'000;

    std::vector<int> samples;
    samples.reserve(10'000);

    std::thread reader(reader_thread, std::ref(samples));
    std::thread writer(writer_thread, kIterations);

    writer.join();
    g_running = false;
    reader.join();

    printf("Final counter: %d (expected: %d)\n", g_counter, kIterations);
    printf("Snapshot:      %d\n", g_snapshot);
    printf("Samples taken: %zu\n", samples.size());

    // On x86, this usually "works" — counter == expected.
    // But it's undefined behavior. On ARM (weaker memory model) or with
    // aggressive optimizations, it could produce wrong results.

    if (g_counter == kIterations) {
        printf("\nResult looks correct — but it's still UB!\n");
        printf("The compiler could legally:\n");
        printf("  - Cache g_counter in a register (reader sees stale value)\n");
        printf("  - Reorder the writes (snapshot before final counter)\n");
        printf("  - Combine multiple increments into one store\n");
    } else {
        printf("\nResult is WRONG — UB manifested!\n");
        printf("Lost updates: %d\n", kIterations - g_counter);
    }

    printf("\nFix: Change 'static int g_counter' to 'static std::atomic<int> g_counter'\n");
    printf("     and 'static int g_snapshot' to 'static std::atomic<int> g_snapshot'.\n");

    return 0;
}

/*
 * FIXED VERSION:
 *
 * #include <atomic>
 *
 * static std::atomic<int> g_counter{0};
 * static std::atomic<int> g_snapshot{0};
 * static std::atomic<bool> g_running{true};
 *
 * // writer:
 *   g_counter.fetch_add(1, std::memory_order_relaxed);
 *
 * // reader:
 *   int val = g_counter.load(std::memory_order_relaxed);
 *
 * // For this use case, relaxed ordering is sufficient because we don't
 * // need inter-variable ordering guarantees — just atomicity.
 */
