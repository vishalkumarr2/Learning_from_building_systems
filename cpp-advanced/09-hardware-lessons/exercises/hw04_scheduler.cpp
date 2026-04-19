// hw04_scheduler.cpp — Linux Scheduler Deep Dive
// Compile: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread hw04_scheduler.cpp -o hw04_scheduler
//
// Note: SCHED_FIFO requires root or CAP_SYS_NICE. Gracefully falls back.
//
// Exercises:
//  1. Context switch cost via pipe ping-pong
//  2. SCHED_FIFO attempt
//  3. Scheduling latency comparison
//  4. Timer resolution measurement
//  5. Priority inversion reproduction
//  6. Priority inheritance fix

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <time.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// 1. Context switch cost: pipe ping-pong between two threads
// ---------------------------------------------------------------------------
static double measure_context_switch_ns() {
    std::printf("── 1. Context Switch Cost (pipe ping-pong) ──\n");

    int pipe_ab[2], pipe_ba[2];
    if (pipe(pipe_ab) != 0 || pipe(pipe_ba) != 0) {
        std::perror("pipe");
        return -1;
    }

    constexpr int kRounds = 100'000;
    char buf = 'x';

    auto t0 = Clock::now();

    std::thread thread_a([&]() {
        for (int i = 0; i < kRounds; ++i) {
            [[maybe_unused]] auto w = write(pipe_ab[1], &buf, 1);
            [[maybe_unused]] auto r = read(pipe_ba[0], &buf, 1);
        }
    });

    std::thread thread_b([&]() {
        for (int i = 0; i < kRounds; ++i) {
            [[maybe_unused]] auto r = read(pipe_ab[0], &buf, 1);
            [[maybe_unused]] auto w = write(pipe_ba[1], &buf, 1);
        }
    });

    thread_a.join();
    thread_b.join();

    auto t1 = Clock::now();
    double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    // Each round = 2 context switches (A→B and B→A)
    double per_switch_ns = total_ns / (kRounds * 2.0);

    close(pipe_ab[0]); close(pipe_ab[1]);
    close(pipe_ba[0]); close(pipe_ba[1]);

    std::printf("  Rounds: %d, Total: %.1f ms\n", kRounds, total_ns / 1e6);
    std::printf("  Context switch cost: %.0f ns (%.2f µs)\n\n", per_switch_ns, per_switch_ns / 1000);

    return per_switch_ns;
}

// ---------------------------------------------------------------------------
// 2. Attempt SCHED_FIFO
// ---------------------------------------------------------------------------
static bool try_sched_fifo(int priority = 1) {
    std::printf("── 2. SCHED_FIFO Attempt ──\n");

    struct sched_param param{};
    param.sched_priority = priority;

    int ret = sched_setscheduler(0, SCHED_FIFO, &param);
    if (ret == 0) {
        std::printf("  ✓ SCHED_FIFO set successfully (priority %d)\n", priority);
        // Restore SCHED_OTHER
        param.sched_priority = 0;
        sched_setscheduler(0, SCHED_OTHER, &param);
        std::printf("  Restored SCHED_OTHER\n\n");
        return true;
    } else {
        std::printf("  ✗ Cannot set SCHED_FIFO: %s\n", std::strerror(errno));
        std::printf("  (Run as root or with CAP_SYS_NICE to enable)\n\n");
        return false;
    }
}

// ---------------------------------------------------------------------------
// 3. Scheduling latency: measure jitter of timed wakeups
// ---------------------------------------------------------------------------
static void measure_scheduling_latency(int policy, const char* name) {
    struct sched_param param{};
    if (policy == SCHED_FIFO) {
        param.sched_priority = 1;
        if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
            std::printf("  [%s] Cannot set policy, skipping.\n", name);
            return;
        }
    }

    constexpr int kSamples = 10'000;
    constexpr int64_t kSleepNs = 1'000; // 1 µs target

    int64_t max_jitter = 0;
    int64_t total_jitter = 0;

    struct timespec req{};
    req.tv_sec = 0;
    req.tv_nsec = kSleepNs;

    for (int i = 0; i < kSamples; ++i) {
        auto t0 = Clock::now();
        clock_nanosleep(CLOCK_MONOTONIC, 0, &req, nullptr);
        auto t1 = Clock::now();
        int64_t actual_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        int64_t jitter = actual_ns - kSleepNs;
        if (jitter < 0) jitter = 0;
        total_jitter += jitter;
        if (jitter > max_jitter) max_jitter = jitter;
    }

    double avg_jitter = static_cast<double>(total_jitter) / kSamples;

    std::printf("  [%s] sleep=1µs, samples=%d\n", name, kSamples);
    std::printf("    avg jitter: %.0f ns, max jitter: %ld ns (%.1f µs)\n",
                avg_jitter, max_jitter, max_jitter / 1000.0);

    // Restore SCHED_OTHER
    if (policy == SCHED_FIFO) {
        param.sched_priority = 0;
        sched_setscheduler(0, SCHED_OTHER, &param);
    }
}

static void run_latency_comparison(bool have_fifo) {
    std::printf("── 3. Scheduling Latency Comparison ──\n");
    measure_scheduling_latency(SCHED_OTHER, "SCHED_OTHER");
    if (have_fifo) {
        measure_scheduling_latency(SCHED_FIFO, "SCHED_FIFO");
    }
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// 4. Timer resolution measurement
// ---------------------------------------------------------------------------
static void measure_timer_resolution() {
    std::printf("── 4. Timer Resolution (clock_nanosleep actual) ──\n");

    constexpr int kSamples = 1'000;
    int64_t min_sleep = INT64_MAX;
    int64_t max_sleep = 0;
    int64_t total = 0;

    struct timespec req{};
    req.tv_sec = 0;
    req.tv_nsec = 1'000; // request 1 µs

    for (int i = 0; i < kSamples; ++i) {
        auto t0 = Clock::now();
        clock_nanosleep(CLOCK_MONOTONIC, 0, &req, nullptr);
        auto t1 = Clock::now();
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        total += ns;
        if (ns < min_sleep) min_sleep = ns;
        if (ns > max_sleep) max_sleep = ns;
    }

    double avg = static_cast<double>(total) / kSamples;
    std::printf("  Requested: 1 µs, Samples: %d\n", kSamples);
    std::printf("  Min: %ld ns, Avg: %.0f ns, Max: %ld ns\n", min_sleep, avg, max_sleep);
    std::printf("  Effective resolution: ~%.1f µs\n\n", avg / 1000.0);
}

// ---------------------------------------------------------------------------
// 5&6. Priority Inversion / Priority Inheritance
// ---------------------------------------------------------------------------

// Shared state for priority inversion demo
struct PrioInvState {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    std::atomic<bool> low_holding{false};
    std::atomic<bool> done{false};
    std::atomic<int64_t> high_wait_ns{0};
};

static void* low_priority_thread(void* arg) {
    auto* state = static_cast<PrioInvState*>(arg);
    pthread_mutex_lock(&state->mutex);
    state->low_holding.store(true);

    // Low-priority holds mutex and does "work"
    // Busy-wait to simulate work while holding the lock
    auto t0 = Clock::now();
    while (std::chrono::duration<double, std::milli>(Clock::now() - t0).count() < 50) {
        // simulate work
    }

    pthread_mutex_unlock(&state->mutex);
    return nullptr;
}

static void* med_priority_thread(void* arg) {
    auto* state = static_cast<PrioInvState*>(arg);

    // Wait until low has the mutex
    while (!state->low_holding.load()) {
        std::this_thread::yield();
    }

    // Medium priority busy-spins, preventing low from running (on single core)
    // and thus blocking high indirectly
    auto t0 = Clock::now();
    while (!state->done.load()) {
        // busy spin — this is the priority inversion!
        if (std::chrono::duration<double, std::milli>(Clock::now() - t0).count() > 200) {
            break; // safety timeout
        }
    }
    return nullptr;
}

static void* high_priority_thread(void* arg) {
    auto* state = static_cast<PrioInvState*>(arg);

    // Wait until low has the mutex
    while (!state->low_holding.load()) {
        std::this_thread::yield();
    }

    // Small delay to let medium start spinning
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    auto t0 = Clock::now();
    pthread_mutex_lock(&state->mutex);  // blocked! Low holds it, med prevents low from running
    auto t1 = Clock::now();

    state->high_wait_ns.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    pthread_mutex_unlock(&state->mutex);
    state->done.store(true);
    return nullptr;
}

static void run_priority_inversion(bool use_inherit) {
    const char* label = use_inherit ? "WITH Priority Inheritance" : "WITHOUT Priority Inheritance";
    std::printf("  [%s]\n", label);

    PrioInvState state{};

    // Set up mutex
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    if (use_inherit) {
        pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
    }
    pthread_mutex_init(&state.mutex, &attr);
    pthread_mutexattr_destroy(&attr);

    pthread_t t_low, t_med, t_high;
    pthread_attr_t tattr;

    auto create_rt_thread = [](pthread_t* t, void*(*fn)(void*), void* arg,
                               int priority, pthread_attr_t* attr) -> bool {
        pthread_attr_init(attr);
        struct sched_param sp{};
        sp.sched_priority = priority;

        // Try real-time scheduling
        if (pthread_attr_setschedpolicy(attr, SCHED_FIFO) == 0 &&
            pthread_attr_setschedparam(attr, &sp) == 0 &&
            pthread_attr_setinheritsched(attr, PTHREAD_EXPLICIT_SCHED) == 0) {
            if (pthread_create(t, attr, fn, arg) == 0) {
                return true;
            }
        }

        // Fall back to default scheduling
        pthread_attr_init(attr);
        pthread_create(t, nullptr, fn, arg);
        return false;
    };

    bool rt_ok = create_rt_thread(&t_low, low_priority_thread, &state, 10, &tattr);
    create_rt_thread(&t_med, med_priority_thread, &state, 50, &tattr);
    create_rt_thread(&t_high, high_priority_thread, &state, 90, &tattr);

    pthread_join(t_high, nullptr);
    pthread_join(t_med, nullptr);
    pthread_join(t_low, nullptr);

    int64_t wait = state.high_wait_ns.load();
    std::printf("    High-priority thread waited: %ld ns (%.2f ms)\n",
                wait, wait / 1e6);
    if (!rt_ok) {
        std::printf("    (Note: running without RT privileges, results are illustrative)\n");
    }

    pthread_mutex_destroy(&state.mutex);
}

static void priority_inversion_demo() {
    std::printf("── 5&6. Priority Inversion & Inheritance ──\n");
    std::printf("  3 threads: LOW(10) holds mutex, MED(50) busy-spins, HIGH(90) wants mutex\n\n");

    run_priority_inversion(false);
    std::printf("\n");
    run_priority_inversion(true);
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║       HW04 — Linux Scheduler Deep Dive              ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    double ctx_switch_ns = measure_context_switch_ns();
    bool have_fifo = try_sched_fifo();
    run_latency_comparison(have_fifo);
    measure_timer_resolution();
    priority_inversion_demo();

    // --- Summary ---
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  Results Summary                                             ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════╣\n");
    std::printf("║  Context switch cost     : %8.0f ns (%5.2f µs)           ║\n",
                ctx_switch_ns, ctx_switch_ns / 1000);
    std::printf("║  SCHED_FIFO available    : %s                              ║\n",
                have_fifo ? "yes" : "no ");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
