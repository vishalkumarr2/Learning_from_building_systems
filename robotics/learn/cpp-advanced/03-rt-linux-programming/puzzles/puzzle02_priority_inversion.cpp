// Week 3 — Puzzle 02: Priority Inversion (The Mars Pathfinder Bug)
//
// Reproduce the classic priority inversion scenario:
//   LOW thread holds a mutex
//   MED thread busy-spins (preempts LOW, doesn't need mutex)
//   HIGH thread blocks on mutex — can't run because MED preempts LOW
//
// Then fix it with PTHREAD_PRIO_INHERIT.
//
// Build:  cmake --build .
// Run:    sudo ./puzzle02_priority_inversion   (needs SCHED_FIFO)
//   or:   sudo setcap cap_sys_nice+ep ./puzzle02_priority_inversion

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using ms    = std::chrono::milliseconds;

// ─── Timeline recorder ──────────────────────────────────────────────
struct Event {
    int64_t     time_us;
    const char* thread;
    const char* action;
};

constexpr int MAX_EVENTS = 256;
Event        g_events[MAX_EVENTS];
std::atomic<int> g_event_idx{0};

int64_t g_start_us = 0;

void record(const char* thread, const char* action) {
    auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now().time_since_epoch()).count();
    int idx = g_event_idx.fetch_add(1, std::memory_order_relaxed);
    if (idx < MAX_EVENTS) {
        g_events[idx] = {now - g_start_us, thread, action};
    }
}

void print_timeline() {
    int n = g_event_idx.load(std::memory_order_relaxed);
    if (n > MAX_EVENTS) n = MAX_EVENTS;

    std::printf("\n  Timeline:\n");
    std::printf("  %10s  %-6s  %s\n", "Time(μs)", "Thread", "Event");
    std::printf("  %10s  %-6s  %s\n", "────────", "──────", "─────");
    for (int i = 0; i < n; ++i) {
        std::printf("  %10ld  %-6s  %s\n",
                    g_events[i].time_us,
                    g_events[i].thread,
                    g_events[i].action);
    }
}

// ─── Busy spin helper ────────────────────────────────────────────────
static volatile double spin_sink;

void busy_spin_ms(int ms_val) {
    auto end = Clock::now() + ms(ms_val);
    double x = 1.0;
    while (Clock::now() < end) {
        x = x * 1.0001 + 0.0001;
    }
    spin_sink = x;
}

// ─── Set thread RT priority ──────────────────────────────────────────
bool set_fifo(int priority) {
    struct sched_param param{};
    param.sched_priority = priority;
    int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    if (rc != 0) {
        std::fprintf(stderr, "pthread_setschedparam(FIFO, %d): %s\n",
                     priority, std::strerror(rc));
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Scenario 1: WITHOUT Priority Inheritance (shows inversion)
// ═══════════════════════════════════════════════════════════════════════

void run_without_pi() {
    std::printf("\n=== WITHOUT Priority Inheritance ===\n");
    std::printf("  Expected: HIGH blocked by MED (priority inversion)\n\n");

    g_event_idx.store(0, std::memory_order_relaxed);
    g_start_us = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now().time_since_epoch()).count();

    // Normal mutex (no PI)
    pthread_mutex_t mutex;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    // Explicitly NOT setting PTHREAD_PRIO_INHERIT
    pthread_mutex_init(&mutex, &attr);
    pthread_mutexattr_destroy(&attr);

    std::atomic<bool> low_has_lock{false};
    std::atomic<bool> done{false};

    // LOW priority thread: grabs mutex, holds it for 50ms
    pthread_t low_thread;
    auto low_fn = [](void* arg) -> void* {
        auto* ctx = static_cast<std::tuple<pthread_mutex_t*, std::atomic<bool>*>*>(arg);
        auto [mtx, has_lock] = *ctx;

        set_fifo(10);  // LOW priority
        record("LOW", "started");

        pthread_mutex_lock(mtx);
        record("LOW", "acquired mutex");
        has_lock->store(true, std::memory_order_release);

        // Hold mutex while doing work
        busy_spin_ms(50);

        record("LOW", "releasing mutex");
        pthread_mutex_unlock(mtx);
        record("LOW", "done");

        return nullptr;
    };

    auto low_ctx = std::make_tuple(&mutex, &low_has_lock);
    pthread_create(&low_thread, nullptr, low_fn, &low_ctx);

    // Wait for LOW to grab the mutex
    while (!low_has_lock.load(std::memory_order_acquire)) {
        usleep(100);
    }
    usleep(1000);  // Let LOW start its work

    // MED priority thread: busy-spins, doesn't need mutex
    pthread_t med_thread;
    auto med_fn = [](void* /*arg*/) -> void* {
        set_fifo(50);  // MED priority — preempts LOW
        record("MED", "started (preempts LOW!)");

        // Busy spin — prevents LOW from running!
        busy_spin_ms(40);

        record("MED", "done");
        return nullptr;
    };

    pthread_create(&med_thread, nullptr, med_fn, &done);

    usleep(1000);  // Let MED start

    // HIGH priority thread: needs the mutex
    pthread_t high_thread;
    auto high_fn = [](void* arg) -> void* {
        auto* mtx = static_cast<pthread_mutex_t*>(arg);
        set_fifo(90);  // HIGH priority
        record("HIGH", "started, trying to lock mutex...");

        auto t0 = Clock::now();
        pthread_mutex_lock(mtx);
        auto t1 = Clock::now();
        auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        char buf[128];
        std::snprintf(buf, sizeof(buf), "acquired mutex (waited %ldμs!)", wait_us);
        record("HIGH", buf);

        pthread_mutex_unlock(mtx);
        record("HIGH", "done");
        return nullptr;
    };

    pthread_create(&high_thread, nullptr, high_fn, &mutex);

    pthread_join(high_thread, nullptr);
    pthread_join(med_thread, nullptr);
    pthread_join(low_thread, nullptr);
    pthread_mutex_destroy(&mutex);

    print_timeline();
    std::printf("\n  ↑ HIGH was blocked unnecessarily by MED!\n");
    std::printf("  MED preempted LOW, preventing LOW from releasing mutex.\n");
}

// ═══════════════════════════════════════════════════════════════════════
// Scenario 2: WITH Priority Inheritance (fixes inversion)
// ═══════════════════════════════════════════════════════════════════════

void run_with_pi() {
    std::printf("\n\n=== WITH Priority Inheritance (PTHREAD_PRIO_INHERIT) ===\n");
    std::printf("  Expected: LOW boosted to HIGH's priority, MED can't preempt\n\n");

    g_event_idx.store(0, std::memory_order_relaxed);
    g_start_us = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now().time_since_epoch()).count();

    // PI mutex
    pthread_mutex_t mutex;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);  // ← THE FIX
    pthread_mutex_init(&mutex, &attr);
    pthread_mutexattr_destroy(&attr);

    std::atomic<bool> low_has_lock{false};
    std::atomic<bool> done{false};

    // Same thread functions as above, reusing the pattern
    pthread_t low_thread;
    auto low_fn = [](void* arg) -> void* {
        auto* ctx = static_cast<std::tuple<pthread_mutex_t*, std::atomic<bool>*>*>(arg);
        auto [mtx, has_lock] = *ctx;

        set_fifo(10);
        record("LOW", "started");

        pthread_mutex_lock(mtx);
        record("LOW", "acquired mutex");
        has_lock->store(true, std::memory_order_release);

        busy_spin_ms(50);

        record("LOW", "releasing mutex");
        pthread_mutex_unlock(mtx);
        record("LOW", "done");
        return nullptr;
    };

    auto low_ctx = std::make_tuple(&mutex, &low_has_lock);
    pthread_create(&low_thread, nullptr, low_fn, &low_ctx);

    while (!low_has_lock.load(std::memory_order_acquire)) {
        usleep(100);
    }
    usleep(1000);

    pthread_t med_thread;
    auto med_fn = [](void* /*arg*/) -> void* {
        set_fifo(50);
        record("MED", "started");

        busy_spin_ms(40);

        record("MED", "done");
        return nullptr;
    };

    pthread_create(&med_thread, nullptr, med_fn, &done);
    usleep(1000);

    pthread_t high_thread;
    auto high_fn = [](void* arg) -> void* {
        auto* mtx = static_cast<pthread_mutex_t*>(arg);
        set_fifo(90);
        record("HIGH", "started, trying to lock mutex...");

        auto t0 = Clock::now();
        pthread_mutex_lock(mtx);
        auto t1 = Clock::now();
        auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        char buf[128];
        std::snprintf(buf, sizeof(buf), "acquired mutex (waited %ldμs)", wait_us);
        record("HIGH", buf);

        pthread_mutex_unlock(mtx);
        record("HIGH", "done");
        return nullptr;
    };

    pthread_create(&high_thread, nullptr, high_fn, &mutex);

    pthread_join(high_thread, nullptr);
    pthread_join(med_thread, nullptr);
    pthread_join(low_thread, nullptr);
    pthread_mutex_destroy(&mutex);

    print_timeline();
    std::printf("\n  ↑ LOW was boosted to HIGH's priority while holding mutex.\n");
    std::printf("  MED couldn't preempt boosted LOW. HIGH got mutex faster.\n");
}

// ═══════════════════════════════════════════════════════════════════════
int main() {
    std::printf("╔════════════════════════════════════════════════════╗\n");
    std::printf("║  Puzzle 02: Priority Inversion (Mars Pathfinder)  ║\n");
    std::printf("╚════════════════════════════════════════════════════╝\n");

    // Check if we can use RT scheduling
    struct sched_param test_param{};
    test_param.sched_priority = 10;
    if (sched_setscheduler(0, SCHED_FIFO, &test_param) != 0) {
        std::printf("\n⚠ Cannot set SCHED_FIFO: %s\n", std::strerror(errno));
        std::printf("  Run as root or with CAP_SYS_NICE to see priority inversion.\n");
        std::printf("  sudo ./puzzle02_priority_inversion\n");
        return 1;
    }
    // Reset back to normal for main thread
    test_param.sched_priority = 0;
    sched_setscheduler(0, SCHED_OTHER, &test_param);

    run_without_pi();
    run_with_pi();

    std::printf("\n─── The Mars Pathfinder Story ───\n");
    std::printf("On July 4, 1997, the Pathfinder lander repeatedly reset.\n");
    std::printf("A low-priority thread held a mutex needed by a high-priority\n");
    std::printf("bus manager thread. A medium-priority comms thread preempted\n");
    std::printf("the low thread, starving the high thread, triggering watchdog.\n");
    std::printf("Fix: enable priority inheritance (done via C command uplink).\n");

    return 0;
}
