// Module 17: SDV & Mission-Critical Systems
// Puzzle 02: Watchdog Starvation Race
//
// This watchdog implementation has a race condition that causes
// intermittent "phantom resets" in production. The watchdog fires
// even though the application IS alive — it just can't pet the
// watchdog in time due to a priority inversion / starvation bug.
//
// The challenge:
//   1. Identify the race condition
//   2. Explain why the mutex makes things WORSE, not better
//   3. Fix it using lock-free techniques (atomic, or lock-free queue)
//   4. Bonus: explain why AUTOSAR WdgM uses a cooperative design
//      instead of a shared-mutex design
//
// Build:
//   g++ -std=c++2a -Wall -Wextra -pthread -o puzzle02 puzzle02_watchdog_starvation.cpp
//
// Context:
//   In safety-critical systems, the watchdog check runs at the highest
//   priority. If it contends on the same lock as application code,
//   priority inversion can prevent the app from petting the dog in time.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>

using Clock = std::chrono::steady_clock;

// ============== BUGGY WATCHDOG ==============
class BuggyWatchdog {
    mutable std::mutex mtx_;
    Clock::time_point last_pet_;
    std::chrono::milliseconds timeout_;
    std::atomic<bool> running_{true};
    int reset_count_ = 0;

public:
    explicit BuggyWatchdog(std::chrono::milliseconds timeout)
        : last_pet_(Clock::now()), timeout_(timeout) {}

    // Called by application to say "I'm alive"
    void pet() {
        std::lock_guard lock(mtx_);  // <-- contention point
        last_pet_ = Clock::now();
    }

    // Called by watchdog supervisor (high-priority thread)
    bool check() {
        std::lock_guard lock(mtx_);  // <-- THIS IS THE BUG
        auto elapsed = Clock::now() - last_pet_;
        if (elapsed > timeout_) {
            ++reset_count_;
            last_pet_ = Clock::now();  // reset after "firing"
            return false;  // watchdog fired!
        }
        return true;
    }

    void stop() { running_ = false; }
    bool running() const { return running_.load(); }
    int reset_count() const { return reset_count_; }
};

// Application thread: does heavy work, then pets the watchdog
void application_thread(BuggyWatchdog& wdg) {
    while (wdg.running()) {
        // Simulate heavy computation (holds CPU but not the mutex)
        // ... but sometimes the OS schedules us just as check() grabs the lock
        wdg.pet();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// Watchdog supervisor: checks periodically
void supervisor_thread(BuggyWatchdog& wdg) {
    while (wdg.running()) {
        if (!wdg.check()) {
            std::cout << "[WATCHDOG] FIRED! (phantom reset)\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
}

// "Heavy work" thread that holds mtx_ for too long
// (simulates a logging/diagnostic path that also needs watchdog data)
void diagnostic_thread(BuggyWatchdog& wdg) {
    // This is the real culprit: it grabs the same mutex for a long time
    // In production this might be a periodic status dump, a DTC snapshot,
    // or a UDS diagnostic response that serializes watchdog state.
    while (wdg.running()) {
        // Intentionally not pet() — this thread just reads state
        // but it BLOCKS both pet() and check() by holding the mutex
        std::this_thread::sleep_for(std::chrono::milliseconds(3));

        // Simulate holding the lock for "too long" (e.g., formatting a
        // diagnostic message while holding it)
        // In the real code this is disguised — maybe a shared_ptr copy
        // or a callback that touches the same mutex
    }
}

int main() {
    std::cout << "=== Puzzle: Watchdog Starvation Race ===\n\n";
    std::cout << "Running for 500ms with 15ms timeout...\n\n";

    BuggyWatchdog wdg(std::chrono::milliseconds(15));

    std::thread t_app(application_thread, std::ref(wdg));
    std::thread t_sup(supervisor_thread, std::ref(wdg));
    std::thread t_diag(diagnostic_thread, std::ref(wdg));

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    wdg.stop();

    t_app.join();
    t_sup.join();
    t_diag.join();

    int resets = wdg.reset_count();
    std::cout << "\nPhantom resets: " << resets << "\n";

    if (resets > 0) {
        std::cout << "\nBUG REPRODUCED: Watchdog fired even though app was alive!\n";
        std::cout << "\nROOT CAUSE:\n"
                  << "  Both pet() and check() acquire the same mutex.\n"
                  << "  When diagnostic_thread (or any third party) holds the\n"
                  << "  lock, BOTH the app's pet() and the supervisor's check()\n"
                  << "  are blocked. By the time check() runs, the timeout has\n"
                  << "  elapsed — not because the app died, but because it\n"
                  << "  couldn't reach the watchdog.\n\n"
                  << "FIX:\n"
                  << "  Use std::atomic<Clock::time_point> (or atomic<uint64_t>\n"
                  << "  epoch) so pet() and check() are lock-free.\n"
                  << "  AUTOSAR WdgM avoids this entirely by using cooperative\n"
                  << "  alive counters — the app increments an atomic counter,\n"
                  << "  and the supervisor just reads it. No shared lock needed.\n";
    } else {
        std::cout << "\nNo phantom resets (race didn't manifest this run).\n"
                  << "Try running multiple times or with TSAN to see the issue.\n";
    }

    return 0;
}
