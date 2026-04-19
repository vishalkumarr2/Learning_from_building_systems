#pragma once
// ============================================================================
// Watchdog — Software watchdog timer
//
// Must be kicked within the timeout period or the expiry callback fires.
// Runs a monitor on a separate thread.
// ============================================================================

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace flight_sw {

class Watchdog {
public:
    using Callback = std::function<void()>;

    explicit Watchdog(std::chrono::milliseconds timeout, Callback on_expiry)
        : timeout_{timeout}
        , on_expiry_{std::move(on_expiry)}
    {}

    ~Watchdog() {
        disarm();
    }

    // Non-copyable, non-movable
    Watchdog(Watchdog const&) = delete;
    Watchdog& operator=(Watchdog const&) = delete;

    // Start the watchdog monitor thread
    inline void arm() {
        if (armed_.load(std::memory_order_relaxed)) return;

        armed_.store(true, std::memory_order_release);
        expired_.store(false, std::memory_order_relaxed);
        kicked_.store(true, std::memory_order_release); // initial kick

        thread_ = std::thread([this] { monitor_loop(); });
    }

    // Stop the watchdog monitor
    inline void disarm() {
        armed_.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            cv_.notify_all();
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    // Reset the watchdog timer (call periodically from main loop)
    inline void kick() noexcept {
        kicked_.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(mtx_);
        cv_.notify_one();
    }

    [[nodiscard]] inline bool is_armed() const noexcept {
        return armed_.load(std::memory_order_acquire);
    }

    [[nodiscard]] inline bool has_expired() const noexcept {
        return expired_.load(std::memory_order_acquire);
    }

private:
    std::chrono::milliseconds timeout_;
    Callback                  on_expiry_;
    std::atomic<bool>         armed_{false};
    std::atomic<bool>         expired_{false};
    std::atomic<bool>         kicked_{false};
    std::mutex                mtx_;
    std::condition_variable   cv_;
    std::thread               thread_;

    inline void monitor_loop() {
        while (armed_.load(std::memory_order_acquire)) {
            // Clear the kick flag
            kicked_.store(false, std::memory_order_release);

            // Wait for timeout or kick
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, timeout_, [this] {
                    return kicked_.load(std::memory_order_acquire) ||
                           !armed_.load(std::memory_order_acquire);
                });
            }

            if (!armed_.load(std::memory_order_acquire)) break;

            // If not kicked within timeout → fire
            if (!kicked_.load(std::memory_order_acquire)) {
                expired_.store(true, std::memory_order_release);
                if (on_expiry_) {
                    on_expiry_();
                }
                break; // Watchdog fires once
            }
        }
    }
};

} // namespace flight_sw
