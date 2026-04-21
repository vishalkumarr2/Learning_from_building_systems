/**
 * Exercise 03: Cooperative Coroutine Scheduler
 *
 * This exercise builds a simple cooperative scheduler that:
 *   - Manages a queue of coroutines (tasks)
 *   - Round-robin resumes each coroutine until all complete
 *   - Demonstrates priority scheduling as a variant
 *   - Shows how this relates to event loops like ROS2 executors
 *
 * Concepts demonstrated:
 *   - Custom awaiter that yields control back to the scheduler
 *   - Scheduler as a coroutine_handle queue manager
 *   - Cooperative multitasking without threads
 *   - Priority scheduling by reordering the run queue
 *
 * Compile:
 *   g++-10 -std=c++20 -fcoroutines -Wall -Wextra -o ex03 ex03_coroutine_scheduler.cpp
 *   clang++-14 -std=c++20 -Wall -Wextra -o ex03 ex03_coroutine_scheduler.cpp
 */

#include <coroutine>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// Forward declaration — the scheduler must be accessible to the SchedulerTask
// =============================================================================

class Scheduler;

// Global pointer to the currently running scheduler.
// In production you'd use thread_local or pass via coroutine promise.
// For educational simplicity, we use a global.
static Scheduler* g_scheduler = nullptr;

// =============================================================================
// SchedulerTask — a coroutine that the scheduler manages
// =============================================================================

class SchedulerTask {
public:
    struct promise_type {
        SchedulerTask get_return_object() {
            return SchedulerTask{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        // Suspend at start — scheduler will resume us when ready
        std::suspend_always initial_suspend() noexcept { return {}; }

        // Suspend at end — scheduler detects done() and cleans up
        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() { std::terminate(); }
        void return_void() {}
    };

    explicit SchedulerTask(std::coroutine_handle<promise_type> h)
        : handle_(h) {}

    SchedulerTask(SchedulerTask&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    SchedulerTask& operator=(SchedulerTask&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    SchedulerTask(const SchedulerTask&) = delete;
    SchedulerTask& operator=(const SchedulerTask&) = delete;

    ~SchedulerTask() {
        if (handle_) handle_.destroy();
    }

    std::coroutine_handle<> handle() const { return handle_; }
    bool done() const { return handle_ ? handle_.done() : true; }

private:
    std::coroutine_handle<promise_type> handle_;
};

// =============================================================================
// Scheduler — round-robin cooperative scheduler
// =============================================================================

class Scheduler {
public:
    // An awaitable that suspends the current coroutine and
    // puts it back in the scheduler's run queue.
    struct YieldAwaiter {
        bool await_ready() noexcept {
            return false; // Always suspend
        }

        void await_suspend(std::coroutine_handle<> h) noexcept {
            // Put this coroutine back in the run queue
            if (g_scheduler) {
                g_scheduler->enqueue(h);
            }
        }

        void await_resume() noexcept {
            // Nothing to return — yield is just a suspension point
        }
    };

    // Spawn a task — add its handle to the run queue
    void spawn(SchedulerTask task) {
        auto h = task.handle();
        tasks_.push_back(std::move(task));
        run_queue_.push(h);
    }

    // Run all tasks until they all complete.
    // This is the event loop — analogous to rclcpp::spin().
    void run() {
        g_scheduler = this;
        std::size_t iteration = 0;

        while (!run_queue_.empty()) {
            auto handle = run_queue_.front();
            run_queue_.pop();

            if (!handle.done()) {
                ++iteration;
                handle.resume();
                // After resume, the coroutine either:
                //   1. Hit co_await yield() → YieldAwaiter enqueued it back
                //   2. Completed → handle.done() is true, don't re-enqueue
            }
            // If done, it falls out of the queue naturally
        }

        g_scheduler = nullptr;
        std::cout << "  [Scheduler] All tasks complete after " << iteration
                  << " iterations.\n";
    }

    void enqueue(std::coroutine_handle<> h) { run_queue_.push(h); }

private:
    std::vector<SchedulerTask> tasks_; // Owns the task lifetimes
    std::queue<std::coroutine_handle<>> run_queue_; // Run queue (non-owning)
};

// =============================================================================
// yield() — suspension point that returns control to the scheduler
// =============================================================================

// Any coroutine can call: co_await yield();
// This is the cooperative handoff point — like calling ros::spinOnce()
// or std::this_thread::yield() but for coroutines.
Scheduler::YieldAwaiter yield() { return {}; }

// =============================================================================
// Example tasks — each does "work" in steps, yielding between them
// =============================================================================

SchedulerTask sensor_reader(const std::string& name, int readings) {
    for (int i = 0; i < readings; ++i) {
        std::cout << "    [" << name << "] Reading sensor data " << (i + 1)
                  << "/" << readings << "\n";
        co_await yield(); // Give other tasks a chance to run
    }
    std::cout << "    [" << name << "] Done reading.\n";
}

SchedulerTask data_processor(const std::string& name, int steps) {
    for (int i = 0; i < steps; ++i) {
        std::cout << "    [" << name << "] Processing step " << (i + 1) << "/"
                  << steps << "\n";
        co_await yield();
    }
    std::cout << "    [" << name << "] Processing complete.\n";
}

SchedulerTask logger(const std::string& name, int entries) {
    for (int i = 0; i < entries; ++i) {
        std::cout << "    [" << name << "] Logging entry " << (i + 1) << "/"
                  << entries << "\n";
        co_await yield();
    }
    std::cout << "    [" << name << "] Logging complete.\n";
}

// =============================================================================
// Priority Scheduler — variant with priority levels
// =============================================================================

class PriorityScheduler {
public:
    // Priority levels (lower number = higher priority)
    enum class Priority : int { HIGH = 0, NORMAL = 1, LOW = 2 };

    struct PriorityYieldAwaiter {
        Priority prio;

        bool await_ready() noexcept { return false; }

        void await_suspend(std::coroutine_handle<> h) noexcept {
            if (g_priority_scheduler) {
                g_priority_scheduler->enqueue(h, prio);
            }
        }

        void await_resume() noexcept {}
    };

    void spawn(SchedulerTask task, Priority prio = Priority::NORMAL) {
        auto h = task.handle();
        tasks_.push_back(std::move(task));
        enqueue(h, prio);
    }

    void run() {
        g_priority_scheduler = this;
        std::size_t iteration = 0;

        while (!pq_.empty()) {
            auto [prio, handle] = pq_.top();
            pq_.pop();

            if (!handle.done()) {
                ++iteration;
                handle.resume();
            }
        }

        g_priority_scheduler = nullptr;
        std::cout << "  [PriorityScheduler] All tasks complete after "
                  << iteration << " iterations.\n";
    }

    void enqueue(std::coroutine_handle<> h, Priority prio) {
        pq_.push({prio, h});
    }

    static PriorityScheduler* g_priority_scheduler;

private:
    struct Entry {
        Priority priority;
        std::coroutine_handle<> handle;

        // Higher priority (lower number) should come first
        bool operator<(const Entry& other) const {
            return static_cast<int>(priority) >
                   static_cast<int>(other.priority);
        }
    };

    std::vector<SchedulerTask> tasks_;
    std::priority_queue<Entry> pq_;
};

PriorityScheduler* PriorityScheduler::g_priority_scheduler = nullptr;

// Helper to yield with a specific priority
PriorityScheduler::PriorityYieldAwaiter yield_with_priority(
    PriorityScheduler::Priority prio) {
    return {prio};
}

// =============================================================================
// Priority task examples
// =============================================================================

SchedulerTask high_priority_task() {
    for (int i = 0; i < 3; ++i) {
        std::cout << "    [HIGH PRIO] Critical work step " << (i + 1) << "\n";
        co_await yield_with_priority(PriorityScheduler::Priority::HIGH);
    }
    std::cout << "    [HIGH PRIO] Done.\n";
}

SchedulerTask normal_priority_task() {
    for (int i = 0; i < 3; ++i) {
        std::cout << "    [NORMAL]    Regular work step " << (i + 1) << "\n";
        co_await yield_with_priority(PriorityScheduler::Priority::NORMAL);
    }
    std::cout << "    [NORMAL]    Done.\n";
}

SchedulerTask low_priority_task() {
    for (int i = 0; i < 3; ++i) {
        std::cout << "    [LOW PRIO]  Background work step " << (i + 1)
                  << "\n";
        co_await yield_with_priority(PriorityScheduler::Priority::LOW);
    }
    std::cout << "    [LOW PRIO]  Done.\n";
}

// =============================================================================
// main()
// =============================================================================

int main() {
    std::cout << "================================================\n";
    std::cout << " Exercise 03: Cooperative Coroutine Scheduler\n";
    std::cout << "================================================\n\n";

    // --- Round-robin scheduler ---
    std::cout << "--- Round-Robin Scheduler ---\n";
    std::cout << "  Three tasks interleaved cooperatively:\n\n";
    {
        Scheduler sched;
        sched.spawn(sensor_reader("Lidar", 3));
        sched.spawn(data_processor("NavFilter", 4));
        sched.spawn(logger("RosbagWriter", 2));
        sched.run();
    }
    std::cout << "\n";

    // --- Priority scheduler ---
    std::cout << "--- Priority Scheduler ---\n";
    std::cout << "  High priority tasks run before lower ones at each step:\n\n";
    {
        PriorityScheduler sched;
        PriorityScheduler::g_priority_scheduler = &sched;

        sched.spawn(low_priority_task(), PriorityScheduler::Priority::LOW);
        sched.spawn(normal_priority_task(),
                    PriorityScheduler::Priority::NORMAL);
        sched.spawn(high_priority_task(), PriorityScheduler::Priority::HIGH);
        sched.run();
    }
    std::cout << "\n";

    // --- ROS2 executor analogy ---
    std::cout << "--- ROS2 Executor Analogy ---\n";
    std::cout << R"(
  In ROS2, the executor (rclcpp::executors::SingleThreadedExecutor)
  is essentially a cooperative scheduler:

    executor.add_node(lidar_node);
    executor.add_node(nav_node);
    executor.add_node(logger_node);
    executor.spin();  // <-- This is the event loop

  Each callback (subscription, timer, service) is like a coroutine
  step. The executor round-robins between ready callbacks.

  With C++20 coroutines, complex multi-step callbacks could be
  written as linear code with co_await, avoiding callback chains.

  Example: instead of separate on_scan() + on_process() + on_publish()
  callbacks chained via message passing, a single coroutine could:

    auto scan = co_await next_message<LaserScan>("/scan");
    auto cost = co_await compute_costmap(scan);
    co_await publish("/costmap", cost);

  This is the direction modern async frameworks are heading.
)" << "\n";

    std::cout << "================================================\n";
    std::cout << " All scheduler exercises complete.\n";
    std::cout << "================================================\n";

    return 0;
}
