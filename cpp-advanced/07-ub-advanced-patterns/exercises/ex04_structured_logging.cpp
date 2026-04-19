// Week 7 — Exercise 4: RT-Safe Structured Logging System
// =======================================================
// Compile: g++ -std=c++20 -O2 -Wall -Wextra ex04_structured_logging.cpp -o ex04 -pthread
//
// Topics:
// 1. TraceBuffer<N> — fixed-size ring buffer with atomic write index
// 2. Entry: {timestamp_ns, event_id, payload}
// 3. record() is RT-safe: no allocation, no syscalls
// 4. dump() is non-RT: returns sorted vector
// 5. LogLevel as atomic — changeable at runtime without locks
// 6. "Flight data recorder" capturing last 10000 events
// 7. Simulated crash → dump trace buffer

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

// ============================================================
// Event IDs — define your trace points as an enum
// ============================================================

enum class EventId : uint16_t {
    // Sensor events
    SENSOR_READ        = 0x0100,
    SENSOR_INVALID     = 0x0101,
    SENSOR_TIMEOUT     = 0x0102,

    // Control loop events
    CONTROL_TICK       = 0x0200,
    CONTROL_SETPOINT   = 0x0201,
    CONTROL_OUTPUT     = 0x0202,
    CONTROL_SATURATED  = 0x0203,

    // Motor events
    MOTOR_CMD          = 0x0300,
    MOTOR_FAULT        = 0x0301,
    MOTOR_OVERCURRENT  = 0x0302,

    // System events
    SYS_HEARTBEAT      = 0x0400,
    SYS_WATCHDOG_KICK  = 0x0401,
    SYS_ERROR          = 0x0402,
    SYS_CRASH          = 0x04FF,

    // Navigation events
    NAV_GOAL_SET       = 0x0500,
    NAV_GOAL_REACHED   = 0x0501,
    NAV_OBSTACLE       = 0x0502,
    NAV_LOST           = 0x0503,
};

// ============================================================
// Log Level — atomic, changeable at runtime
// ============================================================

enum class LogLevel : int {
    TRACE   = 0,
    DEBUG   = 1,
    INFO    = 2,
    WARNING = 3,
    ERROR   = 4,
    FATAL   = 5,
};

// Global atomic log level — any thread can read/write without locks
inline std::atomic<int> g_log_level{static_cast<int>(LogLevel::TRACE)};

inline void set_log_level(LogLevel level) {
    g_log_level.store(static_cast<int>(level), std::memory_order_relaxed);
}

inline bool should_log(LogLevel level) {
    return static_cast<int>(level) >= g_log_level.load(std::memory_order_relaxed);
}

// ============================================================
// Trace Entry — 32 bytes, cache-line friendly
// ============================================================

struct alignas(32) TraceEntry {
    uint64_t timestamp_ns;  // 8 bytes — monotonic clock
    uint16_t event_id;      // 2 bytes — EventId
    uint8_t  level;         // 1 byte  — LogLevel
    uint8_t  _pad;          // 1 byte
    uint32_t thread_id;     // 4 bytes — thread identifier
    uint64_t payload;       // 8 bytes — generic data (pointer, int, bits)
    uint64_t payload2;      // 8 bytes — secondary data
    // Total: 32 bytes
};

static_assert(sizeof(TraceEntry) == 32, "TraceEntry must be 32 bytes");

// ============================================================
// TraceBuffer<N> — Lock-free Ring Buffer
// ============================================================

// N must be a power of 2 for fast modulo (bitwise AND)
template <size_t N>
class TraceBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");

public:
    TraceBuffer() : write_idx_(0) {
        // Zero-initialize all entries
        std::memset(entries_, 0, sizeof(entries_));
    }

    // RT-SAFE: No allocation, no syscalls, no locks
    // Single atomic operation: fetch_add on write index
    void record(EventId event_id, LogLevel level,
                uint64_t payload = 0, uint64_t payload2 = 0) {
        if (!should_log(level)) return;

        // Get monotonic timestamp (clock_gettime is vDSO on Linux — no syscall)
        auto now = std::chrono::steady_clock::now();
        uint64_t ts = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()
            ).count()
        );

        // Atomic increment — each writer gets a unique slot
        uint64_t idx = write_idx_.fetch_add(1, std::memory_order_relaxed);
        uint64_t slot = idx & (N - 1); // Fast modulo for power-of-2

        // Write entry (non-atomic writes to the slot — OK because each slot
        // is written by exactly one thread at a time)
        entries_[slot].timestamp_ns = ts;
        entries_[slot].event_id = static_cast<uint16_t>(event_id);
        entries_[slot].level = static_cast<uint8_t>(level);
        entries_[slot].thread_id = thread_id();
        entries_[slot].payload = payload;
        entries_[slot].payload2 = payload2;
    }

    // NON-RT: Allocates, sorts — call only from non-RT context
    std::vector<TraceEntry> dump() const {
        uint64_t total = write_idx_.load(std::memory_order_acquire);
        uint64_t count = std::min(total, static_cast<uint64_t>(N));
        uint64_t start = (total > N) ? (total - N) : 0;

        std::vector<TraceEntry> result;
        result.reserve(count);

        for (uint64_t i = start; i < total; ++i) {
            result.push_back(entries_[i & (N - 1)]);
        }

        // Sort by timestamp (entries may be slightly out of order due to
        // concurrent writes)
        std::sort(result.begin(), result.end(),
                  [](const TraceEntry& a, const TraceEntry& b) {
                      return a.timestamp_ns < b.timestamp_ns;
                  });

        return result;
    }

    uint64_t total_events() const {
        return write_idx_.load(std::memory_order_relaxed);
    }

    size_t capacity() const { return N; }

private:
    static uint32_t thread_id() {
        // Quick thread identifier using hash of thread::id
        auto id = std::this_thread::get_id();
        return static_cast<uint32_t>(std::hash<std::thread::id>{}(id));
    }

    TraceEntry entries_[N];
    std::atomic<uint64_t> write_idx_;
};

// ============================================================
// Event Name Lookup (for dump formatting)
// ============================================================

const char* event_name(uint16_t id) {
    switch (static_cast<EventId>(id)) {
        case EventId::SENSOR_READ:       return "SENSOR_READ";
        case EventId::SENSOR_INVALID:    return "SENSOR_INVALID";
        case EventId::SENSOR_TIMEOUT:    return "SENSOR_TIMEOUT";
        case EventId::CONTROL_TICK:      return "CONTROL_TICK";
        case EventId::CONTROL_SETPOINT:  return "CONTROL_SETPOINT";
        case EventId::CONTROL_OUTPUT:    return "CONTROL_OUTPUT";
        case EventId::CONTROL_SATURATED: return "CONTROL_SATURATED";
        case EventId::MOTOR_CMD:         return "MOTOR_CMD";
        case EventId::MOTOR_FAULT:       return "MOTOR_FAULT";
        case EventId::MOTOR_OVERCURRENT: return "MOTOR_OVERCURRENT";
        case EventId::SYS_HEARTBEAT:     return "SYS_HEARTBEAT";
        case EventId::SYS_WATCHDOG_KICK: return "SYS_WATCHDOG_KICK";
        case EventId::SYS_ERROR:         return "SYS_ERROR";
        case EventId::SYS_CRASH:         return "SYS_CRASH";
        case EventId::NAV_GOAL_SET:      return "NAV_GOAL_SET";
        case EventId::NAV_GOAL_REACHED:  return "NAV_GOAL_REACHED";
        case EventId::NAV_OBSTACLE:      return "NAV_OBSTACLE";
        case EventId::NAV_LOST:          return "NAV_LOST";
        default:                         return "UNKNOWN";
    }
}

const char* level_name(uint8_t lvl) {
    switch (static_cast<LogLevel>(lvl)) {
        case LogLevel::TRACE:   return "TRACE";
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "?";
    }
}

// ============================================================
// Flight Data Recorder — captures last 10000 events
// ============================================================

// 8192 = 2^13, closest power-of-2 under 10000
// For exactly 10000, we'd need 16384 (2^14) — using that
static constexpr size_t RECORDER_SIZE = 16384;

class FlightDataRecorder {
public:
    void record(EventId event, LogLevel level,
                uint64_t payload = 0, uint64_t payload2 = 0) {
        buffer_.record(event, level, payload, payload2);
    }

    // Called on crash — dump everything to stderr
    void crash_dump(const std::string& reason) {
        std::cerr << "\n╔══════════════════════════════════════════════╗\n";
        std::cerr << "║       FLIGHT DATA RECORDER — CRASH DUMP      ║\n";
        std::cerr << "╠══════════════════════════════════════════════╣\n";
        std::cerr << "║ Reason: " << reason << "\n";
        std::cerr << "║ Total events recorded: " << buffer_.total_events() << "\n";
        std::cerr << "║ Buffer capacity: " << buffer_.capacity() << "\n";
        std::cerr << "╚══════════════════════════════════════════════╝\n\n";

        auto entries = buffer_.dump();

        // Print last 50 entries (most recent)
        size_t start = entries.size() > 50 ? entries.size() - 50 : 0;
        std::cerr << "Last " << (entries.size() - start) << " events:\n";
        std::cerr << "─────────────────────────────────────────────────────────────\n";

        uint64_t first_ts = entries.empty() ? 0 : entries[start].timestamp_ns;
        for (size_t i = start; i < entries.size(); ++i) {
            const auto& e = entries[i];
            double relative_ms = static_cast<double>(e.timestamp_ns - first_ts) / 1e6;
            std::cerr << "  [+" << relative_ms << " ms] "
                      << level_name(e.level) << " "
                      << event_name(e.event_id)
                      << " payload=" << e.payload;
            if (e.payload2 != 0) {
                std::cerr << " payload2=" << e.payload2;
            }
            std::cerr << "\n";
        }
        std::cerr << "─────────────────────────────────────────────────────────────\n";
    }

private:
    TraceBuffer<RECORDER_SIZE> buffer_;
};

// Global flight data recorder instance
static FlightDataRecorder g_fdr;

// ============================================================
// Simulated Robot Control Loop
// ============================================================

void simulate_control_loop(int iterations) {
    double setpoint = 1.0;
    double measurement = 0.0;
    double integral = 0.0;
    double dt = 0.001;
    double kp = 2.0, ki = 0.5;
    int fault_at = iterations - 10; // Simulate fault near the end

    for (int i = 0; i < iterations; ++i) {
        // Sensor read
        double sensor_val = measurement + 0.01 * (i % 3 == 0 ? 1.0 : -0.5);
        g_fdr.record(EventId::SENSOR_READ, LogLevel::TRACE,
                     static_cast<uint64_t>(sensor_val * 1000));

        // Control tick
        double error = setpoint - sensor_val;
        integral += error * dt;
        double output = kp * error + ki * integral;

        g_fdr.record(EventId::CONTROL_TICK, LogLevel::TRACE,
                     static_cast<uint64_t>(i));

        // Check saturation
        if (output > 10.0 || output < -10.0) {
            output = std::clamp(output, -10.0, 10.0);
            g_fdr.record(EventId::CONTROL_SATURATED, LogLevel::WARNING,
                         static_cast<uint64_t>(output * 1000));
        }

        // Motor command
        g_fdr.record(EventId::MOTOR_CMD, LogLevel::DEBUG,
                     static_cast<uint64_t>(output * 1000));

        // Heartbeat every 100 iterations
        if (i % 100 == 0) {
            g_fdr.record(EventId::SYS_HEARTBEAT, LogLevel::INFO,
                         static_cast<uint64_t>(i));
        }

        // Simulate motor fault!
        if (i == fault_at) {
            g_fdr.record(EventId::MOTOR_OVERCURRENT, LogLevel::ERROR,
                         15000, // 15A overcurrent
                         static_cast<uint64_t>(i));
            g_fdr.record(EventId::MOTOR_FAULT, LogLevel::FATAL,
                         0xDEAD,
                         static_cast<uint64_t>(i));
        }

        measurement += 0.1 * output * dt;
    }

    // Crash!
    g_fdr.record(EventId::SYS_CRASH, LogLevel::FATAL, 0xBADC0DE);
}

// ============================================================
// Multi-threaded Demo
// ============================================================

void sensor_thread(TraceBuffer<4096>& buf, int count) {
    for (int i = 0; i < count; ++i) {
        buf.record(EventId::SENSOR_READ, LogLevel::TRACE,
                   static_cast<uint64_t>(i * 100));
    }
}

void control_thread(TraceBuffer<4096>& buf, int count) {
    for (int i = 0; i < count; ++i) {
        buf.record(EventId::CONTROL_TICK, LogLevel::DEBUG,
                   static_cast<uint64_t>(i));
    }
}

void demo_multi_threaded() {
    std::cout << "\nPart 2: Multi-threaded Trace Buffer\n";
    TraceBuffer<4096> buf;

    constexpr int EVENTS_PER_THREAD = 10000;
    std::thread t1(sensor_thread, std::ref(buf), EVENTS_PER_THREAD);
    std::thread t2(control_thread, std::ref(buf), EVENTS_PER_THREAD);

    t1.join();
    t2.join();

    std::cout << "  Total events: " << buf.total_events() << "\n";
    std::cout << "  Expected: " << 2 * EVENTS_PER_THREAD << "\n";

    auto entries = buf.dump();
    std::cout << "  Dumped entries: " << entries.size() << "\n";

    // Count by type
    int sensor_count = 0, control_count = 0;
    for (const auto& e : entries) {
        if (static_cast<EventId>(e.event_id) == EventId::SENSOR_READ) ++sensor_count;
        if (static_cast<EventId>(e.event_id) == EventId::CONTROL_TICK) ++control_count;
    }
    std::cout << "  Sensor events: " << sensor_count << "\n";
    std::cout << "  Control events: " << control_count << "\n";
}

// ============================================================
// Log Level Runtime Change Demo
// ============================================================

void demo_log_level() {
    std::cout << "\nPart 3: Runtime Log Level Change\n";
    TraceBuffer<256> buf;

    // Log at TRACE level
    set_log_level(LogLevel::TRACE);
    for (int i = 0; i < 10; ++i) {
        buf.record(EventId::SENSOR_READ, LogLevel::TRACE, static_cast<uint64_t>(i));
    }
    std::cout << "  After 10 TRACE records (level=TRACE): "
              << buf.total_events() << " events\n";

    // Raise to WARNING — TRACE events filtered
    set_log_level(LogLevel::WARNING);
    for (int i = 0; i < 10; ++i) {
        buf.record(EventId::SENSOR_READ, LogLevel::TRACE, static_cast<uint64_t>(i));
    }
    std::cout << "  After 10 TRACE records (level=WARNING): "
              << buf.total_events() << " events (same — filtered out)\n";

    // WARNING events still recorded
    for (int i = 0; i < 5; ++i) {
        buf.record(EventId::CONTROL_SATURATED, LogLevel::WARNING, static_cast<uint64_t>(i));
    }
    std::cout << "  After 5 WARNING records (level=WARNING): "
              << buf.total_events() << " events\n";

    // Reset
    set_log_level(LogLevel::TRACE);
}

// ============================================================
// Benchmark: record() latency
// ============================================================

void benchmark_record() {
    std::cout << "\nPart 4: record() Latency Benchmark\n";
    TraceBuffer<16384> buf;
    constexpr int ITERS = 1'000'000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        buf.record(EventId::CONTROL_TICK, LogLevel::TRACE,
                   static_cast<uint64_t>(i));
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "  " << ITERS << " records in " << ns / 1'000'000 << " ms\n";
    std::cout << "  Average: " << ns / ITERS << " ns per record\n";
    std::cout << "  (Target: < 100ns for RT-safe logging)\n";
}

// ============================================================

int main() {
    std::cout << "=== RT-SAFE STRUCTURED LOGGING ===\n\n";

    // Part 1: Flight Data Recorder with simulated crash
    std::cout << "Part 1: Flight Data Recorder — Simulated Crash\n";
    std::cout << "  Running 1000+iteration control loop...\n";
    simulate_control_loop(1000);
    g_fdr.crash_dump("Motor overcurrent → controller fault");

    // Part 2: Multi-threaded
    demo_multi_threaded();

    // Part 3: Runtime log level
    demo_log_level();

    // Part 4: Benchmark
    benchmark_record();

    std::cout << "\n=== DONE ===\n";
    return 0;
}
