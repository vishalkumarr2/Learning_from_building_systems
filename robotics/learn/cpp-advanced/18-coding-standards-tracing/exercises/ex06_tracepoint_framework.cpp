// Exercise 6: Build a Lightweight Tracepoint Framework
//
// Implements a simplified version of LTTng-UST style userspace tracing:
//   - Zero-overhead when disabled (compile-time switch)
//   - Ring buffer for lock-free recording
//   - Structured event format with timestamps
//   - Post-mortem dump of trace events
//
// This teaches the CONCEPTS behind LTTng without requiring lttng-tools.
//
// TOOL PRACTICE:
//   After understanding this, try real LTTng:
//     sudo apt install lttng-tools liblttng-ust-dev
//     lttng create my-session && lttng enable-event --userspace '*'

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// ========================================================================
// Tracepoint Framework
// ========================================================================

namespace trace {

// Compile-time tracing enable/disable
// In production: controlled by CMake option or preprocessor define
#ifndef TRACING_ENABLED
#define TRACING_ENABLED 1
#endif

// Event types — like LTTng tracepoint providers
enum class EventType : uint8_t {
    kNone = 0,
    kFunctionEntry,
    kFunctionExit,
    kSensorReading,
    kStateChange,
    kError,
    kCustom
};

// Structured trace event — fixed-size for ring buffer
struct TraceEvent {
    uint64_t timestamp_ns = 0;   // monotonic clock
    EventType evt_type_ = EventType::kNone;
    uint8_t thread_id = 0;       // simplified thread ID (0-255)
    uint16_t payload_id = 0;     // function/sensor/state ID
    int32_t value = 0;           // arbitrary payload
    char tag[16] = {};           // human-readable tag

    void set_tag(const char* t) {
        std::strncpy(tag, t, sizeof(tag) - 1);
        tag[sizeof(tag) - 1] = '\0';
    }
};

static_assert(sizeof(TraceEvent) <= 40, "TraceEvent should be compact");

// Ring buffer — lock-free single-producer or multi-producer with atomics
class RingBuffer {
public:
    static constexpr size_t kCapacity = 4096;

    void record(const TraceEvent& event) {
        size_t idx = write_pos_.fetch_add(1, std::memory_order_relaxed) % kCapacity;
        events_[idx] = event;
        count_.fetch_add(1, std::memory_order_release);
    }

    // Dump all events in chronological order (approximate — ring buffer wraps)
    size_t dump(TraceEvent* out, size_t max_events) const {
        size_t total = count_.load(std::memory_order_acquire);
        size_t start = (total > kCapacity) ? (total - kCapacity) : 0;
        size_t to_dump = (total > kCapacity) ? kCapacity : total;
        if (to_dump > max_events) to_dump = max_events;

        for (size_t i = 0; i < to_dump; ++i) {
            size_t ring_idx = (start + i) % kCapacity;
            out[i] = events_[ring_idx];
        }
        return to_dump;
    }

    size_t event_count() const { return count_.load(std::memory_order_acquire); }

    void clear() {
        write_pos_.store(0, std::memory_order_relaxed);
        count_.store(0, std::memory_order_release);
    }

private:
    std::array<TraceEvent, kCapacity> events_{};
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> count_{0};
};

// Global trace buffer (singleton pattern, like LTTng session)
inline RingBuffer& global_buffer() {
    static RingBuffer buf;
    return buf;
}

// Get monotonic timestamp in nanoseconds
inline uint64_t now_ns() {
    auto tp = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            tp.time_since_epoch()).count());
}

// Simplified thread ID (hash to 0-255)
inline uint8_t thread_index() {
    static thread_local uint8_t tid = static_cast<uint8_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFF);
    return tid;
}

// ── Tracepoint macros (zero-overhead when disabled) ──

#if TRACING_ENABLED

#define TRACE_EVENT(evt_type, id, val, tag_str)            \
    do {                                                     \
        ::trace::TraceEvent ev;                              \
        ev.timestamp_ns = ::trace::now_ns();                 \
        ev.evt_type_ = (evt_type);                           \
        ev.thread_id = ::trace::thread_index();              \
        ev.payload_id = static_cast<uint16_t>(id);           \
        ev.value = static_cast<int32_t>(val);                \
        ev.set_tag(tag_str);                                 \
        ::trace::global_buffer().record(ev);                 \
    } while (0)

#define TRACE_FUNCTION_ENTRY(name)                           \
    TRACE_EVENT(::trace::EventType::kFunctionEntry, __LINE__, 0, name)

#define TRACE_FUNCTION_EXIT(name)                            \
    TRACE_EVENT(::trace::EventType::kFunctionExit, __LINE__, 0, name)

#define TRACE_SENSOR(id, value)                              \
    TRACE_EVENT(::trace::EventType::kSensorReading, id, value, "sensor")

#define TRACE_STATE(id, new_state)                           \
    TRACE_EVENT(::trace::EventType::kStateChange, id, new_state, "state")

#define TRACE_ERROR(id, error_code)                          \
    TRACE_EVENT(::trace::EventType::kError, id, error_code, "error")

// RAII scope tracer (like LTTng nested regions)
class ScopeTracer {
    const char* name_;
public:
    explicit ScopeTracer(const char* name) : name_(name) {
        TRACE_FUNCTION_ENTRY(name_);
    }
    ~ScopeTracer() {
        TRACE_FUNCTION_EXIT(name_);
    }
    ScopeTracer(const ScopeTracer&) = delete;
    ScopeTracer& operator=(const ScopeTracer&) = delete;
};

#define TRACE_SCOPE(name) ::trace::ScopeTracer _scope_tracer_##__LINE__(name)

#else  // TRACING_ENABLED == 0

#define TRACE_EVENT(type, id, val, tag) ((void)0)
#define TRACE_FUNCTION_ENTRY(name) ((void)0)
#define TRACE_FUNCTION_EXIT(name) ((void)0)
#define TRACE_SENSOR(id, value) ((void)0)
#define TRACE_STATE(id, new_state) ((void)0)
#define TRACE_ERROR(id, error_code) ((void)0)
#define TRACE_SCOPE(name) ((void)0)

#endif  // TRACING_ENABLED

// Print helper
const char* event_type_name(EventType t) {
    switch (t) {
        case EventType::kNone:          return "NONE";
        case EventType::kFunctionEntry: return "ENTER";
        case EventType::kFunctionExit:  return "EXIT";
        case EventType::kSensorReading: return "SENSOR";
        case EventType::kStateChange:   return "STATE";
        case EventType::kError:         return "ERROR";
        case EventType::kCustom:        return "CUSTOM";
    }
    return "???";
}

void print_trace(size_t max_events = 100) {
    std::vector<TraceEvent> events(max_events);
    size_t n = global_buffer().dump(events.data(), max_events);

    std::printf("\n--- Trace Dump (%zu events) ---\n", n);
    std::printf("%-16s %-8s %-4s %-6s %-8s %s\n",
                "Timestamp(ns)", "Type", "TID", "ID", "Value", "Tag");
    for (size_t i = 0; i < n; ++i) {
        const auto& e = events[i];
        std::printf("%-16lu %-8s %-4u %-6u %-8d %s\n",
                    static_cast<unsigned long>(e.timestamp_ns),
                    event_type_name(e.evt_type_),
                    static_cast<unsigned>(e.thread_id),
                    static_cast<unsigned>(e.payload_id),
                    e.value,
                    e.tag);
    }
}

}  // namespace trace

// ========================================================================
// Example: Instrumented Robot Control Loop
// ========================================================================

namespace robot {

enum class State { kIdle, kMoving, kStopped, kError };

class Controller {
    State state_ = State::kIdle;
    int cycle_count_ = 0;

public:
    void read_sensors() {
        TRACE_SCOPE("read_sensors");
        // Simulate sensor readings
        TRACE_SENSOR(1, 42 + cycle_count_);   // encoder
        TRACE_SENSOR(2, 100 - cycle_count_);  // lidar range
    }

    void compute_control() {
        TRACE_SCOPE("compute_ctrl");
        // Simulate computation
        volatile double x = 0.0;
        for (int i = 0; i < 100; ++i) x += 0.01;
        (void)x;
    }

    void send_commands() {
        TRACE_SCOPE("send_cmds");
        cycle_count_++;
        if (cycle_count_ == 5) {
            state_ = State::kMoving;
            TRACE_STATE(0, static_cast<int>(state_));
        }
        if (cycle_count_ == 8) {
            TRACE_ERROR(99, -1);  // simulated error
            state_ = State::kError;
            TRACE_STATE(0, static_cast<int>(state_));
        }
    }

    void run_cycle() {
        TRACE_SCOPE("control_cycle");
        read_sensors();
        compute_control();
        send_commands();
    }

    State state() const { return state_; }
    int cycles() const { return cycle_count_; }
};

}  // namespace robot

// ========================================================================
// Tests
// ========================================================================

void test_ring_buffer() {
    trace::RingBuffer rb;

    // Record some events
    for (int i = 0; i < 100; ++i) {
        trace::TraceEvent ev;
        ev.timestamp_ns = static_cast<uint64_t>(i);
        ev.evt_type_ = trace::EventType::kCustom;
        ev.value = i;
        rb.record(ev);
    }
    assert(rb.event_count() == 100);

    // Dump and verify
    std::vector<trace::TraceEvent> out(100);
    size_t n = rb.dump(out.data(), 100);
    assert(n == 100);
    assert(out[0].value == 0);
    assert(out[99].value == 99);
}

void test_ring_buffer_wrap() {
    trace::RingBuffer rb;

    // Overflow the ring buffer
    for (int i = 0; i < 5000; ++i) {
        trace::TraceEvent ev;
        ev.value = i;
        rb.record(ev);
    }
    assert(rb.event_count() == 5000);

    // Should only keep last kCapacity events
    std::vector<trace::TraceEvent> out(trace::RingBuffer::kCapacity);
    size_t n = rb.dump(out.data(), trace::RingBuffer::kCapacity);
    assert(n == trace::RingBuffer::kCapacity);
    // First dumped event should be event #(5000 - 4096) = #904
    assert(out[0].value == 904);
}

void test_instrumented_controller() {
    trace::global_buffer().clear();

    robot::Controller ctrl;
    for (int i = 0; i < 10; ++i)
        ctrl.run_cycle();

    assert(ctrl.cycles() == 10);
    assert(ctrl.state() == robot::State::kError);

    // Verify we recorded trace events
    size_t count = trace::global_buffer().event_count();
    assert(count > 50);  // Should have many events from 10 cycles
}

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    test_ring_buffer();
    test_ring_buffer_wrap();
    test_instrumented_controller();

    std::printf("ex06_tracepoint_framework: ALL TESTS PASSED\n");

    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0)
            dump = true;
    }

    if (dump) {
        trace::global_buffer().clear();
        robot::Controller ctrl;
        for (int i = 0; i < 10; ++i)
            ctrl.run_cycle();
        trace::print_trace(200);
    }

    return 0;
}
