// ============================================================================
// Test stub — compiles standalone or with GTest via CMake
// ============================================================================

#ifdef HAS_GTEST
#include <gtest/gtest.h>
#else
#include <cassert>
#include <iostream>
#define TEST(suite, name) void suite##_##name()
#define EXPECT_TRUE(x)  assert(x)
#define EXPECT_FALSE(x) assert(!(x))
#define EXPECT_EQ(a, b) assert((a) == (b))
#define RUN_TEST(suite, name) do { \
    std::cout << "  " #suite "." #name "..."; \
    suite##_##name(); \
    std::cout << " OK\n"; } while(0)
#endif

#include <flight_sw/mode_manager.hpp>
#include <flight_sw/spsc_queue.hpp>
#include <flight_sw/pid_controller.hpp>
#include <flight_sw/sensor_sim.hpp>
#include <flight_sw/trace_buffer.hpp>
#include <flight_sw/health_monitor.hpp>

using namespace flight_sw;

// ── ModeManager tests ────────────────────────────────────────────────

TEST(ModeManager, StartsInBoot) {
    ModeManager mm;
    EXPECT_EQ(mm.current_state_name(), "Boot");
    EXPECT_TRUE(mm.verify_state_id());
}

TEST(ModeManager, InitCompleteGoesToNominal) {
    ModeManager mm;
    bool changed = mm.transition(InitComplete{});
    EXPECT_TRUE(changed);
    EXPECT_EQ(mm.current_state_name(), "Nominal");
}

TEST(ModeManager, SensorFaultDegrades) {
    ModeManager mm;
    mm.transition(InitComplete{});
    mm.transition(SensorFault{});
    EXPECT_EQ(mm.current_state_name(), "Degraded");
}

TEST(ModeManager, DoubleFaultSafeStop) {
    ModeManager mm;
    mm.transition(InitComplete{});
    mm.transition(SensorFault{});
    mm.transition(SensorFault{});
    EXPECT_EQ(mm.current_state_name(), "SafeStop");
}

TEST(ModeManager, RecoveryFromDegraded) {
    ModeManager mm;
    mm.transition(InitComplete{});
    mm.transition(SensorFault{});
    mm.transition(SensorRecovered{});
    EXPECT_EQ(mm.current_state_name(), "Nominal");
}

TEST(ModeManager, ShutdownFromAnywhere) {
    ModeManager mm;
    mm.transition(ShutdownCmd{});
    EXPECT_EQ(mm.current_state_name(), "Shutdown");
}

TEST(ModeManager, WatchdogTimeoutSafeStop) {
    ModeManager mm;
    mm.transition(InitComplete{});
    mm.transition(WatchdogTimeout{});
    EXPECT_EQ(mm.current_state_name(), "SafeStop");
}

TEST(ModeManager, HammingIdIntegrity) {
    ModeManager mm;
    EXPECT_TRUE(mm.verify_state_id());
    EXPECT_EQ(mm.current_state_id(), state_id::BOOT);
    mm.transition(InitComplete{});
    EXPECT_EQ(mm.current_state_id(), state_id::NOMINAL);
}

// ── SPSCQueue tests ──────────────────────────────────────────────────

TEST(SPSCQueue, PushPopBasic) {
    SPSCQueue<int, 8> q;
    EXPECT_TRUE(q.try_push(42));
    auto val = q.try_pop();
    EXPECT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
}

TEST(SPSCQueue, FullQueueRejects) {
    SPSCQueue<int, 4> q; // capacity = 3 (one slot unused)
    EXPECT_TRUE(q.try_push(1));
    EXPECT_TRUE(q.try_push(2));
    EXPECT_TRUE(q.try_push(3));
    EXPECT_FALSE(q.try_push(4)); // Full
}

TEST(SPSCQueue, EmptyPopReturnsNullopt) {
    SPSCQueue<int, 4> q;
    auto val = q.try_pop();
    EXPECT_FALSE(val.has_value());
}

TEST(SPSCQueue, FIFOOrder) {
    SPSCQueue<int, 16> q;
    for (int i = 0; i < 10; ++i) (void)q.try_push(i);
    for (int i = 0; i < 10; ++i) {
        auto val = q.try_pop();
        EXPECT_TRUE(val.has_value());
        EXPECT_EQ(*val, i);
    }
}

// ── PIDController tests ──────────────────────────────────────────────

TEST(PID, ZeroErrorZeroOutput) {
    PIDConfig cfg{.kp=1.0, .ki=0.0, .kd=0.0, .output_min=-10, .output_max=10, .dt=0.01};
    PIDController pid(cfg);
    pid.set_setpoint(5.0);
    double out = pid.update(5.0); // error = 0
    EXPECT_TRUE(out < 0.001 && out > -0.001);
}

TEST(PID, OutputSaturation) {
    PIDConfig cfg{.kp=100.0, .ki=0.0, .kd=0.0, .output_min=-1.0, .output_max=1.0, .dt=0.01};
    PIDController pid(cfg);
    pid.set_setpoint(10.0);
    double out = pid.update(0.0); // huge error
    EXPECT_TRUE(out <= 1.0);
    EXPECT_TRUE(out >= -1.0);
}

// ── SensorSim tests ──────────────────────────────────────────────────

TEST(SensorSim, NominalReadingsValid) {
    SensorSim s(100.0, 0.01, 123);
    for (int i = 0; i < 10; ++i) {
        auto r = s.read();
        EXPECT_TRUE(r.is_valid);
    }
}

TEST(SensorSim, StuckModeFreezes) {
    SensorSim s(100.0, 0.0, 42); // zero noise for determinism
    (void)s.read(); // prime one reading before stuck
    s.set_fault_mode(FaultMode::STUCK);
    auto r2 = s.read();
    auto r3 = s.read();
    // Values should be identical (stuck)
    EXPECT_TRUE(r2.values[0] == r3.values[0]);
}

// ── TraceBuffer tests ────────────────────────────────────────────────

TEST(TraceBuffer, RecordAndCount) {
    TraceBuffer<16> tb;
    tb.record("TEST", "hello");
    tb.record("TEST", "world");
    EXPECT_EQ(tb.count(), 2u);
}

TEST(TraceBuffer, WrapAround) {
    TraceBuffer<4> tb;
    for (int i = 0; i < 10; ++i) {
        tb.record("TEST", "msg");
    }
    // Count capped at N
    EXPECT_EQ(tb.count(), 4u);
}

// ── HealthMonitor tests ──────────────────────────────────────────────

TEST(HealthMonitor, NoFailureWhenBeating) {
    HealthMonitor hm;
    hm.register_subsystem("test", 10.0, 3);
    hm.heartbeat("test");
    auto failed = hm.check();
    EXPECT_TRUE(failed.empty());
}

// ── Main (standalone mode) ───────────────────────────────────────────
#ifndef HAS_GTEST
int main() {
    std::cout << "=== Flight SW Unit Tests (standalone) ===\n";

    RUN_TEST(ModeManager, StartsInBoot);
    RUN_TEST(ModeManager, InitCompleteGoesToNominal);
    RUN_TEST(ModeManager, SensorFaultDegrades);
    RUN_TEST(ModeManager, DoubleFaultSafeStop);
    RUN_TEST(ModeManager, RecoveryFromDegraded);
    RUN_TEST(ModeManager, ShutdownFromAnywhere);
    RUN_TEST(ModeManager, WatchdogTimeoutSafeStop);
    RUN_TEST(ModeManager, HammingIdIntegrity);

    RUN_TEST(SPSCQueue, PushPopBasic);
    RUN_TEST(SPSCQueue, FullQueueRejects);
    RUN_TEST(SPSCQueue, EmptyPopReturnsNullopt);
    RUN_TEST(SPSCQueue, FIFOOrder);

    RUN_TEST(PID, ZeroErrorZeroOutput);
    RUN_TEST(PID, OutputSaturation);

    RUN_TEST(SensorSim, NominalReadingsValid);
    RUN_TEST(SensorSim, StuckModeFreezes);

    RUN_TEST(TraceBuffer, RecordAndCount);
    RUN_TEST(TraceBuffer, WrapAround);

    RUN_TEST(HealthMonitor, NoFailureWhenBeating);

    std::cout << "\nAll tests passed.\n";
    return 0;
}
#endif
