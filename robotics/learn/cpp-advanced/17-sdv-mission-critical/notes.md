# Module 17: Software-Defined Vehicles & Mission-Critical C++

## Overview

A **Software-Defined Vehicle (SDV)** is a vehicle where software — not mechanical linkages or fixed ECU firmware — defines behavior, features, and capabilities. The vehicle becomes a rolling computer that receives over-the-air (OTA) updates, runs containerized services, and coordinates dozens of heterogeneous processors through middleware.

**Mission-critical** means failure causes loss of life, destruction of expensive hardware, or catastrophic environmental damage. In an SDV, braking, steering, and power management are mission-critical. Infotainment is not. The boundary between them is enforced by architecture, not wishful thinking.

This module teaches the C++ patterns, architectures, and disciplines used in production SDV systems — from AUTOSAR Adaptive to zonal architectures, from functional safety (ISO 26262) to cybersecurity (ISO/SAE 21434), from deterministic execution to graceful degradation.

---

## 1. The SDV Architecture Revolution

### 1.1 From Distributed ECUs to Centralized Compute

Traditional vehicles have 70–150 Electronic Control Units (ECUs), each running a single function: one for ABS, one for airbags, one for power windows. Each ECU is a microcontroller (ARM Cortex-M, Renesas RH850) running bare-metal or AUTOSAR Classic.

**Problems with the distributed model:**

| Problem | Impact |
|---|---|
| 150 ECUs × 150 unique firmware builds | Integration testing is exponential |
| CAN bus bandwidth (~1 Mbps) | Cannot move camera frames or lidar clouds |
| No OTA update for most ECUs | Recalls are physical, costing $100M+ |
| Each supplier owns their ECU firmware | OEM cannot innovate on software |
| Fixed function per ECU | A parking sensor ECU can't help with ADAS |

**The SDV answer: Zonal Architecture**

```
┌─────────────────────────────────────────────────┐
│              Central Compute (HPC)               │
│   ┌─────────┐  ┌──────────┐  ┌──────────────┐  │
│   │ ADAS    │  │ Vehicle  │  │ Infotainment │  │
│   │ Domain  │  │ Dynamics │  │ Domain       │  │
│   └────┬────┘  └────┬─────┘  └──────┬───────┘  │
│        │            │               │           │
│   ┌────┴────────────┴───────────────┴────┐      │
│   │    Service-Oriented Middleware        │      │
│   │    (SOME/IP, DDS, Zenoh)             │      │
│   └────┬────────────┬───────────────┬────┘      │
└────────┼────────────┼───────────────┼───────────┘
         │ Ethernet   │ Ethernet      │ Ethernet
    ┌────┴────┐  ┌────┴────┐    ┌─────┴────┐
    │ Zone    │  │ Zone    │    │ Zone     │
    │ Gateway │  │ Gateway │    │ Gateway  │
    │ (Front) │  │ (Rear)  │    │ (Left)   │
    └────┬────┘  └────┬────┘    └────┬─────┘
         │            │              │
    Sensors,      Sensors,       Sensors,
    Actuators     Actuators      Actuators
```

**Key shifts:**
- **ECU → Software service**: ABS becomes a containerized service, deployable independently
- **CAN → Automotive Ethernet**: 100 Mbps → 10 Gbps backbone
- **Fixed firmware → OTA-updatable**: Services updated like phone apps
- **Supplier ECU → OEM-controlled compute**: Software differentiation at the OEM level

### 1.2 The SDV Software Stack

```
┌─────────────────────────────────────────────┐
│           Application Layer                  │
│  (ADAS, Body Control, Chassis, Diagnostics) │
├─────────────────────────────────────────────┤
│           Service Layer                      │
│  (AUTOSAR Adaptive ara::com, DDS, SOME/IP)  │
├─────────────────────────────────────────────┤
│           Middleware / OS Abstraction         │
│  (POSIX, Execution Manager, Health Monitor)  │
├─────────────────────────────────────────────┤
│           OS Layer                           │
│  (QNX, Linux + PREEMPT_RT, PikeOS, seL4)    │
├─────────────────────────────────────────────┤
│           Hypervisor                         │
│  (XEN, ACRN, QNX Hypervisor)                │
├─────────────────────────────────────────────┤
│           Hardware                           │
│  (NVIDIA Orin, Qualcomm SA8650, NXP S32G)   │
└─────────────────────────────────────────────┘
```

### 1.3 Why C++ Dominates SDV

| Language | Use Case | Why |
|---|---|---|
| **C++17/20** | ADAS, middleware, vehicle dynamics | Deterministic, zero-cost abstractions, ISO 26262 qualified toolchains |
| **C** | Microcontroller firmware, bootloaders | No runtime, smallest footprint |
| **Rust** | New security-critical components | Memory safety, but immature toolchain qualification |
| **Python** | Test automation, ML model training | Not for production runtime |
| **Java/Kotlin** | Android Automotive infotainment | Not for safety-critical paths |

C++ owns the mission-critical layer because:
1. **Deterministic timing** — no GC pauses, RAII for resource management
2. **Qualified compilers** — GCC, Clang, and Green Hills are ISO 26262 qualified
3. **AUTOSAR Adaptive** — defined entirely in C++14/17
4. **Zero-cost abstractions** — `std::variant`, `constexpr`, templates compile to optimal code
5. **Decades of automotive validation** — proven in every mass-production vehicle since 2005

---

## 2. AUTOSAR Adaptive Platform — The SDV Framework

### 2.1 What is AUTOSAR Adaptive?

AUTOSAR Adaptive is a standardized middleware platform for high-performance automotive computers. Unlike AUTOSAR Classic (static, single-threaded, C-only), Adaptive is:

- **POSIX-based**: Runs on Linux or QNX
- **C++14/17**: All APIs are modern C++
- **Service-oriented**: Components communicate via service discovery and proxies
- **Dynamic**: Services start, stop, and update at runtime

### 2.2 Key Functional Clusters

| Cluster | Purpose | C++ API Namespace |
|---|---|---|
| **Communication Management (ara::com)** | Service-oriented communication (SOME/IP, DDS) | `ara::com` |
| **Execution Management (ara::exec)** | Process lifecycle, deterministic scheduling | `ara::exec` |
| **Diagnostics (ara::diag)** | UDS (ISO 14229), DTCs, fault memory | `ara::diag` |
| **Persistency (ara::per)** | Key-value store, file proxies | `ara::per` |
| **Update & Config (ara::ucm)** | OTA updates, software package management | `ara::ucm` |
| **Cryptography (ara::crypto)** | HSM integration, secure boot, TLS | `ara::crypto` |
| **Identity & Access (ara::iam)** | Service-level access control | `ara::iam` |
| **Health Management (ara::phm)** | Watchdog supervision, health monitoring | `ara::phm` |
| **Log & Trace (ara::log)** | DLT-compatible structured logging | `ara::log` |

### 2.3 ara::com — Service Communication in C++

The core of Adaptive AUTOSAR. Services are defined in ARXML (Adaptive autosar XML), then code-generated into proxy/skeleton classes.

```cpp
// Generated service interface: RadarService
// Provider side (Skeleton)
class RadarServiceSkeleton : public ara::com::ServiceSkeleton {
public:
    // Event: new radar object detected
    ara::com::Event<RadarObject> object_detected;

    // Method: calibrate the radar sensor
    ara::com::Future<CalibrationResult> Calibrate(
        CalibrationParams const& params);

    // Field: current radar status (get/set/notify)
    ara::com::Field<RadarStatus> status;
};

// Consumer side (Proxy)
void consume_radar_data() {
    // Find all instances of RadarService
    auto handles = ara::com::FindService<RadarServiceProxy>(
        ara::com::InstanceSpecifier("radar_front"));

    if (handles.empty()) {
        ara::log::LogError() << "No radar service found";
        return;
    }

    // Create proxy from first handle
    RadarServiceProxy proxy(handles[0]);

    // Subscribe to events
    proxy.object_detected.Subscribe(10);  // queue depth
    proxy.object_detected.SetReceiveHandler([](auto& samples) {
        for (auto& obj : samples) {
            process_radar_object(*obj);
        }
    });

    // Call a method (returns Future)
    auto future = proxy.Calibrate(CalibrationParams{.mode = Mode::FULL});
    auto result = future.get();  // blocks until response
    if (result.success) {
        ara::log::LogInfo() << "Radar calibrated OK";
    }
}
```

### 2.4 ara::exec — Deterministic Execution

The Execution Manager controls when and how processes run. In mission-critical systems, you need **deterministic scheduling** — the same inputs always produce the same outputs in the same time.

```cpp
#include <ara/exec/execution_client.h>
#include <ara/exec/deterministic_client.h>

int main() {
    // Report to Execution Manager that we're alive
    ara::exec::ExecutionClient exec_client;
    exec_client.ReportExecutionState(
        ara::exec::ExecutionState::kRunning);

    // Deterministic execution: wait for activation cycle
    ara::exec::DeterministicClient det_client;

    while (true) {
        // Block until the Execution Manager triggers this cycle
        auto activation = det_client.WaitForActivation();

        if (activation == ara::exec::ActivationReturnType::kTerminate) {
            break;  // graceful shutdown requested
        }

        // This code runs at a guaranteed period (e.g., every 10ms)
        // WCET must be bounded and verified
        read_sensors();
        compute_control();
        write_actuators();
    }

    exec_client.ReportExecutionState(
        ara::exec::ExecutionState::kTerminating);
    return 0;
}
```

### 2.5 ara::phm — Health Monitoring

Mission-critical services must be supervised. If a service misses its deadline, the platform must react — escalating from restart to safe state.

```cpp
#include <ara/phm/supervised_entity.h>

class BrakeController {
    ara::phm::SupervisedEntity supervised_;
    
public:
    BrakeController()
        : supervised_(ara::com::InstanceSpecifier("brake_ctrl")) {}

    void control_cycle() {
        // Report alive checkpoint — must happen within configured deadline
        supervised_.ReportCheckpoint(
            ara::phm::Checkpoint::kAliveCheckpoint);

        // --- Begin safety-critical computation ---
        auto brake_request = read_brake_pedal();
        auto wheel_speeds = read_wheel_speeds();
        auto brake_torque = compute_abs(brake_request, wheel_speeds);
        apply_brake_torque(brake_torque);
        // --- End safety-critical computation ---

        // Report logical checkpoint — confirms correct execution flow
        supervised_.ReportCheckpoint(
            ara::phm::Checkpoint::kLogicalCheckpoint);
    }
};
```

**Supervision modes:**
- **Alive supervision**: "Are you still running?" — periodic heartbeat
- **Deadline supervision**: "Did you finish in time?" — WCET enforcement
- **Logical supervision**: "Did you execute correctly?" — checkpoint sequence verification

---

## 3. Functional Safety — ISO 26262 for C++ Developers

### 3.1 ASIL Levels and What They Mean for Code

ISO 26262 defines Automotive Safety Integrity Levels:

| ASIL | Risk | Example Systems | Code Requirements |
|---|---|---|---|
| **QM** | No safety requirement | Radio, seat memory | Standard quality |
| **ASIL A** | Low risk | Headlights, wipers | Basic coding guidelines |
| **ASIL B** | Medium risk | Cruise control, EPB | Unit testing, code review |
| **ASIL C** | High risk | ABS, ESC | MC/DC coverage, static analysis |
| **ASIL D** | Highest risk | Steering, airbag, braking | All of the above + formal methods |

### 3.2 MISRA C++ 2023 — The Coding Standard

MISRA C++ 2023 (based on C++17) replaces the ancient MISRA C++ 2008. Key rules for SDV:

```cpp
// === RULE: No implicit conversions that lose data ===
// VIOLATION:
double sensor_value = 3.14159;
int quantized = sensor_value;  // silent truncation!

// COMPLIANT:
int quantized = static_cast<int>(sensor_value);  // explicit intent

// === RULE: No dynamic_cast in safety-critical paths ===
// VIOLATION:
void process(SensorBase* s) {
    if (auto* lidar = dynamic_cast<LidarSensor*>(s)) {  // RTTI overhead
        lidar->get_pointcloud();
    }
}

// COMPLIANT: Use static polymorphism or variant
using Sensor = std::variant<LidarSensor, RadarSensor, CameraSensor>;
void process(Sensor& s) {
    std::visit([](auto& sensor) { sensor.process(); }, s);
}

// === RULE: All switch cases must be handled ===
// VIOLATION:
enum class GearState { PARK, REVERSE, NEUTRAL, DRIVE };
void handle_gear(GearState g) {
    switch (g) {
        case GearState::PARK: park(); break;
        case GearState::DRIVE: drive(); break;
        // Missing REVERSE and NEUTRAL!
    }
}

// COMPLIANT:
void handle_gear(GearState g) {
    switch (g) {
        case GearState::PARK:    park(); break;
        case GearState::REVERSE: reverse(); break;
        case GearState::NEUTRAL: neutral(); break;
        case GearState::DRIVE:   drive(); break;
        // No default — compiler warns on missing enum values
    }
}
```

### 3.3 MC/DC Coverage — The Gold Standard

Modified Condition/Decision Coverage (MC/DC) requires that every condition in a decision independently affects the outcome. Required for ASIL C/D.

```cpp
// Decision: (A && B) || C
// MC/DC requires test cases proving each condition matters independently:
//
// Test 1: A=T, B=T, C=F → true  (baseline for A and B)
// Test 2: A=F, B=T, C=F → false (A independently affects result)
// Test 3: A=T, B=F, C=F → false (B independently affects result)
// Test 4: A=F, B=F, C=T → true  (C independently affects result)
//
// 4 test cases for 3 conditions (vs 8 for exhaustive, 2 for statement coverage)

bool should_apply_emergency_brake(bool obstacle_detected,
                                    bool closing_fast,
                                    bool driver_unresponsive) {
    return (obstacle_detected && closing_fast) || driver_unresponsive;
}

// MC/DC test suite:
static_assert(should_apply_emergency_brake(true, true, false) == true);
static_assert(should_apply_emergency_brake(false, true, false) == false);
static_assert(should_apply_emergency_brake(true, false, false) == false);
static_assert(should_apply_emergency_brake(false, false, true) == true);
```

### 3.4 Freedom from Interference (FFI)

ASIL D code (braking) must not be corrupted by QM code (infotainment). This is enforced through:

1. **Memory Protection Units (MPU)**: Hardware isolation between partitions
2. **Time Partitioning**: Hypervisor guarantees CPU time per partition
3. **Communication Firewalls**: Only validated messages cross partition boundaries

```cpp
// === Partition Boundary — validated message passing ===
// An ASIL D brake service receives speed from an ASIL B sensor.
// The message MUST be validated before use.

struct SpeedMessage {
    float speed_mps;
    uint32_t sequence;
    uint32_t crc32;
    uint64_t timestamp_ns;
};

// E2E Protection (AUTOSAR E2E Profile 4)
bool validate_speed_message(SpeedMessage const& msg,
                             uint32_t expected_seq) {
    // 1. CRC check — detect bit flips
    if (compute_crc32(&msg, offsetof(SpeedMessage, crc32)) != msg.crc32) {
        return false;
    }
    // 2. Sequence check — detect message loss or repetition
    if (msg.sequence != expected_seq) {
        return false;
    }
    // 3. Timeout check — detect stale data
    auto now = get_monotonic_time_ns();
    if (now - msg.timestamp_ns > MAX_AGE_NS) {
        return false;
    }
    // 4. Plausibility check — detect sensor fault
    if (msg.speed_mps < -1.0f || msg.speed_mps > 100.0f) {
        return false;  // physically impossible
    }
    return true;
}
```

---

## 4. AUTOSAR E2E Protection — Defending Data in Transit

### 4.1 Why Every Message Needs a Bodyguard

In a vehicle, messages travel over shared buses and Ethernet. Faults include:
- **Bit flips** (EMI, cosmic rays)
- **Message loss** (bus overload, gateway crash)
- **Message repetition** (stuck sender, replay attack)
- **Message reordering** (gateway buffering)
- **Message delay** (network congestion)
- **Message insertion** (compromised node)

AUTOSAR E2E Protection adds a safety wrapper to every message:

```cpp
// E2E Profile 4 (most common in Adaptive AUTOSAR)
struct E2EHeader {
    uint32_t data_id;       // unique message type identifier
    uint16_t length;        // payload length
    uint16_t counter;       // sequence number (0–65535)
    uint32_t crc;           // CRC-32/Ethernet over header + payload
};

class E2EProtector {
    uint16_t tx_counter_ = 0;
    uint16_t rx_expected_ = 0;
    static constexpr uint16_t MAX_DELTA = 5;  // tolerate up to 5 missed

public:
    // Sender side: protect outgoing data
    template<typename T>
    std::vector<uint8_t> protect(uint32_t data_id, T const& payload) {
        E2EHeader header{};
        header.data_id = data_id;
        header.length = sizeof(T);
        header.counter = tx_counter_++;

        // Serialize header + payload
        std::vector<uint8_t> buffer(sizeof(header) + sizeof(T));
        std::memcpy(buffer.data(), &header, sizeof(header));
        std::memcpy(buffer.data() + sizeof(header), &payload, sizeof(T));

        // Compute CRC over everything except CRC field itself
        header.crc = crc32_ethernet(buffer.data(),
                                      offsetof(E2EHeader, crc));
        std::memcpy(buffer.data() + offsetof(E2EHeader, crc),
                     &header.crc, sizeof(header.crc));

        return buffer;
    }

    // Receiver side: verify incoming data
    enum class Status { OK, WRONG_CRC, LOST, REPEATED, NO_NEW_DATA };

    template<typename T>
    std::pair<Status, T> check(std::vector<uint8_t> const& buffer,
                                uint32_t expected_data_id) {
        T payload{};
        if (buffer.size() < sizeof(E2EHeader) + sizeof(T)) {
            return {Status::WRONG_CRC, payload};
        }

        E2EHeader header{};
        std::memcpy(&header, buffer.data(), sizeof(header));

        // Verify data_id
        if (header.data_id != expected_data_id) {
            return {Status::WRONG_CRC, payload};
        }

        // Verify CRC
        uint32_t stored_crc = header.crc;
        header.crc = 0;
        // ... (recompute and compare)

        // Check counter for loss/repetition
        int16_t delta = static_cast<int16_t>(
            header.counter - rx_expected_);
        if (delta == 0) {
            // Perfect — expected counter
            rx_expected_ = header.counter + 1;
        } else if (delta > 0 && delta <= MAX_DELTA) {
            // Lost some messages but within tolerance
            rx_expected_ = header.counter + 1;
            return {Status::LOST, payload};
        } else if (delta < 0) {
            return {Status::REPEATED, payload};
        }

        std::memcpy(&payload, buffer.data() + sizeof(E2EHeader),
                     sizeof(T));
        return {Status::OK, payload};
    }
};
```

---

## 5. Service-Oriented Communication — SOME/IP and DDS

### 5.1 SOME/IP — Scalable Service-Oriented Middleware over IP

SOME/IP is the dominant SDV middleware. It provides:
- **Service discovery**: Find services on the network dynamically
- **Request/Response**: RPC-style method calls
- **Fire & Forget**: One-way notifications
- **Events**: Publish/subscribe with configurable event groups
- **Fields**: Get/set/notify for service state

```cpp
// === SOME/IP Service Discovery ===
// When a service starts, it announces itself via SD (Service Discovery).
// Consumers "find" services by (ServiceID, InstanceID, MajorVersion).
//
// Wire format: UDP multicast on 224.0.0.1:30490 (default)
// SD offers/finds are periodic, with configurable initial delay and
// repetition phase timing.

// Production pattern: vsomeip-based service
#include <vsomeip/vsomeip.hpp>

class SpeedService {
    std::shared_ptr<vsomeip::application> app_;
    static constexpr vsomeip::service_t SERVICE_ID = 0x1234;
    static constexpr vsomeip::instance_t INSTANCE_ID = 0x0001;
    static constexpr vsomeip::event_t EVENT_SPEED = 0x8001;
    static constexpr vsomeip::eventgroup_t EVENTGROUP_SPEED = 0x0001;

public:
    void init() {
        app_ = vsomeip::runtime::get()->create_application("speed_svc");
        app_->init();

        // Offer the service
        app_->offer_service(SERVICE_ID, INSTANCE_ID);

        // Offer event group
        std::set<vsomeip::eventgroup_t> groups = {EVENTGROUP_SPEED};
        app_->offer_event(SERVICE_ID, INSTANCE_ID, EVENT_SPEED,
                           groups, vsomeip::event_type_e::ET_FIELD);
    }

    void publish_speed(float speed_mps) {
        // Serialize speed to payload
        auto payload = vsomeip::runtime::get()->create_payload();
        std::vector<vsomeip::byte_t> data(sizeof(float));
        std::memcpy(data.data(), &speed_mps, sizeof(float));
        payload->set_data(std::move(data));

        // Notify all subscribers
        app_->notify(SERVICE_ID, INSTANCE_ID, EVENT_SPEED, payload);
    }

    void run() { app_->start(); }  // blocks, processes messages
};
```

### 5.2 DDS — Data Distribution Service

DDS (OMG standard) is used in ADAS and autonomous driving. Unlike SOME/IP, DDS provides:
- **QoS policies**: Reliability, durability, deadline, lifespan
- **Decentralized discovery**: No broker, peer-to-peer
- **Type safety**: IDL-defined data types

```cpp
// DDS with Cyclone DDS (Eclipse project, used by ROS 2)
#include <dds/dds.hpp>

// IDL-generated type
struct VehicleSpeed {
    float speed_mps;
    uint64_t timestamp_ns;
    uint8_t quality;  // 0=invalid, 1=estimated, 2=measured
};

// Publisher
void publish_speed_dds() {
    dds::domain::DomainParticipant dp(0);  // domain 0

    dds::topic::Topic<VehicleSpeed> topic(dp, "VehicleSpeed");

    // QoS: Reliable delivery with 50ms deadline
    dds::pub::qos::DataWriterQos qos;
    qos << dds::core::policy::Reliability::Reliable(
              dds::core::Duration::from_millisecs(100))
        << dds::core::policy::Deadline(
              dds::core::Duration::from_millisecs(50));

    dds::pub::DataWriter<VehicleSpeed> writer(
        dds::pub::Publisher(dp), topic, qos);

    VehicleSpeed msg{.speed_mps = 13.5f,
                      .timestamp_ns = get_monotonic_ns(),
                      .quality = 2};
    writer.write(msg);
}
```

### 5.3 SOME/IP vs DDS — When to Use Which

| Feature | SOME/IP | DDS |
|---|---|---|
| **Primary use** | Classic vehicle services | ADAS, autonomous driving |
| **Discovery** | Centralized SD | Decentralized (SPDP/SEDP) |
| **QoS** | Basic (reliable/unreliable) | Rich (22+ QoS policies) |
| **Serialization** | Custom (SOME/IP-SD) | CDR (CORBA standard) |
| **Latency** | ~100µs (UDP) | ~50µs (shared memory) |
| **Tooling** | vsomeip, CommonAPI | Cyclone DDS, FastDDS, RTI |
| **Standard** | AUTOSAR | OMG |
| **ROS 2 integration** | Via bridge | Native (rmw_cyclonedds) |

---

## 6. OTA Updates — Updating Mission-Critical Software Safely

### 6.1 The A/B Partition Strategy

You cannot update running software in-place. The vehicle might lose power mid-update. The solution is A/B partitioning:

```
┌──────────────────────────────────────────────┐
│                   Storage                     │
│  ┌──────────────┐  ┌──────────────┐          │
│  │  Partition A  │  │  Partition B  │         │
│  │  (Active)     │  │  (Standby)    │         │
│  │  v2.3.1       │  │  v2.4.0       │         │
│  │  RUNNING      │  │  READY        │         │
│  └──────────────┘  └──────────────┘          │
│                                               │
│  Boot selector: → A (current) → B (on reboot)│
└──────────────────────────────────────────────┘
```

**Update flow:**
1. Download new software to standby partition (B)
2. Verify integrity (hash chain, code signature)
3. Mark B as "pending"
4. At next safe opportunity (vehicle parked, engine off), reboot into B
5. B runs self-tests; if pass → mark B as "active"
6. If B fails self-test → rollback to A automatically

### 6.2 OTA Update Manager in C++

```cpp
// Simplified AUTOSAR UCM (Update and Configuration Management)
class UpdateManager {
    enum class State {
        IDLE,
        DOWNLOADING,
        VERIFYING,
        READY,
        ACTIVATING,
        ROLLING_BACK
    };
    
    State state_ = State::IDLE;
    std::string active_partition_ = "A";
    std::string standby_partition_ = "B";

public:
    // Step 1: Download package to standby partition
    Result download(std::string_view url,
                    std::string_view expected_hash) {
        assert(state_ == State::IDLE);
        state_ = State::DOWNLOADING;

        auto data = fetch_package(url);
        if (!data) {
            state_ = State::IDLE;
            return Result::DOWNLOAD_FAILED;
        }

        // Verify cryptographic hash
        auto actual_hash = sha256(data.value());
        if (actual_hash != expected_hash) {
            state_ = State::IDLE;
            return Result::HASH_MISMATCH;
        }

        // Verify code signature (public key embedded in hardware)
        if (!verify_signature(data.value(), get_oem_public_key())) {
            state_ = State::IDLE;
            return Result::SIGNATURE_INVALID;
        }

        write_to_partition(standby_partition_, data.value());
        state_ = State::VERIFYING;
        return Result::OK;
    }

    // Step 2: Verify the written partition
    Result verify() {
        assert(state_ == State::VERIFYING);
        auto readback_hash = hash_partition(standby_partition_);
        if (readback_hash != expected_partition_hash_) {
            state_ = State::IDLE;
            return Result::VERIFY_FAILED;
        }
        state_ = State::READY;
        return Result::OK;
    }

    // Step 3: Activate on next boot (only when vehicle is safe)
    Result activate() {
        assert(state_ == State::READY);
        assert(vehicle_is_parked());     // safety precondition
        assert(!engine_is_running());    // safety precondition

        set_boot_partition(standby_partition_);
        state_ = State::ACTIVATING;
        request_reboot();
        return Result::OK;
    }

    // Step 4: Called by new software after boot
    void confirm_activation() {
        // Self-test passed — mark this partition as good
        mark_partition_good(active_partition_);
        state_ = State::IDLE;
    }

    // Called if self-test fails after boot
    void rollback() {
        state_ = State::ROLLING_BACK;
        set_boot_partition(get_previous_partition());
        mark_partition_bad(active_partition_);
        request_reboot();
    }
};
```

---

## 7. Cybersecurity — ISO/SAE 21434 for SDV

### 7.1 The Threat Landscape

A connected vehicle has multiple attack surfaces:
- **OBD-II port**: Physical access to CAN bus
- **Telematics unit (TCU)**: Cellular modem, remote attack surface
- **Infotainment**: Wi-Fi, Bluetooth, USB, browser
- **V2X**: Vehicle-to-everything communication
- **OTA updates**: Compromised update server
- **Charging ports** (EVs): Power-line communication

### 7.2 Secure Boot Chain

```
┌─────────────────┐
│ Hardware Root    │
│ of Trust (HSM)  │──── OEM Root Key (burned in silicon)
└───────┬─────────┘
        │ Verifies
┌───────┴─────────┐
│ Bootloader      │──── Signed by OEM
│ (ROM/BootROM)   │
└───────┬─────────┘
        │ Verifies
┌───────┴─────────┐
│ Secondary       │──── Signed by OEM
│ Bootloader      │
└───────┬─────────┘
        │ Verifies
┌───────┴─────────┐
│ Hypervisor      │──── Signed by OEM
└───────┬─────────┘
        │ Verifies
┌───────┬─────────┐
│ Linux │ QNX     │──── Signed kernel images
└───────┴─────────┘
        │ Verifies
┌───────┬─────────┐
│ App1  │ App2    │──── Signed application manifests
└───────┴─────────┘
```

### 7.3 SecOC — Secure Onboard Communication

Messages on CAN and Ethernet can be spoofed. SecOC adds authentication:

```cpp
// SecOC: Authenticated CAN/Ethernet message
class SecOCAuthenticator {
    std::array<uint8_t, 16> key_;  // Pre-shared key from HSM
    uint32_t freshness_counter_ = 0;

public:
    // Authenticate outgoing message
    std::vector<uint8_t> authenticate(
            std::span<const uint8_t> payload) {
        ++freshness_counter_;

        // Build data to authenticate: payload + freshness
        std::vector<uint8_t> auth_data;
        auth_data.insert(auth_data.end(),
                          payload.begin(), payload.end());
        auto fc_bytes = to_bytes(freshness_counter_);
        auth_data.insert(auth_data.end(),
                          fc_bytes.begin(), fc_bytes.end());

        // Compute truncated CMAC (AES-128-CMAC, truncated to 4 bytes)
        auto mac = cmac_aes128(key_, auth_data);
        auto truncated_mac = truncate(mac, 4);  // 32-bit MAC

        // Output: payload + truncated freshness + truncated MAC
        std::vector<uint8_t> secured_pdu;
        secured_pdu.insert(secured_pdu.end(),
                            payload.begin(), payload.end());
        secured_pdu.push_back(
            static_cast<uint8_t>(freshness_counter_ & 0xFF));
        secured_pdu.insert(secured_pdu.end(),
                            truncated_mac.begin(),
                            truncated_mac.end());
        return secured_pdu;
    }

    // Verify incoming message
    bool verify(std::span<const uint8_t> secured_pdu,
                size_t payload_len) {
        // Extract payload, freshness hint, MAC
        // Reconstruct full freshness from hint + local counter
        // Recompute CMAC and compare
        // Reject if freshness is too old (replay protection)
        // ... (implementation mirrors authenticate in reverse)
        return true;  // placeholder
    }
};
```

---

## 8. Diagnostics — UDS and Fault Management

### 8.1 UDS (Unified Diagnostic Services — ISO 14229)

Every vehicle must support diagnostics for manufacturing, service, and compliance. UDS defines standardized services:

```cpp
// UDS Service IDs relevant to SDV
enum class UDSService : uint8_t {
    DiagnosticSessionControl   = 0x10,
    ECUReset                   = 0x11,
    SecurityAccess             = 0x27,
    CommunicationControl       = 0x28,
    ReadDataByIdentifier       = 0x22,
    WriteDataByIdentifier      = 0x2E,
    RoutineControl             = 0x31,
    RequestDownload            = 0x34,
    TransferData               = 0x36,
    RequestTransferExit        = 0x37,
    ReadDTCInformation         = 0x19,
    ClearDiagnosticInformation = 0x14,
    ControlDTCSetting          = 0x85
};

// Production DTC (Diagnostic Trouble Code) manager
class DTCManager {
    struct DTCEntry {
        uint32_t dtc_number;          // e.g., 0xC07300 = CAN comm error
        uint8_t  status;              // bit-encoded status mask
        uint64_t first_occurrence_ns;
        uint64_t last_occurrence_ns;
        uint32_t occurrence_count;
        std::array<uint8_t, 32> snapshot; // freeze-frame data
    };

    std::vector<DTCEntry> fault_memory_;
    static constexpr size_t MAX_DTCS = 256;

public:
    // Report a new fault
    void report_fault(uint32_t dtc, std::span<const uint8_t> snapshot) {
        auto it = std::find_if(fault_memory_.begin(),
                                fault_memory_.end(),
                                [dtc](auto& e) {
                                    return e.dtc_number == dtc;
                                });

        auto now = get_monotonic_ns();
        if (it != fault_memory_.end()) {
            // Update existing fault
            it->last_occurrence_ns = now;
            it->occurrence_count++;
            it->status |= 0x01;  // testFailed bit
        } else if (fault_memory_.size() < MAX_DTCS) {
            // New fault
            DTCEntry entry{};
            entry.dtc_number = dtc;
            entry.status = 0x09;  // testFailed + confirmedDTC
            entry.first_occurrence_ns = now;
            entry.last_occurrence_ns = now;
            entry.occurrence_count = 1;
            std::copy_n(snapshot.data(),
                         std::min(snapshot.size(), size_t(32)),
                         entry.snapshot.begin());
            fault_memory_.push_back(entry);
        }
    }

    // Clear all DTCs (UDS service 0x14)
    void clear_all() {
        fault_memory_.clear();
    }
};
```

---

## 9. Graceful Degradation — When Things Go Wrong

### 9.1 The Degradation State Machine

Mission-critical systems must not just crash — they must degrade gracefully. A self-driving car with a failed lidar should not slam to a halt at highway speed.

```cpp
// Vehicle-level degradation manager
enum class DegradationLevel {
    FULL_AUTONOMY,       // L4: all sensors and systems nominal
    LIMITED_AUTONOMY,    // Reduced sensor set, lower speed
    DRIVER_ASSIST,       // L2: driver must monitor
    MANUAL_CONTROL,      // All autonomy disabled, driver in control
    MINIMAL_RISK,        // Pull over and stop safely
    SAFE_STOP            // Emergency stop, hazard lights, unlock doors
};

class DegradationManager {
    DegradationLevel current_ = DegradationLevel::FULL_AUTONOMY;

    // Sensor health tracking
    struct SensorStatus {
        bool lidar_front = true;
        bool lidar_rear = true;
        bool radar_front = true;
        bool camera_front = true;
        bool camera_surround = true;
        bool gnss = true;
        bool imu = true;
    };

    SensorStatus sensors_;

public:
    DegradationLevel evaluate() {
        int critical_failures = 0;
        int redundancy_lost = 0;

        if (!sensors_.lidar_front && !sensors_.radar_front) {
            // Forward perception lost entirely
            return DegradationLevel::SAFE_STOP;
        }

        if (!sensors_.lidar_front || !sensors_.radar_front) {
            // Lost one forward sensor — still have redundancy
            redundancy_lost++;
        }

        if (!sensors_.imu) {
            // IMU is single-point-of-failure for localization
            return DegradationLevel::MINIMAL_RISK;
        }

        if (!sensors_.camera_front) {
            // Can't read signs, traffic lights
            return DegradationLevel::DRIVER_ASSIST;
        }

        if (redundancy_lost >= 2) {
            return DegradationLevel::LIMITED_AUTONOMY;
        }

        return DegradationLevel::FULL_AUTONOMY;
    }

    // Transition with logging and notification
    void transition(DegradationLevel new_level) {
        if (new_level == current_) return;

        // Degradation can only go DOWN (more restrictive), never UP
        // without explicit operator/system confirmation
        if (static_cast<int>(new_level) < static_cast<int>(current_)) {
            log_warning("Attempted upgrade from {} to {} — blocked",
                         current_, new_level);
            return;
        }

        log_critical("Degradation: {} → {}", current_, new_level);
        notify_driver(new_level);
        notify_fleet_management(new_level);
        current_ = new_level;
    }
};
```

### 9.2 Fail-Operational vs Fail-Safe

| Mode | Behavior | Example |
|---|---|---|
| **Fail-safe** | System enters safe state on failure | Airbag ECU: fire → safe state (deployed) |
| **Fail-operational** | System continues in degraded mode | Steering: one motor fails → second motor takes over |
| **Fail-silent** | System stops producing output | Sensor: bad data → stop publishing (don't corrupt downstream) |

For SDV, ADAS **must** be fail-operational for L3+. A highway autopilot cannot just "turn off" — the driver may not be paying attention.

---

## 10. Deterministic Execution and Real-Time Patterns

### 10.1 Cyclic Execution Pattern

Mission-critical SDV software runs in fixed-period cycles (typically 10ms for vehicle dynamics, 33ms for perception, 100ms for planning).

```cpp
// Production cyclic execution with jitter monitoring
class CyclicTask {
    std::chrono::nanoseconds period_;
    std::chrono::nanoseconds wcet_;       // worst-case execution time
    std::chrono::nanoseconds max_jitter_;  // max observed jitter
    uint64_t overrun_count_ = 0;

public:
    CyclicTask(std::chrono::milliseconds period,
               std::chrono::microseconds budget)
        : period_(period)
        , wcet_(budget) {}

    void run() {
        // Pin to CPU core (avoid migration)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(2, &cpuset);  // core 2 for safety tasks
        pthread_setaffinity_np(pthread_self(),
                                sizeof(cpuset), &cpuset);

        // Set SCHED_FIFO with high priority
        struct sched_param param;
        param.sched_priority = 90;  // high but below watchdog
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

        // Lock all memory to prevent page faults
        mlockall(MCL_CURRENT | MCL_FUTURE);

        auto next_wake = std::chrono::steady_clock::now();

        while (running_) {
            next_wake += period_;

            auto start = std::chrono::steady_clock::now();

            // --- Deterministic work ---
            read_inputs();
            compute();
            write_outputs();
            // --- End deterministic work ---

            auto end = std::chrono::steady_clock::now();
            auto execution_time = end - start;

            // Monitor for overruns
            if (execution_time > wcet_) {
                ++overrun_count_;
                report_overrun(execution_time);
            }

            // Sleep until next period
            std::this_thread::sleep_until(next_wake);

            // Measure jitter
            auto actual_wake = std::chrono::steady_clock::now();
            auto jitter = actual_wake - next_wake;
            if (jitter > max_jitter_) {
                max_jitter_ = jitter;
            }
        }
    }
};
```

### 10.2 Lock-Free Communication Between Cycles

Different cycle rates need to share data without blocking. Triple-buffering avoids locks:

```cpp
// Lock-free triple buffer for producer-consumer between different cycles
template<typename T>
class TripleBuffer {
    std::array<T, 3> buffers_;
    std::atomic<uint8_t> write_idx_{0};   // producer writes here
    std::atomic<uint8_t> read_idx_{2};    // consumer reads here
    std::atomic<uint8_t> middle_idx_{1};  // swap buffer

public:
    // Producer: write new data, then swap
    void write(T const& data) {
        buffers_[write_idx_.load(std::memory_order_relaxed)] = data;

        // Swap write and middle buffers
        uint8_t w = write_idx_.load(std::memory_order_relaxed);
        uint8_t m = middle_idx_.exchange(w, std::memory_order_acq_rel);
        write_idx_.store(m, std::memory_order_release);
    }

    // Consumer: swap and read latest data
    T const& read() {
        // Swap read and middle buffers
        uint8_t r = read_idx_.load(std::memory_order_relaxed);
        uint8_t m = middle_idx_.exchange(r, std::memory_order_acq_rel);
        read_idx_.store(m, std::memory_order_release);

        return buffers_[read_idx_.load(std::memory_order_acquire)];
    }
};

// Usage: 10ms control cycle produces, 33ms perception cycle consumes
TripleBuffer<VehicleState> state_buffer;

// 10ms cycle (producer)
void control_cycle() {
    VehicleState state = compute_vehicle_state();
    state_buffer.write(state);  // non-blocking
}

// 33ms cycle (consumer)
void perception_cycle() {
    auto const& state = state_buffer.read();  // non-blocking, latest data
    fuse_with_perception(state);
}
```

---

## 11. Production Patterns — Lessons from Industry

### 11.1 The Service Manifest Pattern

Every SDV service needs a machine-readable manifest for the Execution Manager:

```json
{
    "service_name": "emergency_brake_assist",
    "version": "2.4.1",
    "asil_level": "D",
    "execution": {
        "startup_order": 5,
        "scheduling": "FIXED_PRIORITY",
        "priority": 95,
        "cpu_affinity": [2, 3],
        "memory_limit_mb": 64,
        "cycle_time_ms": 10,
        "wcet_ms": 5
    },
    "health_monitoring": {
        "alive_timeout_ms": 50,
        "deadline_ms": 8,
        "max_consecutive_failures": 3,
        "recovery_action": "RESTART_THEN_DEGRADE"
    },
    "dependencies": [
        {"service": "radar_front", "required": true},
        {"service": "camera_front", "required": false},
        {"service": "vehicle_speed", "required": true}
    ],
    "e2e_protection": {
        "profile": "PROFILE_04",
        "data_id_list": [
            {"topic": "radar_objects", "data_id": "0x1234"},
            {"topic": "brake_command", "data_id": "0x5678"}
        ]
    }
}
```

### 11.2 The Watchdog Cascade Pattern

Production vehicles use cascading watchdogs — if one level fails to detect a hang, the next level catches it:

```
┌──────────────────────────────────────┐
│ Level 3: Hardware Watchdog           │
│ (Reset entire SoC if not fed)        │
│ Timeout: 500ms                       │
├──────────────────────────────────────┤
│ Level 2: OS-level Health Monitor     │
│ (Restart crashed processes)          │
│ Timeout: 100ms per process           │
├──────────────────────────────────────┤
│ Level 1: Application Watchdog        │
│ (Detect stuck threads within app)    │
│ Timeout: 20ms per cycle              │
└──────────────────────────────────────┘
```

### 11.3 The Black Box Recorder Pattern

Like an aircraft flight recorder, SDV systems continuously record data for post-incident analysis:

```cpp
class BlackBoxRecorder {
    // Ring buffer: last N seconds of data, constantly overwritten
    static constexpr size_t BUFFER_SIZE = 1024 * 1024 * 128;  // 128 MB
    std::array<uint8_t, BUFFER_SIZE> ring_buffer_;
    size_t write_pos_ = 0;

    // Trigger conditions for saving
    bool triggered_ = false;
    size_t pre_trigger_size_;   // how much data to save before trigger
    size_t post_trigger_size_;  // how much to continue recording after

public:
    void record(std::span<const uint8_t> data) {
        for (auto byte : data) {
            ring_buffer_[write_pos_ % BUFFER_SIZE] = byte;
            ++write_pos_;
        }
    }

    // Called when an incident is detected (collision, AEB activation, etc.)
    void trigger_save(std::string_view reason) {
        if (triggered_) return;  // already triggered
        triggered_ = true;

        // Save pre-trigger data (last 30 seconds)
        // Continue recording post-trigger data (next 10 seconds)
        // Write to persistent storage with timestamp and reason
        save_to_flash(reason, get_pre_trigger_data(),
                       get_post_trigger_data());
    }
};
```

---

## 12. Development Toolchain for SDV

### 12.1 Build System

SDV projects typically use:
- **CMake** with Conan or vcpkg for dependency management
- **Cross-compilation** toolchains (aarch64-linux-gnu for ARM targets)
- **Yocto/Buildroot** for building the entire Linux distribution
- **CI/CD** with hardware-in-the-loop (HiL) testing

### 12.2 Testing Pyramid for SDV

```
         ╱╲
        ╱  ╲  Vehicle-level tests (real car, test track)
       ╱ VT ╲
      ╱──────╲
     ╱  HiL   ╲  Hardware-in-the-loop (real ECU, simulated vehicle)
    ╱──────────╲
   ╱   SiL      ╲  Software-in-the-loop (compiled binary, vehicle model)
  ╱──────────────╲
 ╱  Integration    ╲  Multiple services running together
╱──────────────────╲
╱ Unit Tests         ╲  Individual function/class testing
──────────────────────
```

### 12.3 Static Analysis Tools

| Tool | Purpose | Standard |
|---|---|---|
| **Polyspace** | Prove absence of runtime errors | ISO 26262 qualified |
| **Coverity** | Find bugs in C/C++ at scale | CWE, MISRA |
| **LDRA** | MISRA compliance, MC/DC coverage | ISO 26262 qualified |
| **clang-tidy** | Modern C++ checks, MISRA subset | Open source |
| **cppcheck** | Undefined behavior, leaks | Open source |
| **PVS-Studio** | Advanced pattern detection | Automotive certification |

---

## Summary — Key Takeaways

1. **SDV = software-defined, not software-only** — hardware still matters for safety
2. **AUTOSAR Adaptive** is the de facto middleware standard (C++14/17, POSIX-based)
3. **ISO 26262** dictates your coding rules, testing strategy, and architecture
4. **E2E protection** is mandatory — every message is a potential corruption vector
5. **Graceful degradation** beats crash-and-restart in mission-critical systems
6. **Deterministic execution** requires OS-level support (PREEMPT_RT, CPU pinning, mlockall)
7. **Cybersecurity** (ISO/SAE 21434) is as important as functional safety
8. **OTA updates** need A/B partitions, cryptographic verification, and rollback capability
9. **Triple buffering** and lock-free patterns are essential for multi-rate systems
10. **Test at every level** — from unit tests to hardware-in-the-loop to vehicle tests
