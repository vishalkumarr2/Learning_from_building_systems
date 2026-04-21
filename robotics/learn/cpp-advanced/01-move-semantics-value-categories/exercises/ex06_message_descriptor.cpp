// =============================================================================
// Exercise 6: Compile-Time Message Descriptor — Week 1 Mini-Project
// =============================================================================
// A compile-time message serialization framework:
//   - Message types with compile-time IDs (FNV-1a hash)
//   - MessageRegistry with compile-time ID collision detection
//   - Zero-heap dispatch by ID
//   - Benchmark dispatching 1M messages
//
// Build: g++ -std=c++2a -fconcepts -O2 -Wall -Wextra -Wpedantic ex06_message_descriptor.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <type_traits>

// =============================================================================
// FNV-1a Hash (compile-time string → uint32_t)
// =============================================================================

constexpr uint32_t fnv1a_32(std::string_view sv) {
    uint32_t hash = 0x811c9dc5u;
    for (char c : sv) {
        hash ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
        hash *= 0x01000193u;
    }
    return hash;
}

// =============================================================================
// Message Concept — what every message type must provide
// =============================================================================

template<typename T>
concept Message = requires {
    { T::id };
    requires std::is_convertible_v<decltype(T::id), uint32_t>;
    { T::name };
    requires std::is_convertible_v<decltype(T::name), std::string_view>;
    requires std::is_trivially_copyable_v<T>;
};

// =============================================================================
// Message Types
// =============================================================================

struct ImuMessage {
    static constexpr uint32_t id = fnv1a_32("ImuMessage");
    static constexpr std::string_view name = "ImuMessage";

    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
    uint64_t timestamp_ns;
};

struct OdomMessage {
    static constexpr uint32_t id = fnv1a_32("OdomMessage");
    static constexpr std::string_view name = "OdomMessage";

    float x, y, theta;
    float vx, vy, omega;
    uint64_t timestamp_ns;
};

struct BatteryMessage {
    static constexpr uint32_t id = fnv1a_32("BatteryMessage");
    static constexpr std::string_view name = "BatteryMessage";

    float voltage;
    float current;
    float temperature_c;
    uint8_t state_of_charge;
    uint8_t padding[3];
    uint64_t timestamp_ns;
};

struct HeartbeatMessage {
    static constexpr uint32_t id = fnv1a_32("HeartbeatMessage");
    static constexpr std::string_view name = "HeartbeatMessage";

    uint32_t sequence;
    uint32_t uptime_sec;
    uint64_t timestamp_ns;
};

static_assert(Message<ImuMessage>);
static_assert(Message<OdomMessage>);
static_assert(Message<BatteryMessage>);
static_assert(Message<HeartbeatMessage>);

// =============================================================================
// Compile-Time ID Collision Detection
// =============================================================================

template<uint32_t... Ids>
constexpr bool all_unique() {
    constexpr std::array<uint32_t, sizeof...(Ids)> arr{Ids...};
    for (std::size_t i = 0; i < arr.size(); ++i) {
        for (std::size_t j = i + 1; j < arr.size(); ++j) {
            if (arr[i] == arr[j]) return false;
        }
    }
    return true;
}

// =============================================================================
// MessageRegistry — compile-time validated, zero-heap dispatch
// =============================================================================

template<typename... Msgs>
class MessageRegistry {
    static_assert(all_unique<Msgs::id...>(),
        "Message ID collision detected! Two message types have the same hash.");

    static constexpr std::size_t max_msg_size_impl() {
        std::size_t m = 0;
        std::size_t sizes[] = { sizeof(Msgs)... };
        for (auto s : sizes) {
            if (s > m) m = s;
        }
        return m;
    }

public:
    static constexpr std::size_t message_count = sizeof...(Msgs);
    static constexpr std::size_t max_size = max_msg_size_impl();

    static constexpr std::string_view name_for_id(uint32_t id) {
        std::string_view result = "unknown";
        ((Msgs::id == id ? (result = Msgs::name, true) : false) || ...);
        return result;
    }

    static constexpr bool has_id(uint32_t id) {
        return ((Msgs::id == id) || ...);
    }

    template<typename Handler>
    static bool dispatch(uint32_t id, const void* data, Handler&& handler) {
        bool handled = false;
        (dispatch_one<Msgs>(id, data, handler, handled), ...);
        return handled;
    }

private:
    template<typename Msg, typename Handler>
    static void dispatch_one(uint32_t id, const void* data,
                             Handler& handler, bool& handled) {
        if (!handled && Msg::id == id) {
            Msg msg;
            std::memcpy(&msg, data, sizeof(Msg));
            handler(msg);
            handled = true;
        }
    }
};

// =============================================================================
// Instantiate the registry
// =============================================================================

using RobotMessages = MessageRegistry<
    ImuMessage,
    OdomMessage,
    BatteryMessage,
    HeartbeatMessage
>;

static_assert(RobotMessages::message_count == 4);
static_assert(RobotMessages::has_id(ImuMessage::id));
static_assert(RobotMessages::has_id(OdomMessage::id));
static_assert(!RobotMessages::has_id(0xDEADBEEF));
static_assert(RobotMessages::name_for_id(ImuMessage::id) == "ImuMessage");

// =============================================================================
// Handlers
// =============================================================================

struct PrintHandler {
    void operator()(const ImuMessage& msg) const {
        std::cout << "  IMU: accel=(" << msg.accel_x << "," << msg.accel_y
                  << "," << msg.accel_z << ") ts=" << msg.timestamp_ns << "\n";
    }
    void operator()(const OdomMessage& msg) const {
        std::cout << "  Odom: pos=(" << msg.x << "," << msg.y
                  << "," << msg.theta << ") ts=" << msg.timestamp_ns << "\n";
    }
    void operator()(const BatteryMessage& msg) const {
        std::cout << "  Battery: " << msg.voltage << "V, "
                  << static_cast<int>(msg.state_of_charge) << "% ts="
                  << msg.timestamp_ns << "\n";
    }
    void operator()(const HeartbeatMessage& msg) const {
        std::cout << "  Heartbeat: seq=" << msg.sequence
                  << " uptime=" << msg.uptime_sec << "s\n";
    }
};

struct CountHandler {
    int imu_count = 0;
    int odom_count = 0;
    int battery_count = 0;
    int heartbeat_count = 0;

    void operator()(const ImuMessage&) { ++imu_count; }
    void operator()(const OdomMessage&) { ++odom_count; }
    void operator()(const BatteryMessage&) { ++battery_count; }
    void operator()(const HeartbeatMessage&) { ++heartbeat_count; }
};

// =============================================================================
// Wire format: [uint32_t id][uint32_t size][payload...]
// =============================================================================

struct WireHeader {
    uint32_t msg_id;
    uint32_t payload_size;
};

template<typename Msg>
std::size_t serialize(const Msg& msg, void* buffer, std::size_t buffer_size) {
    constexpr std::size_t total = sizeof(WireHeader) + sizeof(Msg);
    if (buffer_size < total) return 0;

    WireHeader header{Msg::id, sizeof(Msg)};
    auto* dst = static_cast<std::byte*>(buffer);
    std::memcpy(dst, &header, sizeof(header));
    std::memcpy(dst + sizeof(header), &msg, sizeof(msg));
    return total;
}

// =============================================================================
// Benchmark: dispatch 1M messages
// =============================================================================

void benchmark_dispatch() {
    std::cout << "\n--- Benchmark: 1M message dispatch ---\n";

    constexpr int N = 1'000'000;

    alignas(16) std::byte buffer[sizeof(WireHeader) + RobotMessages::max_size];

    ImuMessage imu_msg{1.0f, 2.0f, 9.81f, 0.01f, 0.02f, 0.03f, 1000000};
    OdomMessage odom_msg{1.5f, 2.5f, 0.3f, 0.1f, 0.0f, 0.05f, 2000000};
    BatteryMessage bat_msg{24.5f, 1.2f, 35.0f, 85, {}, 3000000};
    HeartbeatMessage hb_msg{1, 3600, 4000000};

    CountHandler counter{};

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < N; ++i) {
        std::size_t written = 0;
        switch (i % 4) {
            case 0: written = serialize(imu_msg, buffer, sizeof(buffer)); break;
            case 1: written = serialize(odom_msg, buffer, sizeof(buffer)); break;
            case 2: written = serialize(bat_msg, buffer, sizeof(buffer)); break;
            case 3: written = serialize(hb_msg, buffer, sizeof(buffer)); break;
        }
        assert(written > 0);

        WireHeader header;
        std::memcpy(&header, buffer, sizeof(header));
        const void* payload = buffer + sizeof(WireHeader);

        bool ok = RobotMessages::dispatch(header.msg_id, payload, counter);
        assert(ok);
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    std::cout << "  Total: " << us << " us for " << N << " dispatches\n";
    std::cout << "  Per message: " << (static_cast<double>(us) / N * 1000) << " ns\n";
    std::cout << "  Counts: imu=" << counter.imu_count
              << " odom=" << counter.odom_count
              << " bat=" << counter.battery_count
              << " hb=" << counter.heartbeat_count << "\n";

    assert(counter.imu_count == N / 4);
    assert(counter.odom_count == N / 4);
    assert(counter.battery_count == N / 4);
    assert(counter.heartbeat_count == N / 4);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Compile-Time Message Descriptor ===\n";

    std::cout << "\n--- Registry ---\n";
    std::cout << "  Message count: " << RobotMessages::message_count << "\n";
    std::cout << "  Max message size: " << RobotMessages::max_size << " bytes\n";
    std::cout << "  ImuMessage     id=0x" << std::hex << ImuMessage::id
              << " size=" << std::dec << sizeof(ImuMessage) << "\n";
    std::cout << "  OdomMessage    id=0x" << std::hex << OdomMessage::id
              << " size=" << std::dec << sizeof(OdomMessage) << "\n";
    std::cout << "  BatteryMessage id=0x" << std::hex << BatteryMessage::id
              << " size=" << std::dec << sizeof(BatteryMessage) << "\n";
    std::cout << "  HeartbeatMsg   id=0x" << std::hex << HeartbeatMessage::id
              << " size=" << std::dec << sizeof(HeartbeatMessage) << "\n";

    std::cout << "\n--- Dispatch Demo ---\n";
    PrintHandler printer;

    ImuMessage imu{1.0f, 2.0f, 9.81f, 0.01f, 0.02f, 0.03f, 1234567890};
    RobotMessages::dispatch(ImuMessage::id, &imu, printer);

    OdomMessage odom{1.5f, 2.5f, 0.785f, 0.1f, 0.0f, 0.05f, 1234567891};
    RobotMessages::dispatch(OdomMessage::id, &odom, printer);

    BatteryMessage bat{24.5f, 1.2f, 35.0f, 85, {}, 1234567892};
    RobotMessages::dispatch(BatteryMessage::id, &bat, printer);

    HeartbeatMessage hb{42, 3600, 1234567893};
    RobotMessages::dispatch(HeartbeatMessage::id, &hb, printer);

    bool ok = RobotMessages::dispatch(0xDEADBEEF, nullptr, printer);
    std::cout << "  Unknown ID dispatch: " << (ok ? "handled" : "not handled") << "\n";
    assert(!ok);

    std::cout << "\n--- Serialization ---\n";
    alignas(16) std::byte wire[128];
    std::size_t n = serialize(imu, wire, sizeof(wire));
    std::cout << "  Serialized ImuMessage: " << n << " bytes\n";

    WireHeader hdr;
    std::memcpy(&hdr, wire, sizeof(hdr));
    std::cout << "  Wire header: id=0x" << std::hex << hdr.msg_id
              << " size=" << std::dec << hdr.payload_size << "\n";

    std::cout << "  Dispatching from wire format:\n";
    RobotMessages::dispatch(hdr.msg_id, wire + sizeof(WireHeader), printer);

    benchmark_dispatch();

    std::cout << "\nAll message descriptor tests passed!\n";
    return 0;
}
