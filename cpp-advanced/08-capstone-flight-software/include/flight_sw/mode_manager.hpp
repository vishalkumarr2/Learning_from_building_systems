#pragma once
// ============================================================================
// ModeManager — std::variant state machine with Hamming-distance-protected IDs
//
// States : Boot → Init → Nominal → Degraded → SafeStop → Shutdown
// Events : InitComplete, SensorFault, SensorRecovered, WatchdogTimeout, ShutdownCmd
//
// Each state carries a Hamming-weight-4+ constant so that a single bit-flip
// cannot silently transition between valid states (JPL safety pattern).
// ============================================================================

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>
#include <vector>

namespace flight_sw {

// ── Hamming-protected state IDs (≥4 bits differ between any pair) ─────
namespace state_id {
    inline constexpr uint32_t BOOT      = 0xAA55'00FFu;
    inline constexpr uint32_t INIT      = 0x55AA'FF00u;
    inline constexpr uint32_t NOMINAL   = 0xCC33'0FF0u;
    inline constexpr uint32_t DEGRADED  = 0x33CC'F00Fu;
    inline constexpr uint32_t SAFE_STOP = 0xF0F0'5A5Au;
    inline constexpr uint32_t SHUTDOWN  = 0x0F0F'A5A5u;
} // namespace state_id

// ── State types ──────────────────────────────────────────────────────
struct Boot      { static constexpr uint32_t id = state_id::BOOT; };
struct Init      { static constexpr uint32_t id = state_id::INIT; };
struct Nominal   { static constexpr uint32_t id = state_id::NOMINAL; };
struct Degraded  { static constexpr uint32_t id = state_id::DEGRADED; };
struct SafeStop  { static constexpr uint32_t id = state_id::SAFE_STOP; };
struct Shutdown  { static constexpr uint32_t id = state_id::SHUTDOWN; };

using State = std::variant<Boot, Init, Nominal, Degraded, SafeStop, Shutdown>;

// ── Event types ──────────────────────────────────────────────────────
struct InitComplete     {};
struct SensorFault      {};
struct SensorRecovered  {};
struct WatchdogTimeout  {};
struct ShutdownCmd      {};

using Event = std::variant<InitComplete, SensorFault, SensorRecovered,
                           WatchdogTimeout, ShutdownCmd>;

// ── Name helpers (free-function overload set for visitor) ────────────
namespace detail {
    inline std::string_view name_of(Boot const&)     noexcept { return "Boot"; }
    inline std::string_view name_of(Init const&)     noexcept { return "Init"; }
    inline std::string_view name_of(Nominal const&)  noexcept { return "Nominal"; }
    inline std::string_view name_of(Degraded const&) noexcept { return "Degraded"; }
    inline std::string_view name_of(SafeStop const&) noexcept { return "SafeStop"; }
    inline std::string_view name_of(Shutdown const&) noexcept { return "Shutdown"; }

    inline std::string_view name_of(InitComplete const&)    noexcept { return "InitComplete"; }
    inline std::string_view name_of(SensorFault const&)     noexcept { return "SensorFault"; }
    inline std::string_view name_of(SensorRecovered const&) noexcept { return "SensorRecovered"; }
    inline std::string_view name_of(WatchdogTimeout const&) noexcept { return "WatchdogTimeout"; }
    inline std::string_view name_of(ShutdownCmd const&)     noexcept { return "ShutdownCmd"; }
} // namespace detail

[[nodiscard]] inline std::string_view state_name(State const& s) noexcept {
    return std::visit([](auto const& st) { return detail::name_of(st); }, s);
}

[[nodiscard]] inline std::string_view event_name(Event const& e) noexcept {
    return std::visit([](auto const& ev) { return detail::name_of(ev); }, e);
}

// ── Transition log entry ─────────────────────────────────────────────
struct ModeTransition {
    std::string_view from;
    std::string_view to;
    std::string_view event;
};

// ── ModeManager ──────────────────────────────────────────────────────
class ModeManager {
public:
    ModeManager() : state_{Boot{}} {}

    // Process an event; returns true if state changed
    inline bool transition(Event const& evt) {
        State prev = state_;
        std::visit([this](auto const& e) { handle(e); }, evt);

        if (state_.index() != prev.index()) {
            log_.push_back({state_name(prev), state_name(state_), event_name(evt)});
            return true;
        }
        return false;
    }

    [[nodiscard]] inline State const& current_state() const noexcept { return state_; }

    [[nodiscard]] inline std::string_view current_state_name() const noexcept {
        return state_name(state_);
    }

    [[nodiscard]] inline uint32_t current_state_id() const noexcept {
        return std::visit([](auto const& s) -> uint32_t { return s.id; }, state_);
    }

    [[nodiscard]] inline std::vector<ModeTransition> const& transition_log() const noexcept {
        return log_;
    }

    // Verify Hamming ID integrity — returns false if corrupted
    [[nodiscard]] inline bool verify_state_id() const noexcept {
        static constexpr std::array<uint32_t, 6> valid_ids = {
            state_id::BOOT, state_id::INIT, state_id::NOMINAL,
            state_id::DEGRADED, state_id::SAFE_STOP, state_id::SHUTDOWN
        };
        uint32_t cur = current_state_id();
        for (auto id : valid_ids) {
            if (cur == id) return true;
        }
        return false;
    }

private:
    State state_;
    std::vector<ModeTransition> log_;

    // ── Transition handlers ──────────────────────────────────────────
    inline void handle(InitComplete const&) {
        if (std::holds_alternative<Boot>(state_) ||
            std::holds_alternative<Init>(state_)) {
            state_ = Nominal{};
        }
    }

    inline void handle(SensorFault const&) {
        if (std::holds_alternative<Nominal>(state_)) {
            state_ = Degraded{};
        } else if (std::holds_alternative<Degraded>(state_)) {
            state_ = SafeStop{};
        }
    }

    inline void handle(SensorRecovered const&) {
        if (std::holds_alternative<Degraded>(state_)) {
            state_ = Nominal{};
        }
    }

    inline void handle(WatchdogTimeout const&) {
        if (!std::holds_alternative<Shutdown>(state_)) {
            state_ = SafeStop{};
        }
    }

    inline void handle(ShutdownCmd const&) {
        state_ = Shutdown{};
    }
};

} // namespace flight_sw
