// ex03_variant_state_machine.cpp — Robot state machine with std::variant + std::visit
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread ex03_variant_state_machine.cpp -o ex03
//
// Demonstrates:
// - std::variant as a type-safe tagged union for states
// - std::visit with overloaded lambda visitor for exhaustive matching
// - The "overloaded" helper template (inheriting from multiple lambdas)
// - Compile-time guarantee: if you forget a (State, Event) combination, it won't compile
//   (when the catch-all is removed)

#include <iostream>
#include <string>
#include <variant>
#include <vector>

// ── overloaded{} helper ──────────────────────────────────────────────
// Combines multiple lambdas into one callable via overload resolution.
// C++17 deduction guide makes it work with brace-init.

template <typename... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
// C++17 deduction guide (still needed in C++20 for some compilers)
template <typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// ── States ───────────────────────────────────────────────────────────

struct Idle {};
struct Moving { double gx; double gy; };
struct Rotating { double target; };
struct Error { int code; std::string msg; };

using State = std::variant<Idle, Moving, Rotating, Error>;

// ── Events ───────────────────────────────────────────────────────────

struct Start { double x; double y; };
struct Reached {};
struct Rotate { double angle; };
struct Fault { int code; };

using Event = std::variant<Start, Reached, Rotate, Fault>;

// ── State name helper ────────────────────────────────────────────────

std::string state_name(const State& s) {
    return std::visit(overloaded{
        [](const Idle&)     { return std::string{"Idle"}; },
        [](const Moving& m) { return std::string{"Moving("}
                                + std::to_string(m.gx) + ", "
                                + std::to_string(m.gy) + ")"; },
        [](const Rotating& r) { return std::string{"Rotating("}
                                  + std::to_string(r.target) + ")"; },
        [](const Error& e) { return std::string{"Error("}
                               + std::to_string(e.code) + ": "
                               + e.msg + ")"; },
    }, s);
}

std::string event_name(const Event& e) {
    return std::visit(overloaded{
        [](const Start& s)  { return std::string{"Start("}
                                + std::to_string(s.x) + ", "
                                + std::to_string(s.y) + ")"; },
        [](const Reached&)  { return std::string{"Reached"}; },
        [](const Rotate& r) { return std::string{"Rotate("}
                                + std::to_string(r.angle) + ")"; },
        [](const Fault& f)  { return std::string{"Fault("}
                                + std::to_string(f.code) + ")"; },
    }, e);
}

// ── Transition function ──────────────────────────────────────────────
// Uses std::visit with an overloaded visitor across (State, Event) pairs.
// Every (State, Event) combination is handled — remove the catch-all at the
// bottom and the compiler will force you to handle each one explicitly.

State transition(const State& current, const Event& event) {
    return std::visit(overloaded{

        // Idle + Start → Moving
        [](const Idle&, const Start& s) -> State {
            return Moving{s.x, s.y};
        },

        // Idle + Rotate → Rotating
        [](const Idle&, const Rotate& r) -> State {
            return Rotating{r.angle};
        },

        // Moving + Reached → Idle
        [](const Moving&, const Reached&) -> State {
            return Idle{};
        },

        // Rotating + Reached → Idle
        [](const Rotating&, const Reached&) -> State {
            return Idle{};
        },

        // Error + Fault → new Error (must be before the generic lambdas to
        // resolve ambiguity on GCC 9 where (Error, Fault) matches both
        // (auto&, Fault&) and (Error&, auto&))
        [](const Error&, const Fault& f) -> State {
            return Error{f.code, "fault code " + std::to_string(f.code)};
        },

        // Any non-Error state + Fault → Error
        [](const auto&, const Fault& f) -> State {
            return Error{f.code, "fault code " + std::to_string(f.code)};
        },

        // Error is absorbing — ignore all events except Fault (handled above)
        [](const Error& e, const auto&) -> State {
            return e;
        },

        // Catch-all for invalid transitions — stays in current state.
        // NOTE: Remove this to get a compile error if you forget a case!
        // The compiler will list exactly which (State, Event) pairs are unhandled.
        [&current](const auto&, const auto&) -> State {
            std::cout << "    [!] invalid transition — staying in current state\n";
            return current;
        },

    }, current, event);
}

// ── main ─────────────────────────────────────────────────────────────

int main() {
    std::cout << "Robot State Machine (std::variant + std::visit)\n";
    std::cout << "================================================\n\n";

    // Define a sequence of events to drive the state machine
    std::vector<Event> events = {
        Start{3.0, 4.0},   // Idle → Moving(3,4)
        Reached{},          // Moving → Idle
        Rotate{1.57},       // Idle → Rotating(1.57)
        Reached{},          // Rotating → Idle
        Start{10.0, 0.0},  // Idle → Moving(10,0)
        Fault{42},          // Moving → Error(42)
        Start{1.0, 1.0},   // Error → Error (absorbing)
    };

    State state = Idle{};
    std::cout << "Initial: " << state_name(state) << "\n\n";

    for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& event = events[i];
        std::cout << "Step " << (i + 1) << ": event = " << event_name(event) << "\n";

        State next = transition(state, event);

        std::cout << "    " << state_name(state) << " → " << state_name(next) << "\n\n";
        state = next;
    }

    std::cout << "Final state: " << state_name(state) << "\n";

    // ── Exhaustive matching demo ─────────────────────────────────────
    std::cout << "\n--- Exhaustive match on final state ---\n";
    std::visit(overloaded{
        [](const Idle&)       { std::cout << "Robot is idle.\n"; },
        [](const Moving& m)   { std::cout << "Robot moving to ("
                                           << m.gx << "," << m.gy << ")\n"; },
        [](const Rotating& r) { std::cout << "Robot rotating to "
                                           << r.target << " rad\n"; },
        [](const Error& e)    { std::cout << "Robot in error state: ["
                                           << e.code << "] " << e.msg << "\n"; },
    }, state);

    return 0;
}
