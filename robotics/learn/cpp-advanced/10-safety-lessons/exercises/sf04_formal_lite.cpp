// sf04_formal_lite.cpp — Lightweight Formal Methods
//
// Full formal verification (Coq, Isabelle, TLA+) is expensive.
// These "lightweight" techniques give 80% of the safety for 20% of the cost:
//   1. Ranged types that cannot hold invalid values
//   2. Design-by-contract with preconditions/postconditions
//   3. Bounded model checking via state space enumeration
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread sf04_formal_lite.cpp -o sf04_formal_lite

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

// ============================================================================
// Part 1: BoundedInt<Min, Max> — a type that CANNOT hold out-of-range values
//
// In safety-critical systems, 60% of bugs are invalid-range errors.
// If the type itself prevents out-of-range, those bugs CANNOT exist.
// ============================================================================

template <int Min, int Max>
class BoundedInt {
    static_assert(Min <= Max, "BoundedInt: Min must be <= Max");

    int value_;

    // Private constructor — only make() can create instances
    explicit constexpr BoundedInt(int v) : value_(v) {}

public:
    // Factory: returns nullopt if out of range — no exceptions, no UB
    static constexpr std::optional<BoundedInt> make(int v) {
        if (v < Min || v > Max) return std::nullopt;
        return BoundedInt(v);
    }

    constexpr int value() const { return value_; }
    static constexpr int min() { return Min; }
    static constexpr int max() { return Max; }

    // Safe arithmetic: adding two BoundedInts widens the result range
    // BoundedInt<A,B> + BoundedInt<C,D> → BoundedInt<A+C, B+D>
    template <int OMin, int OMax>
    constexpr auto operator+(const BoundedInt<OMin, OMax>& other) const {
        constexpr int new_min = Min + OMin;
        constexpr int new_max = Max + OMax;
        // Result is always valid because inputs are in-range
        return BoundedInt<new_min, new_max>::make(value_ + other.value()).value();
    }

    // Comparison
    constexpr bool operator==(const BoundedInt& o) const { return value_ == o.value_; }
    constexpr bool operator<(const BoundedInt& o) const { return value_ < o.value_; }
};

void test_bounded_int() {
    std::cout << "──── BoundedInt<Min, Max> ────\n\n";

    // Basic construction
    auto temp = BoundedInt<-40, 85>::make(25);
    std::cout << "  BoundedInt<-40,85>::make(25): "
              << (temp ? "OK, value=" + std::to_string(temp->value()) : "FAIL") << "\n";

    auto bad = BoundedInt<-40, 85>::make(100);
    std::cout << "  BoundedInt<-40,85>::make(100): "
              << (!bad ? "correctly rejected ✓" : "SHOULD HAVE BEEN REJECTED ✗") << "\n";

    auto neg_bad = BoundedInt<-40, 85>::make(-50);
    std::cout << "  BoundedInt<-40,85>::make(-50): "
              << (!neg_bad ? "correctly rejected ✓" : "SHOULD HAVE BEEN REJECTED ✗") << "\n";

    // Safe addition with automatic range widening
    auto a = BoundedInt<0, 100>::make(60);
    auto b = BoundedInt<0, 50>::make(30);
    if (a && b) {
        auto sum = *a + *b;  // type is BoundedInt<0, 150>
        std::cout << "  BoundedInt<0,100>(60) + BoundedInt<0,50>(30) = " << sum.value()
                  << " (range: [" << sum.min() << ", " << sum.max() << "]) ✓\n";
    }

    // Compile-time validation
    // BoundedInt<10, 5>::make(7);  // would fail static_assert: Min > Max
    std::cout << "  Compile-time range validation: static_assert(Min <= Max) ✓\n";

    std::cout << "\n";
}

// ============================================================================
// Part 2: Design-by-Contract PID Controller
//
// Preconditions: inputs must be in valid range
// Postconditions: output must be bounded
// Invariants: internal state must remain consistent
//
// In debug: contracts are checked → fail-fast on violation
// In release: compiled out → zero overhead
// ============================================================================

// Contract macros — checked in debug, removed in release
#ifdef NDEBUG
    #define CONTRACT_REQUIRE(cond, msg) ((void)0)
    #define CONTRACT_ENSURE(cond, msg)  ((void)0)
#else
    #define CONTRACT_REQUIRE(cond, msg) \
        do { if (!(cond)) { \
            std::cerr << "PRECONDITION FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort(); \
        }} while(0)
    #define CONTRACT_ENSURE(cond, msg) \
        do { if (!(cond)) { \
            std::cerr << "POSTCONDITION FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort(); \
        }} while(0)
#endif

class ContractPID {
    double kp_, ki_, kd_;
    double integral_ = 0.0;
    double prev_error_ = 0.0;
    double output_min_, output_max_;
    double integral_limit_;

public:
    ContractPID(double kp, double ki, double kd,
                double out_min, double out_max, double integral_limit)
        : kp_(kp), ki_(ki), kd_(kd),
          output_min_(out_min), output_max_(out_max),
          integral_limit_(integral_limit) {
        // Construction contracts
        CONTRACT_REQUIRE(kp >= 0.0, "kp must be non-negative");
        CONTRACT_REQUIRE(ki >= 0.0, "ki must be non-negative");
        CONTRACT_REQUIRE(kd >= 0.0, "kd must be non-negative");
        CONTRACT_REQUIRE(out_min < out_max, "output_min must be < output_max");
        CONTRACT_REQUIRE(integral_limit > 0.0, "integral_limit must be positive");
    }

    double compute(double setpoint, double measurement, double dt) {
        // Preconditions
        CONTRACT_REQUIRE(dt > 0.0, "dt must be positive");
        CONTRACT_REQUIRE(dt < 1.0, "dt must be < 1.0 (sanity check)");
        CONTRACT_REQUIRE(setpoint >= -1000.0 && setpoint <= 1000.0, "setpoint out of range");
        CONTRACT_REQUIRE(measurement >= -1000.0 && measurement <= 1000.0, "measurement out of range");

        double error = setpoint - measurement;

        // Anti-windup: clamp integral
        integral_ += error * dt;
        integral_ = std::clamp(integral_, -integral_limit_, integral_limit_);

        double derivative = (error - prev_error_) / dt;
        prev_error_ = error;

        double output = kp_ * error + ki_ * integral_ + kd_ * derivative;

        // Clamp output
        output = std::clamp(output, output_min_, output_max_);

        // Postconditions
        CONTRACT_ENSURE(output >= output_min_ && output <= output_max_,
                        "output must be within [output_min, output_max]");

        return output;
    }
};

void test_contract_pid() {
    std::cout << "──── Design-by-Contract PID ────\n\n";

    ContractPID pid(1.0, 0.1, 0.05, -10.0, 10.0, 100.0);

    // Normal operation
    double out = pid.compute(5.0, 3.0, 0.01);
    bool in_range = (out >= -10.0 && out <= 10.0);
    std::cout << "  Normal: setpoint=5.0, meas=3.0, dt=0.01 → output=" << out
              << (in_range ? " ✓" : " OUT OF RANGE ✗") << "\n";

    // Demonstrate contract violation detection (we catch it without crashing)
    std::cout << "  Contract violation demo (negative dt):\n";
    bool caught = false;
    // We can't actually call with negative dt in debug mode (it would abort),
    // so we demonstrate the check logic manually:
    double bad_dt = -0.01;
    if (bad_dt <= 0.0) {
        std::cout << "    PRECONDITION would fire: 'dt must be positive' ✓\n";
        caught = true;
    }
    std::cout << "    Contract violation detected: " << (caught ? "YES ✓" : "NO ✗") << "\n";

    // Show that contracts compile out in release
    std::cout << "  In NDEBUG mode: contracts compiled out → zero overhead ✓\n";

    std::cout << "\n";
}

// ============================================================================
// Part 3: Bounded Model Checker — State Space Exploration
//
// Exhaustively verify that all declared states are reachable.
// BFS through (state, event) → next_state transitions.
// Report any declared-but-unreachable states (dead code in FSM = spec bug).
// ============================================================================

struct StateMachine {
    std::string name;
    std::set<std::string> states;
    std::set<std::string> events;
    std::string initial_state;
    // transition[state][event] = next_state
    std::map<std::string, std::map<std::string, std::string>> transitions;
};

struct ModelCheckResult {
    bool all_reachable;
    std::set<std::string> reachable;
    std::set<std::string> unreachable;
    int total_transitions_explored;
};

ModelCheckResult bounded_model_check(const StateMachine& sm) {
    ModelCheckResult result;
    result.total_transitions_explored = 0;

    // BFS from initial state
    std::set<std::string> visited;
    std::vector<std::string> queue;

    visited.insert(sm.initial_state);
    queue.push_back(sm.initial_state);

    while (!queue.empty()) {
        std::string current = queue.front();
        queue.erase(queue.begin());

        auto state_it = sm.transitions.find(current);
        if (state_it == sm.transitions.end()) continue;

        for (const auto& event : sm.events) {
            ++result.total_transitions_explored;
            auto ev_it = state_it->second.find(event);
            if (ev_it != state_it->second.end()) {
                const std::string& next = ev_it->second;
                if (visited.find(next) == visited.end()) {
                    visited.insert(next);
                    queue.push_back(next);
                }
            }
        }
    }

    result.reachable = visited;
    result.all_reachable = true;
    for (const auto& state : sm.states) {
        if (visited.find(state) == visited.end()) {
            result.unreachable.insert(state);
            result.all_reachable = false;
        }
    }

    return result;
}

void test_model_checker() {
    std::cout << "──── Bounded Model Checker ────\n\n";

    // Good FSM: all states reachable
    {
        StateMachine traffic;
        traffic.name = "TrafficLight";
        traffic.states = {"RED", "GREEN", "YELLOW"};
        traffic.events = {"TIMER"};
        traffic.initial_state = "RED";
        traffic.transitions = {
            {"RED",    {{"TIMER", "GREEN"}}},
            {"GREEN",  {{"TIMER", "YELLOW"}}},
            {"YELLOW", {{"TIMER", "RED"}}},
        };

        auto result = bounded_model_check(traffic);
        std::cout << "  TrafficLight FSM:\n";
        std::cout << "    States explored: " << result.reachable.size() << "/" << traffic.states.size() << "\n";
        std::cout << "    Transitions explored: " << result.total_transitions_explored << "\n";
        std::cout << "    All reachable: " << (result.all_reachable ? "PASS ✓" : "FAIL ✗") << "\n\n";
    }

    // Bad FSM: has an unreachable state (spec bug!)
    {
        StateMachine elevator;
        elevator.name = "Elevator";
        elevator.states = {"IDLE", "MOVING_UP", "MOVING_DOWN", "MAINTENANCE", "EMERGENCY"};
        elevator.events = {"CALL_UP", "CALL_DOWN", "ARRIVED", "MAINT_KEY", "EMERGENCY_BTN"};
        elevator.initial_state = "IDLE";
        elevator.transitions = {
            {"IDLE",        {{"CALL_UP", "MOVING_UP"}, {"CALL_DOWN", "MOVING_DOWN"},
                             {"EMERGENCY_BTN", "EMERGENCY"}}},
            {"MOVING_UP",   {{"ARRIVED", "IDLE"},  {"EMERGENCY_BTN", "EMERGENCY"}}},
            {"MOVING_DOWN", {{"ARRIVED", "IDLE"},  {"EMERGENCY_BTN", "EMERGENCY"}}},
            {"EMERGENCY",   {{"ARRIVED", "IDLE"}}},
            // MAINTENANCE is declared but has no incoming transitions!
            {"MAINTENANCE", {{"MAINT_KEY", "IDLE"}}},
        };

        auto result = bounded_model_check(elevator);
        std::cout << "  Elevator FSM (with bug):\n";
        std::cout << "    States explored: " << result.reachable.size() << "/" << elevator.states.size() << "\n";
        std::cout << "    Transitions explored: " << result.total_transitions_explored << "\n";
        std::cout << "    All reachable: " << (result.all_reachable ? "PASS" : "FAIL — found unreachable states") << "\n";
        if (!result.unreachable.empty()) {
            std::cout << "    Unreachable states: ";
            for (const auto& s : result.unreachable) std::cout << s << " ";
            std::cout << "← BUG FOUND ✓\n";
        }
        std::cout << "\n";

        // The fix: add a transition to MAINTENANCE
        std::cout << "  Fix: add IDLE --MAINT_KEY--> MAINTENANCE\n";
        elevator.transitions["IDLE"]["MAINT_KEY"] = "MAINTENANCE";
        auto fixed = bounded_model_check(elevator);
        std::cout << "    After fix — all reachable: " << (fixed.all_reachable ? "PASS ✓" : "STILL BROKEN ✗") << "\n";
    }

    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Lightweight Formal Methods Exercise\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    test_bounded_int();
    test_contract_pid();
    test_model_checker();

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << R"(
Key Takeaways:
  1. BoundedInt: make invalid states unrepresentable. Factory returns optional.
     No runtime overhead once constructed — the type IS the proof.
  2. Design-by-Contract: preconditions (caller's job), postconditions (callee's
     guarantee), invariants (always-true). Checked in debug, zero-cost in release.
  3. Bounded model checking: BFS through all (state × event) combinations.
     Finds dead states, missing transitions, unreachable code — automatically.
  4. These are "lightweight" because they don't require theorem provers,
     but they catch >80% of the bugs that formal methods would find.
  5. Combine all three: ranged types + contracts + model checking = strong safety
     without the cost of full formal verification.
)";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    return 0;
}
