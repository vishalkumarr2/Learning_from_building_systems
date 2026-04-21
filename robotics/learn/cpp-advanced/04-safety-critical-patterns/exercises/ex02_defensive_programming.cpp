// =============================================================================
// Exercise 02: Defensive Programming — Design by Contract in C++
// =============================================================================
// Topics:
//   1. Contract class with requires/ensures
//   2. PID controller with contracts
//   3. safe_array<T,N> with configurable bounds checking
//   4. Stack-based expression evaluator with invariant checking
//   5. Saturating arithmetic
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cmath>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <limits>
#include <type_traits>

// ======================== 1. CONTRACT CLASS =================================

// Macro-based contracts with file/line info
#define REQUIRE(cond) \
    Contract::requires((cond), #cond, __FILE__, __LINE__)

#define ENSURE(cond) \
    Contract::ensures((cond), #cond, __FILE__, __LINE__)

#define INVARIANT(cond) \
    Contract::invariant((cond), #cond, __FILE__, __LINE__)

class Contract {
public:
    static void requires(bool condition, const char* expr,
                         const char* file, int line) {
        if (!condition) {
            std::cerr << "PRECONDITION FAILED: " << expr
                      << " at " << file << ":" << line << "\n";
            std::abort();
        }
    }

    static void ensures(bool condition, const char* expr,
                        const char* file, int line) {
        if (!condition) {
            std::cerr << "POSTCONDITION FAILED: " << expr
                      << " at " << file << ":" << line << "\n";
            std::abort();
        }
    }

    static void invariant(bool condition, const char* expr,
                          const char* file, int line) {
        if (!condition) {
            std::cerr << "INVARIANT VIOLATED: " << expr
                      << " at " << file << ":" << line << "\n";
            std::abort();
        }
    }
};

// ======================== 2. PID CONTROLLER WITH CONTRACTS ==================

class PIDController {
public:
    struct Config {
        double kp;
        double ki;
        double kd;
        double output_min;
        double output_max;
        double integral_limit;  // anti-windup
        double dt;              // time step in seconds
    };

    explicit PIDController(const Config& cfg)
        : cfg_(cfg)
        , integral_(0.0)
        , prev_error_(0.0)
    {
        REQUIRE(cfg.kp >= 0.0);
        REQUIRE(cfg.ki >= 0.0);
        REQUIRE(cfg.kd >= 0.0);
        REQUIRE(cfg.output_min < cfg.output_max);
        REQUIRE(cfg.integral_limit > 0.0);
        REQUIRE(cfg.dt > 0.0 && cfg.dt <= 1.0);
        check_invariant();
    }

    double update(double setpoint, double measurement) {
        REQUIRE(std::isfinite(setpoint));
        REQUIRE(std::isfinite(measurement));
        check_invariant();

        double error = setpoint - measurement;

        // Proportional
        double p_term = cfg_.kp * error;

        // Integral with anti-windup
        integral_ += error * cfg_.dt;
        if (integral_ > cfg_.integral_limit) integral_ = cfg_.integral_limit;
        if (integral_ < -cfg_.integral_limit) integral_ = -cfg_.integral_limit;
        double i_term = cfg_.ki * integral_;

        // Derivative
        double derivative = (error - prev_error_) / cfg_.dt;
        double d_term = cfg_.kd * derivative;
        prev_error_ = error;

        // Sum and clamp output
        double output = p_term + i_term + d_term;
        if (output > cfg_.output_max) output = cfg_.output_max;
        if (output < cfg_.output_min) output = cfg_.output_min;

        ENSURE(output >= cfg_.output_min && output <= cfg_.output_max);
        ENSURE(std::abs(integral_) <= cfg_.integral_limit);
        ENSURE(std::isfinite(output));
        check_invariant();

        return output;
    }

    void reset() {
        integral_ = 0.0;
        prev_error_ = 0.0;
        check_invariant();
    }

private:
    void check_invariant() const {
        INVARIANT(std::abs(integral_) <= cfg_.integral_limit + 1e-9);
        INVARIANT(std::isfinite(integral_));
        INVARIANT(std::isfinite(prev_error_));
    }

    Config cfg_;
    double integral_;
    double prev_error_;
};

// ======================== 3. SAFE ARRAY =====================================

enum class BoundsPolicy { kChecked, kUnchecked };

template <typename T, std::size_t N,
          BoundsPolicy Policy =
#ifdef NDEBUG
              BoundsPolicy::kUnchecked
#else
              BoundsPolicy::kChecked
#endif
          >
class safe_array {
public:
    safe_array() = default;
    safe_array(std::initializer_list<T> init) {
        REQUIRE(init.size() <= N);
        std::size_t i = 0;
        for (auto& v : init) {
            data_[i++] = v;
        }
    }

    T& operator[](std::size_t idx) {
        if constexpr (Policy == BoundsPolicy::kChecked) {
            REQUIRE(idx < N);
        }
        return data_[idx];
    }

    const T& operator[](std::size_t idx) const {
        if constexpr (Policy == BoundsPolicy::kChecked) {
            REQUIRE(idx < N);
        }
        return data_[idx];
    }

    [[nodiscard]] constexpr std::size_t size() const { return N; }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

private:
    std::array<T, N> data_{};
};

// ======================== 4. STACK WITH INVARIANT CHECKING ===================

template <typename T, std::size_t Capacity>
class InvariantStack {
public:
    InvariantStack() : top_(0) {
        check_invariant();
    }

    void push(const T& value) {
        REQUIRE(top_ < Capacity);
        data_[top_++] = value;
        ENSURE(top_ > 0);
        ENSURE(top_ <= Capacity);
        check_invariant();
    }

    T pop() {
        REQUIRE(top_ > 0);
        T value = data_[--top_];
        ENSURE(top_ < Capacity);
        check_invariant();
        return value;
    }

    [[nodiscard]] const T& peek() const {
        REQUIRE(top_ > 0);
        return data_[top_ - 1];
    }

    [[nodiscard]] bool empty() const { return top_ == 0; }
    [[nodiscard]] std::size_t size() const { return top_; }

private:
    void check_invariant() const {
        INVARIANT(top_ <= Capacity);
        // After every operation, size must be consistent
    }

    std::array<T, Capacity> data_{};
    std::size_t top_;
};

// Simple expression evaluator using the stack
// Evaluates postfix (RPN) expressions: "3 4 + 2 *" = (3+4)*2 = 14
double eval_rpn(const std::string& expr) {
    REQUIRE(!expr.empty());

    InvariantStack<double, 64> stack;
    std::istringstream iss(expr);
    std::string token;

    while (iss >> token) {
        if (token == "+" || token == "-" || token == "*" || token == "/") {
            REQUIRE(stack.size() >= 2);
            double b = stack.pop();
            double a = stack.pop();
            double result = 0.0;
            if (token == "+") result = a + b;
            else if (token == "-") result = a - b;
            else if (token == "*") result = a * b;
            else if (token == "/") {
                REQUIRE(std::abs(b) > 1e-12);
                result = a / b;
            }
            ENSURE(std::isfinite(result));
            stack.push(result);
        } else {
            stack.push(std::stod(token));
        }
    }

    ENSURE(stack.size() == 1);
    return stack.pop();
}

// ======================== 5. SATURATING ARITHMETIC ==========================

namespace sat {

int32_t add(int32_t a, int32_t b) {
    int64_t result = static_cast<int64_t>(a) + static_cast<int64_t>(b);
    if (result > INT32_MAX) return INT32_MAX;
    if (result < INT32_MIN) return INT32_MIN;
    ENSURE(result >= INT32_MIN && result <= INT32_MAX);
    return static_cast<int32_t>(result);
}

int32_t sub(int32_t a, int32_t b) {
    int64_t result = static_cast<int64_t>(a) - static_cast<int64_t>(b);
    if (result > INT32_MAX) return INT32_MAX;
    if (result < INT32_MIN) return INT32_MIN;
    ENSURE(result >= INT32_MIN && result <= INT32_MAX);
    return static_cast<int32_t>(result);
}

int32_t mul(int32_t a, int32_t b) {
    int64_t result = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    if (result > INT32_MAX) return INT32_MAX;
    if (result < INT32_MIN) return INT32_MIN;
    ENSURE(result >= INT32_MIN && result <= INT32_MAX);
    return static_cast<int32_t>(result);
}

} // namespace sat

// ======================== TESTS =============================================

void test_pid() {
    PIDController::Config cfg{
        .kp = 1.0, .ki = 0.1, .kd = 0.01,
        .output_min = -100.0, .output_max = 100.0,
        .integral_limit = 50.0, .dt = 0.01
    };
    PIDController pid(cfg);

    // Setpoint = 10, measurement starts at 0, output should be positive
    double out = pid.update(10.0, 0.0);
    assert(out > 0.0 && out <= 100.0);

    // Drive many iterations — integral should not wind up past limit
    for (int i = 0; i < 10000; ++i) {
        out = pid.update(10.0, 0.0); // constant error
    }
    assert(out <= 100.0 && out >= -100.0);

    pid.reset();
    assert(pid.update(0.0, 0.0) == 0.0); // zero error -> zero output

    std::cout << "  PID Controller with Contracts: PASS\n";
}

void test_safe_array() {
    safe_array<int, 5> arr{10, 20, 30, 40, 50};
    assert(arr[0] == 10);
    assert(arr[4] == 50);
    assert(arr.size() == 5);

    // Modify
    arr[2] = 99;
    assert(arr[2] == 99);

    // Range-for works
    int sum = 0;
    for (auto v : arr) sum += v;
    assert(sum == 10 + 20 + 99 + 40 + 50);

    std::cout << "  safe_array<T,N>: PASS\n";
}

void test_invariant_stack() {
    InvariantStack<int, 10> s;
    assert(s.empty());

    s.push(1);
    s.push(2);
    s.push(3);
    assert(s.size() == 3);
    assert(s.peek() == 3);

    assert(s.pop() == 3);
    assert(s.pop() == 2);
    assert(s.pop() == 1);
    assert(s.empty());

    std::cout << "  InvariantStack: PASS\n";
}

void test_rpn() {
    assert(std::abs(eval_rpn("3 4 +") - 7.0) < 1e-9);
    assert(std::abs(eval_rpn("3 4 + 2 *") - 14.0) < 1e-9);
    assert(std::abs(eval_rpn("10 2 / 3 -") - 2.0) < 1e-9);
    assert(std::abs(eval_rpn("5 1 2 + 4 * + 3 -") - 14.0) < 1e-9);

    std::cout << "  RPN Evaluator: PASS\n";
}

void test_saturating() {
    // Normal cases
    assert(sat::add(10, 20) == 30);
    assert(sat::sub(10, 20) == -10);
    assert(sat::mul(100, 200) == 20000);

    // Overflow saturates to MAX
    assert(sat::add(INT32_MAX, 1) == INT32_MAX);
    assert(sat::add(INT32_MAX, INT32_MAX) == INT32_MAX);

    // Underflow saturates to MIN
    assert(sat::sub(INT32_MIN, 1) == INT32_MIN);
    assert(sat::add(INT32_MIN, INT32_MIN) == INT32_MIN);

    // Multiply overflow
    assert(sat::mul(INT32_MAX, 2) == INT32_MAX);
    assert(sat::mul(INT32_MIN, 2) == INT32_MIN);
    assert(sat::mul(100000, 100000) == INT32_MAX); // 10^10 > INT32_MAX

    // Zero cases
    assert(sat::add(0, 0) == 0);
    assert(sat::mul(0, INT32_MAX) == 0);

    std::cout << "  Saturating Arithmetic: PASS\n";
}

int main() {
    std::cout << "=== Defensive Programming — Design by Contract ===\n";
    test_pid();
    test_safe_array();
    test_invariant_stack();
    test_rpn();
    test_saturating();
    std::cout << "=== ALL TESTS PASSED ===\n";
    return 0;
}
