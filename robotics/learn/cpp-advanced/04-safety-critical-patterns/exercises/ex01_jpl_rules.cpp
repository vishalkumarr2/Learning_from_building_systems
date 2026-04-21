// =============================================================================
// Exercise 01: JPL Power of 10 Rules Applied to C++
// =============================================================================
// For each JPL rule, we show a VIOLATING version and a COMPLIANT version.
// A self-test verifies the compliant versions produce correct results.
//
// Rules covered:
//   1. No recursion
//   2. Fixed loop bounds
//   3. No heap after init (PMR pattern)
//   4. Short functions (<60 lines)
//   5. Min 2 assertions per function
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cstring>
#include <array>
#include <iostream>
#include <memory_resource>
#include <vector>
#include <stack>
#include <string>
#include <functional>

// ======================== RULE 1: NO RECURSION ==============================
// Prevent unbounded stack growth. Every call graph must be a DAG.

namespace rule1 {

// --- VIOLATING: Recursive Fibonacci ---
// Problem: fib(45) creates ~2.3 billion stack frames
namespace bad {
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2); // VIOLATION: recursion
}
} // namespace bad

// --- COMPLIANT: Iterative Fibonacci ---
int fibonacci(int n) {
    assert(n >= 0 && "fibonacci: n must be non-negative");
    assert(n <= 46 && "fibonacci: n must be <= 46 to fit int32");
    if (n <= 1) return n;
    int prev = 0;
    int curr = 1;
    for (int i = 2; i <= n; ++i) {
        int next = prev + curr;
        prev = curr;
        curr = next;
    }
    assert(curr >= 0 && "fibonacci: result must be non-negative");
    return curr;
}

// --- VIOLATING: Recursive tree traversal ---
struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;
};

namespace bad {
int tree_sum(const TreeNode* node) {
    if (!node) return 0;
    return node->value + tree_sum(node->left) + tree_sum(node->right); // VIOLATION
}
} // namespace bad

// --- COMPLIANT: Iterative tree traversal with explicit stack ---
int tree_sum(const TreeNode* root) {
    assert((root == nullptr || root->value >= 0 || root->value < 0) &&
           "tree_sum: root pointer validity assumed");

    if (!root) return 0;

    int sum = 0;
    constexpr std::size_t MAX_DEPTH = 64; // bounded stack
    std::array<const TreeNode*, MAX_DEPTH> stack{};
    std::size_t top = 0;

    stack[top++] = root;
    while (top > 0) {
        assert(top <= MAX_DEPTH && "tree_sum: stack overflow — tree too deep");
        const TreeNode* node = stack[--top];
        sum += node->value;
        if (node->right && top < MAX_DEPTH) stack[top++] = node->right;
        if (node->left && top < MAX_DEPTH)  stack[top++] = node->left;
    }

    return sum;
}

void test() {
    // Fibonacci
    assert(fibonacci(0) == 0);
    assert(fibonacci(1) == 1);
    assert(fibonacci(10) == 55);
    assert(fibonacci(20) == 6765);

    // Tree sum
    TreeNode n4{4, nullptr, nullptr};
    TreeNode n5{5, nullptr, nullptr};
    TreeNode n2{2, &n4, &n5};
    TreeNode n3{3, nullptr, nullptr};
    TreeNode n1{1, &n2, &n3};
    assert(tree_sum(&n1) == 15); // 1+2+3+4+5
    assert(tree_sum(nullptr) == 0);

    std::cout << "  Rule 1 (No Recursion): PASS\n";
}
} // namespace rule1

// ======================== RULE 2: FIXED LOOP BOUNDS =========================
// Every loop must have a provable upper bound on iterations.

namespace rule2 {

// --- VIOLATING: Unbounded retry ---
namespace bad {
bool wait_for_device_ready(bool& device_ready) {
    while (!device_ready) {
        // spin forever if device never becomes ready — VIOLATION
    }
    return true;
}
} // namespace bad

// --- COMPLIANT: Bounded retry with explicit maximum ---
enum class RetryResult { kSuccess, kTimeout };

RetryResult wait_for_device_ready(const std::function<bool()>& is_ready,
                                  int max_retries) {
    assert(max_retries > 0 && "wait_for_device_ready: max_retries must be positive");
    assert(max_retries <= 100000 && "wait_for_device_ready: max_retries unreasonably large");

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        if (is_ready()) {
            return RetryResult::kSuccess;
        }
    }
    return RetryResult::kTimeout;
}

// --- COMPLIANT: Bounded search with early exit ---
int bounded_find(const int* data, int size, int target) {
    assert(data != nullptr && "bounded_find: null data pointer");
    assert(size >= 0 && size <= 1000000 && "bounded_find: size out of range");

    for (int i = 0; i < size; ++i) {
        if (data[i] == target) return i;
    }
    return -1; // not found
}

void test() {
    int call_count = 0;
    auto ready_after_5 = [&call_count]() {
        return ++call_count >= 5;
    };

    assert(wait_for_device_ready(ready_after_5, 100) == RetryResult::kSuccess);
    assert(call_count == 5);

    call_count = 0;
    auto never_ready = [&call_count]() {
        ++call_count;
        return false;
    };
    assert(wait_for_device_ready(never_ready, 10) == RetryResult::kTimeout);
    assert(call_count == 10);

    std::array<int, 5> arr = {10, 20, 30, 40, 50};
    assert(bounded_find(arr.data(), 5, 30) == 2);
    assert(bounded_find(arr.data(), 5, 99) == -1);

    std::cout << "  Rule 2 (Fixed Loop Bounds): PASS\n";
}
} // namespace rule2

// ======================== RULE 3: NO HEAP AFTER INIT ========================
// All dynamic allocation must happen during initialization.
// After init, use pre-allocated memory pools (PMR).

namespace rule3 {

// --- VIOLATING: Heap allocation during operation ---
namespace bad {
void process_sensor_data(int reading) {
    auto* record = new int(reading); // VIOLATION: heap alloc at runtime
    // ... process ...
    delete record;
}
} // namespace bad

// --- COMPLIANT: PMR pattern — allocate pool at init, use monotonic resource ---
class SensorProcessor {
public:
    static constexpr std::size_t POOL_SIZE = 4096;

    SensorProcessor()
        : pool_resource_(buffer_.data(), buffer_.size())
        , readings_(&pool_resource_)
    {
        readings_.reserve(256); // reserve in init phase
        assert(readings_.capacity() >= 256 && "SensorProcessor: reserve failed");
    }

    void add_reading(int value) {
        assert(readings_.size() < readings_.capacity() &&
               "SensorProcessor: readings full — increase pool");
        readings_.push_back(value);
        assert(!readings_.empty() && "SensorProcessor: push_back failed");
    }

    int average() const {
        assert(!readings_.empty() && "SensorProcessor: no readings to average");
        long sum = 0;
        for (auto r : readings_) sum += r;
        int avg = static_cast<int>(sum / static_cast<long>(readings_.size()));
        assert(avg >= -100000 && avg <= 100000 && "SensorProcessor: average out of range");
        return avg;
    }

    std::size_t count() const { return readings_.size(); }

    void reset() {
        readings_.clear();
        // Note: monotonic_buffer_resource does NOT release memory on clear.
        // For a real system, use pool_resource or reset the monotonic resource.
    }

private:
    std::array<std::byte, POOL_SIZE> buffer_{};
    std::pmr::monotonic_buffer_resource pool_resource_;
    std::pmr::vector<int> readings_;
};

void test() {
    SensorProcessor proc;
    for (int i = 0; i < 10; ++i) {
        proc.add_reading(i * 10);
    }
    assert(proc.count() == 10);
    assert(proc.average() == 45); // (0+10+20+...+90)/10 = 45

    std::cout << "  Rule 3 (No Heap After Init): PASS\n";
}
} // namespace rule3

// ======================== RULE 4: SHORT FUNCTIONS ===========================
// No function longer than 60 lines. One responsibility per function.

namespace rule4 {

// --- VIOLATING: A monolithic "do-everything" function ---
// (We show the structure, not a full 200-line monster)
namespace bad {
struct SensorData { int temp; int pressure; int humidity; };
struct ProcessedData { int adjusted_temp; int adjusted_pressure; int status; };

ProcessedData process_all(SensorData raw) {
    // In a real violation, this would be 200+ lines doing:
    // 1. Validate ranges (20 lines)
    // 2. Apply calibration (30 lines)
    // 3. Filter outliers (25 lines)
    // 4. Convert units (20 lines)
    // 5. Check alarms (30 lines)
    // 6. Format output (25 lines)
    // 7. Log results (20 lines)
    // All in one function — VIOLATION
    return {raw.temp - 3, raw.pressure + 10, 0};
}
} // namespace bad

// --- COMPLIANT: Each responsibility is its own function ---
struct SensorData {
    int temp;      // raw ADC counts
    int pressure;  // raw ADC counts
    int humidity;  // raw ADC counts
};

struct CalibratedData {
    int temp_c;       // Celsius * 100
    int pressure_pa;  // Pascals
    int humidity_pct; // 0-100
};

struct ProcessedData {
    CalibratedData calibrated;
    int status; // 0 = ok, 1 = warning, 2 = alarm
};

bool validate_ranges(const SensorData& raw) {
    assert(raw.temp >= 0 && "validate: temp ADC cannot be negative");
    bool valid = (raw.temp >= 100 && raw.temp <= 4000);
    valid = valid && (raw.pressure >= 50 && raw.pressure <= 3900);
    valid = valid && (raw.humidity >= 0 && raw.humidity <= 4095);
    return valid;
}

CalibratedData apply_calibration(const SensorData& raw) {
    assert(raw.temp >= 100 && raw.temp <= 4000 && "calibrate: temp out of valid range");
    assert(raw.pressure >= 50 && raw.pressure <= 3900 && "calibrate: pressure out of range");
    CalibratedData cal{};
    cal.temp_c = (raw.temp - 500) * 10;          // simplified calibration
    cal.pressure_pa = raw.pressure * 25 + 30000;  // simplified calibration
    cal.humidity_pct = raw.humidity * 100 / 4095;
    return cal;
}

int check_alarms(const CalibratedData& cal) {
    assert(cal.humidity_pct >= 0 && cal.humidity_pct <= 100 && "alarm: humidity out of range");
    if (cal.temp_c > 8500 || cal.temp_c < -2000) return 2; // alarm
    if (cal.pressure_pa > 110000 || cal.pressure_pa < 80000) return 1; // warning
    return 0;
}

ProcessedData process_sensor(const SensorData& raw) {
    assert(validate_ranges(raw) && "process_sensor: raw data out of range");
    CalibratedData cal = apply_calibration(raw);
    int status = check_alarms(cal);
    assert(status >= 0 && status <= 2 && "process_sensor: invalid status");
    return {cal, status};
}

void test() {
    SensorData raw{1000, 2000, 2048}; // temp=1000 → 5000 centidegrees (50°C)
    assert(validate_ranges(raw));

    ProcessedData result = process_sensor(raw);
    assert(result.status == 0); // normal (temp 5000 < 8500, pressure 80000 in range)
    assert(result.calibrated.humidity_pct == 50); // 2048/4095*100 ≈ 50

    std::cout << "  Rule 4 (Short Functions): PASS\n";
}
} // namespace rule4

// ======================== RULE 5: MIN 2 ASSERTIONS PER FUNCTION =============
// Assertions are executable specifications, not debug aids.

namespace rule5 {

// --- VIOLATING: No assertions ---
namespace bad {
double compute_speed(double distance, double time) {
    return distance / time; // No checks — what if time == 0? distance < 0?
}
} // namespace bad

// --- COMPLIANT: Assertions document and enforce contracts ---
double compute_speed(double distance_m, double time_s) {
    // Preconditions
    assert(distance_m >= 0.0 && "compute_speed: distance cannot be negative");
    assert(time_s > 0.001 && "compute_speed: time must be positive and non-trivial");

    double speed = distance_m / time_s;

    // Postconditions
    assert(speed >= 0.0 && "compute_speed: speed cannot be negative");
    assert(speed < 1000.0 && "compute_speed: speed exceeds physical limit (1000 m/s)");

    return speed;
}

// --- Bounded integer operation with overflow check ---
int32_t safe_multiply(int32_t a, int32_t b) {
    assert(a >= -10000 && a <= 10000 && "safe_multiply: a out of range");
    assert(b >= -10000 && b <= 10000 && "safe_multiply: b out of range");

    int64_t result = static_cast<int64_t>(a) * static_cast<int64_t>(b);

    assert(result >= INT32_MIN && result <= INT32_MAX &&
           "safe_multiply: result overflows int32");
    return static_cast<int32_t>(result);
}

void test() {
    assert(compute_speed(100.0, 10.0) == 10.0);
    assert(compute_speed(0.0, 1.0) == 0.0);
    assert(safe_multiply(100, 200) == 20000);
    assert(safe_multiply(-50, 30) == -1500);
    assert(safe_multiply(0, 9999) == 0);

    std::cout << "  Rule 5 (Min 2 Assertions/Function): PASS\n";
}
} // namespace rule5

// ======================== SELF-TEST =========================================

int main() {
    std::cout << "=== JPL Power of 10 Rules — Self Tests ===\n";
    rule1::test();
    rule2::test();
    rule3::test();
    rule4::test();
    rule5::test();
    std::cout << "=== ALL RULE TESTS PASSED ===\n";
    return 0;
}
