// Exercise 2: Static Analysis Traps
//
// These functions contain bugs that static analysis tools catch.
// Each "bad_" function has a subtle defect. The "good_" version fixes it.
// The self-test verifies the fixed versions.
//
// TOOL PRACTICE:
//   cppcheck --enable=all --std=c++20 ex02_static_analysis_traps.cpp
//   clang-tidy -checks='clang-analyzer-*,bugprone-*' ex02_static_analysis_traps.cpp -- -std=c++2a
//
// Can the tools find all 8 bugs? Which ones do they miss?

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// ========================================================================
// Trap 1: Null dereference on error path
// ========================================================================

namespace trap1 {

struct Config {
    int timeout_ms = 100;
    std::string name = "default";
};

// BAD: dereferences result without checking
Config* bad_load_config(bool available) {
    Config* cfg = nullptr;
    if (available) {
        cfg = new Config();
        cfg->timeout_ms = 500;
    }
    // BUG: if !available, cfg is null but we still use it below
    // (In real code, the return is used without check)
    return cfg;
}

// FIXED: use optional or ensure validity
std::unique_ptr<Config> good_load_config(bool available) {
    if (!available) return nullptr;
    auto cfg = std::make_unique<Config>();
    cfg->timeout_ms = 500;
    return cfg;
}

void test() {
    auto cfg = good_load_config(true);
    assert(cfg != nullptr);
    assert(cfg->timeout_ms == 500);

    auto none = good_load_config(false);
    assert(none == nullptr);
}

}  // namespace trap1

// ========================================================================
// Trap 2: Buffer over-read (off-by-one)
// ========================================================================

namespace trap2 {

// BAD: reads one element past the end
double bad_average(const double* data, int count) {
    double sum = 0.0;
    for (int i = 0; i <= count; ++i)  // BUG: <= instead of <
        sum += data[i];
    return sum / static_cast<double>(count);
}

// FIXED: correct loop bound
double good_average(const double* data, int count) {
    if (count <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < count; ++i)
        sum += data[i];
    return sum / static_cast<double>(count);
}

void test() {
    double data[] = {10.0, 20.0, 30.0};
    double avg = good_average(data, 3);
    assert(avg > 19.99 && avg < 20.01);
    assert(good_average(nullptr, 0) == 0.0);
}

}  // namespace trap2

// ========================================================================
// Trap 3: Use-after-move
// ========================================================================

namespace trap3 {

// BAD: uses string after moving it
std::string bad_transfer(std::string& name) {
    std::string transferred = std::move(name);
    // BUG: name is in a valid-but-unspecified state after move
    // Accessing name.size() is technically OK but the value is unreliable
    std::string result = transferred + "_processed";
    result += "_from_" + name;  // name is empty or unspecified!
    return result;
}

// FIXED: don't use the source after move
std::string good_transfer(std::string& name) {
    std::string original = name;  // copy first if we need it later
    std::string transferred = std::move(name);
    std::string result = transferred + "_processed";
    result += "_from_" + original;
    return result;
}

void test() {
    std::string name = "sensor";
    std::string result = good_transfer(name);
    assert(result == "sensor_processed_from_sensor");
}

}  // namespace trap3

// ========================================================================
// Trap 4: Integer overflow in size computation
// ========================================================================

namespace trap4 {

// BAD: integer overflow when multiplying large sizes
void* bad_allocate_matrix(int rows, int cols, int elem_size) {
    // BUG: rows * cols * elem_size can overflow int before malloc
    int total = rows * cols * elem_size;  // overflow if rows=50000, cols=50000, size=8
    return std::malloc(static_cast<size_t>(total));
}

// FIXED: compute in size_t to prevent overflow
void* good_allocate_matrix(size_t rows, size_t cols, size_t elem_size) {
    // Check for overflow before multiplication
    if (cols != 0 && rows > SIZE_MAX / cols) return nullptr;
    size_t count = rows * cols;
    if (elem_size != 0 && count > SIZE_MAX / elem_size) return nullptr;
    size_t total = count * elem_size;
    return std::malloc(total);
}

void test() {
    void* p = good_allocate_matrix(100, 100, sizeof(double));
    assert(p != nullptr);
    std::free(p);

    // Overflow case — should return nullptr, not allocate garbage
    void* overflow = good_allocate_matrix(SIZE_MAX, 2, 1);
    assert(overflow == nullptr);
}

}  // namespace trap4

// ========================================================================
// Trap 5: Dead code / unreachable path
// ========================================================================

namespace trap5 {

enum class SensorState { kOk, kWarning, kError, kFault };

// BAD: unreachable code after exhaustive switch
const char* bad_state_name(SensorState s) {
    switch (s) {
        case SensorState::kOk:      return "OK";
        case SensorState::kWarning: return "WARNING";
        case SensorState::kError:   return "ERROR";
        case SensorState::kFault:   return "FAULT";
    }
    return "UNKNOWN";  // dead code — all cases covered
    // But: if enum is extended, compiler warns about missing case
    // The dead return masks the warning!
}

// FIXED: no default, no dead return — compiler warns on missing enum values
const char* good_state_name(SensorState s) {
    switch (s) {
        case SensorState::kOk:      return "OK";
        case SensorState::kWarning: return "WARNING";
        case SensorState::kError:   return "ERROR";
        case SensorState::kFault:   return "FAULT";
    }
    // If you MUST have a return for -Wreturn-type, use assert(false)
    assert(false && "unhandled SensorState");
    return "UNKNOWN";
}

void test() {
    assert(std::strcmp(good_state_name(SensorState::kOk), "OK") == 0);
    assert(std::strcmp(good_state_name(SensorState::kFault), "FAULT") == 0);
}

}  // namespace trap5

// ========================================================================
// Trap 6: Double-free via aliasing
// ========================================================================

namespace trap6 {

struct Buffer {
    int* data = nullptr;
    size_t size = 0;

    Buffer() = default;

    explicit Buffer(size_t n) : data(new int[n]), size(n) {
        std::memset(data, 0, n * sizeof(int));
    }

    // BAD: default copy does shallow copy → double-free
    // (Intentionally NOT defining copy ctor to show the bug pattern)

    ~Buffer() { delete[] data; }

    // Rule of Five — FIXED version
    Buffer(const Buffer& other) : data(nullptr), size(other.size) {
        if (size > 0) {
            data = new int[size];
            std::memcpy(data, other.data, size * sizeof(int));
        }
    }

    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = (size > 0) ? new int[size] : nullptr;
            if (data)
                std::memcpy(data, other.data, size * sizeof(int));
        }
        return *this;
    }

    Buffer(Buffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

void test() {
    Buffer a(10);
    a.data[0] = 42;

    Buffer b = a;  // copy — should deep-copy
    assert(b.data[0] == 42);
    assert(b.data != a.data);  // different pointers

    Buffer c = std::move(a);  // move
    assert(c.data[0] == 42);
    assert(a.data == nullptr);  // moved-from is null
}

}  // namespace trap6

// ========================================================================
// Trap 7: Signed/unsigned comparison
// ========================================================================

namespace trap7 {

// BAD: signed/unsigned comparison — if threshold is negative, always true
bool bad_is_above_threshold(const std::vector<int>& data, int threshold) {
    for (size_t i = 0; i < data.size(); ++i) {
        // BUG: comparing int (signed) with size_t (unsigned)
        // Not a bug here specifically, but the PATTERN is dangerous
        if (data[i] > threshold)
            return true;
    }
    return false;
}

// FIXED: use consistent types, range-based for
bool good_is_above_threshold(const std::vector<int>& data, int threshold) {
    return std::any_of(data.begin(), data.end(),
                       [threshold](int v) { return v > threshold; });
}

void test() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    assert(good_is_above_threshold(data, 3) == true);
    assert(good_is_above_threshold(data, 5) == false);
    assert(good_is_above_threshold(data, -1) == true);
}

}  // namespace trap7

// ========================================================================
// Trap 8: Resource leak on exception path
// ========================================================================

namespace trap8 {

struct Connection {
    bool connected = false;
    void connect() { connected = true; }
    void disconnect() { connected = false; }
};

// BAD: if process() throws, disconnect() is never called
bool bad_process_data() {
    Connection* conn = new Connection();
    conn->connect();
    // ... process ...
    bool result = conn->connected;
    conn->disconnect();
    delete conn;
    return result;
}

// FIXED: RAII wrapper
class ScopedConnection {
    Connection conn_;
public:
    ScopedConnection() { conn_.connect(); }
    ~ScopedConnection() { conn_.disconnect(); }
    bool is_connected() const { return conn_.connected; }
    // Delete copy
    ScopedConnection(const ScopedConnection&) = delete;
    ScopedConnection& operator=(const ScopedConnection&) = delete;
};

bool good_process_data() {
    ScopedConnection sc;
    // ... process ...
    return sc.is_connected();
    // disconnect() called automatically by destructor
}

void test() {
    assert(good_process_data() == true);
}

}  // namespace trap8

// ========================================================================
// Main
// ========================================================================

int main() {
    trap1::test();
    trap2::test();
    trap3::test();
    trap4::test();
    trap5::test();
    trap6::test();
    trap7::test();
    trap8::test();

    std::printf("ex02_static_analysis_traps: ALL 8 TESTS PASSED\n");
    return 0;
}
