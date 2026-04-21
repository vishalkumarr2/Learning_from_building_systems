// Exercise 1: Coding Standards Enforcement
//
// This exercise contains 8 functions that violate common C++ coding standards
// (CppCoreGuidelines, MISRA, CERT). Each has a "bad" version and a "fixed"
// version. The self-test verifies the fixed versions produce correct results.
//
// TOOL PRACTICE:
//   1. Compile with strict warnings — how many warnings do you see?
//      g++ -std=c++2a -Wall -Wextra -Wpedantic -Wconversion -Wshadow
//          -Wold-style-cast -Wdouble-promotion -Wsign-conversion ex01_coding_standards.cpp
//
//   2. Run clang-tidy on this file:
//      clang-tidy -checks='cppcoreguidelines-*,modernize-*,bugprone-*,cert-*'
//          ex01_coding_standards.cpp -- -std=c++2a
//
//   3. Run cppcheck:
//      cppcheck --enable=all --std=c++20 ex01_coding_standards.cpp
//
// After running the tools, examine the warnings and understand why each
// pattern is flagged.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// ========================================================================
// Violation 1: Owning raw pointer (CppCoreGuidelines R.1, R.3)
// ========================================================================

namespace violation1 {

// BAD: raw pointer ownership — if process() throws, we leak
struct SensorData {
    double values[12]{};
    bool valid = false;
};

SensorData* bad_create_sensor_data() {
    auto* data = new SensorData();
    data->valid = true;
    for (int i = 0; i < 12; ++i)
        data->values[i] = static_cast<double>(i) * 0.1;
    return data;  // caller must remember to delete
}

// FIXED: unique_ptr for ownership transfer
std::unique_ptr<SensorData> good_create_sensor_data() {
    auto data = std::make_unique<SensorData>();
    data->valid = true;
    for (int i = 0; i < 12; ++i)
        data->values[i] = static_cast<double>(i) * 0.1;
    return data;  // ownership transferred, no leak possible
}

void test() {
    // Bad version — must manually delete
    SensorData* raw = bad_create_sensor_data();
    assert(raw->valid);
    delete raw;

    // Good version — automatic cleanup
    auto smart = good_create_sensor_data();
    assert(smart->valid);
    assert(smart->values[5] > 0.49 && smart->values[5] < 0.51);
}

}  // namespace violation1

// ========================================================================
// Violation 2: C-style cast (MISRA 7.0.5, CppCoreGuidelines ES.49)
// ========================================================================

namespace violation2 {

// BAD: C-style casts bypass the type system
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
void bad_process(void* raw_data, int size) {
    // C-style cast can do reinterpret_cast silently
    double* arr = (double*)raw_data;
    for (int i = 0; i < size; ++i)
        arr[i] *= 2.0;
}
#pragma GCC diagnostic pop

// FIXED: explicit static_cast or reinterpret_cast
void good_process(void* raw_data, int size) {
    // reinterpret_cast makes the dangerous conversion explicit
    auto* arr = reinterpret_cast<double*>(raw_data);
    for (int i = 0; i < size; ++i)
        arr[i] *= 2.0;
}

void test() {
    double data[] = {1.0, 2.0, 3.0};
    good_process(data, 3);
    assert(data[0] > 1.99 && data[0] < 2.01);
    assert(data[1] > 3.99 && data[1] < 4.01);
    assert(data[2] > 5.99 && data[2] < 6.01);
}

}  // namespace violation2

// ========================================================================
// Violation 3: Implicit narrowing conversion (MISRA 4.6.1, -Wconversion)
// ========================================================================

namespace violation3 {

// BAD: silent narrowing — data loss
struct Packet {
    uint8_t header;
    uint8_t payload_len;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
Packet bad_make_packet(int total_size) {
    Packet p;
    p.header = 0xAA;
    p.payload_len = total_size - 1;  // narrowing: int → uint8_t
    return p;
}
#pragma GCC diagnostic pop

// FIXED: explicit cast with range check
Packet good_make_packet(int total_size) {
    assert(total_size > 0 && total_size <= 256);
    Packet p;
    p.header = 0xAA;
    p.payload_len = static_cast<uint8_t>(total_size - 1);
    return p;
}

void test() {
    Packet p = good_make_packet(10);
    assert(p.header == 0xAA);
    assert(p.payload_len == 9);
}

}  // namespace violation3

// ========================================================================
// Violation 4: Variable shadowing (-Wshadow)
// ========================================================================

namespace violation4 {

// BAD: inner variable shadows outer — easy to use wrong one
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
int bad_count_valid(const std::vector<int>& data, int threshold) {
    int count = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        int count = 0;  // SHADOW: hides outer count!
        if (data[i] > threshold)
            count++;     // increments inner count → lost each iteration
    }
    return count;        // always returns 0
}
#pragma GCC diagnostic pop

// FIXED: unique variable names
int good_count_valid(const std::vector<int>& data, int threshold) {
    int total = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] > threshold)
            total++;
    }
    return total;
}

void test() {
    std::vector<int> data = {1, 5, 3, 7, 2, 8, 4};
    assert(bad_count_valid(data, 4) == 0);   // bug!
    assert(good_count_valid(data, 4) == 3);  // correct
}

}  // namespace violation4

// ========================================================================
// Violation 5: Unnamed lock_guard (CppCoreGuidelines CP.44)
// ========================================================================

namespace violation5 {

struct SharedCounter {
    std::mutex mtx;
    int value = 0;
};

// BAD: anonymous temporary — unlocks immediately!
void bad_increment(SharedCounter& sc) {
    std::lock_guard<std::mutex>{sc.mtx};  // temporary — destroyed immediately
    sc.value++;                           // ← NOT PROTECTED!
}

// FIXED: named lock_guard — lives until scope end
void good_increment(SharedCounter& sc) {
    std::lock_guard<std::mutex> lock{sc.mtx};
    sc.value++;  // protected until 'lock' goes out of scope
}

void test() {
    SharedCounter sc;
    good_increment(sc);
    good_increment(sc);
    assert(sc.value == 2);
}

}  // namespace violation5

// ========================================================================
// Violation 6: Magic numbers (readability-magic-numbers)
// ========================================================================

namespace violation6 {

// BAD: magic numbers everywhere — what do these mean?
double bad_compute_velocity(double encoder_ticks, double dt) {
    return (encoder_ticks / 4096.0) * 3.14159265358979 * 0.1 / dt;
}

// FIXED: named constants with meaning
constexpr double kPi = 3.14159265358979;
constexpr double kTicksPerRevolution = 4096.0;
constexpr double kWheelDiameter_m = 0.1;

double good_compute_velocity(double encoder_ticks, double dt) {
    const double revolutions = encoder_ticks / kTicksPerRevolution;
    const double distance_m = revolutions * kPi * kWheelDiameter_m;
    return distance_m / dt;
}

void test() {
    double bad_v = bad_compute_velocity(4096.0, 1.0);
    double good_v = good_compute_velocity(4096.0, 1.0);
    // Both should compute the same value
    assert(std::abs(bad_v - good_v) < 1e-10);
    // One revolution of a 0.1m wheel at dt=1s
    double expected = kPi * kWheelDiameter_m;
    assert(std::abs(good_v - expected) < 1e-10);
}

}  // namespace violation6

// ========================================================================
// Violation 7: Missing #define guard / using namespace in header scope
// (This is demonstrated as a code pattern, not a separate header file)
// ========================================================================

namespace violation7 {

// BAD: using namespace std in header scope pollutes all includers
// (demonstrated in function scope for this exercise)
int bad_find_max(const std::vector<int>& v) {
    using namespace std;  // pulls ALL of std into scope
    return *max_element(v.begin(), v.end());
}

// FIXED: qualified names or targeted using-declarations
int good_find_max(const std::vector<int>& v) {
    return *std::max_element(v.begin(), v.end());
}

void test() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};
    assert(bad_find_max(v) == 9);
    assert(good_find_max(v) == 9);
}

}  // namespace violation7

// ========================================================================
// Violation 8: Uninitialized member (MISRA 15.0.2, CERT EXP53-CPP)
// ========================================================================

namespace violation8 {

// BAD: members not initialized — reading them is undefined behavior
struct PidController {
    double kp, ki, kd;
    double integral;
    double prev_error;

    double compute(double error, double dt) {
        double derivative = (error - prev_error) / dt;  // UB if prev_error uninit
        integral += error * dt;                          // UB if integral uninit
        prev_error = error;
        return kp * error + ki * integral + kd * derivative;
    }
};

// FIXED: in-class member initializers or constructor init list
struct GoodPidController {
    double kp = 0.0;
    double ki = 0.0;
    double kd = 0.0;
    double integral = 0.0;
    double prev_error = 0.0;

    GoodPidController(double p, double i, double d)
        : kp(p), ki(i), kd(d) {}

    double compute(double error, double dt) {
        double derivative = (error - prev_error) / dt;
        integral += error * dt;
        prev_error = error;
        return kp * error + ki * integral + kd * derivative;
    }
};

void test() {
    GoodPidController pid(1.0, 0.1, 0.01);
    double output = pid.compute(1.0, 0.01);
    // kp*1.0 + ki*0.0 + kd*(1.0/0.01) = 1.0 + 0.0 + 1.0 = 2.0
    // Wait: kd*(error - 0)/dt = 0.01 * (1.0 / 0.01) = 1.0
    // ki * integral = 0.1 * (1.0 * 0.01) = 0.001
    // Total = 1.0 + 0.001 + 1.0 = 2.001
    assert(output > 1.9 && output < 2.1);
}

}  // namespace violation8

// ========================================================================
// Main — run all tests
// ========================================================================

int main() {
    violation1::test();
    violation2::test();
    violation3::test();
    violation4::test();
    violation5::test();
    violation6::test();
    violation7::test();
    violation8::test();

    std::printf("ex01_coding_standards: ALL 8 TESTS PASSED\n");
    return 0;
}
