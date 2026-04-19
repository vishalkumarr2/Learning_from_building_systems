// puzzle01_exception_catch_order.cpp
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 puzzle01_exception_catch_order.cpp -o puzzle01
//
// PUZZLE: What does this program print? Why?
//         Then: what happens if you swap the catch block order?

#include <iostream>
#include <stdexcept>
#include <string>

struct NetworkError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct TimeoutError : NetworkError {
    int timeout_ms;
    TimeoutError(const std::string& msg, int ms)
        : NetworkError(msg), timeout_ms(ms) {}
};

struct DnsError : NetworkError {
    std::string hostname;
    DnsError(const std::string& msg, const std::string& host)
        : NetworkError(msg), hostname(host) {}
};

// ----- Scenario A: base-first ordering -----
void scenario_a(int code) {
    try {
        switch (code) {
            case 1: throw TimeoutError("timed out", 5000);
            case 2: throw DnsError("lookup failed", "api.example.com");
            case 3: throw NetworkError("connection reset");
            default: throw std::runtime_error("unknown");
        }
    }
    // BUG: base class catch comes first — swallows everything derived
    catch (const std::runtime_error& e) {
        std::cout << "  runtime_error: " << e.what() << "\n";
    }
    catch (const NetworkError& e) {
        std::cout << "  NetworkError: " << e.what() << "\n";
    }
    catch (const TimeoutError& e) {
        std::cout << "  TimeoutError: " << e.what() << " (ms=" << e.timeout_ms << ")\n";
    }
    catch (const DnsError& e) {
        std::cout << "  DnsError: " << e.what() << " (host=" << e.hostname << ")\n";
    }
}

// ----- Scenario B: derived-first ordering (correct) -----
void scenario_b(int code) {
    try {
        switch (code) {
            case 1: throw TimeoutError("timed out", 5000);
            case 2: throw DnsError("lookup failed", "api.example.com");
            case 3: throw NetworkError("connection reset");
            default: throw std::runtime_error("unknown");
        }
    }
    // CORRECT: most-derived first
    catch (const TimeoutError& e) {
        std::cout << "  TimeoutError: " << e.what() << " (ms=" << e.timeout_ms << ")\n";
    }
    catch (const DnsError& e) {
        std::cout << "  DnsError: " << e.what() << " (host=" << e.hostname << ")\n";
    }
    catch (const NetworkError& e) {
        std::cout << "  NetworkError: " << e.what() << "\n";
    }
    catch (const std::runtime_error& e) {
        std::cout << "  runtime_error: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "=== Scenario A: base-class catch first (WRONG) ===\n";
    for (int i = 1; i <= 4; ++i) {
        std::cout << "throw #" << i << ":\n";
        scenario_a(i);
    }

    std::cout << "\n=== Scenario B: derived-class catch first (CORRECT) ===\n";
    for (int i = 1; i <= 4; ++i) {
        std::cout << "throw #" << i << ":\n";
        scenario_b(i);
    }

    // Bonus: catch by value vs by reference
    std::cout << "\n=== Bonus: slicing when catching by value ===\n";
    try {
        throw TimeoutError("sliced!", 9999);
    }
    catch (std::runtime_error e) {  // by VALUE — object is sliced!
        std::cout << "  Caught by value: " << e.what() << "\n";
        std::cout << "  typeid: " << typeid(e).name() << "\n";
        // e is now a plain runtime_error — timeout_ms is gone
    }

    try {
        throw TimeoutError("not sliced!", 9999);
    }
    catch (const std::runtime_error& e) {  // by REFERENCE — polymorphism preserved
        std::cout << "  Caught by ref:   " << e.what() << "\n";
        std::cout << "  typeid: " << typeid(e).name() << "\n";
    }
}

/*
 * ═══════════════════════════════════════════════════════
 * ANSWER
 * ═══════════════════════════════════════════════════════
 *
 * === Scenario A output ===
 * throw #1: runtime_error: timed out          <-- TimeoutError caught as runtime_error!
 * throw #2: runtime_error: lookup failed       <-- DnsError caught as runtime_error!
 * throw #3: runtime_error: connection reset    <-- NetworkError caught as runtime_error!
 * throw #4: runtime_error: unknown             <-- plain runtime_error
 *
 * ALL four go to the first catch block because every thrown type
 * IS-A std::runtime_error. C++ tries catch blocks top-to-bottom
 * and picks the FIRST match, not the BEST match.
 *
 * The compiler may warn: "exception of type 'TimeoutError' will be
 * caught by earlier handler for 'std::runtime_error'" (-Wexceptions).
 *
 * === Scenario B output ===
 * throw #1: TimeoutError: timed out (ms=5000)
 * throw #2: DnsError: lookup failed (host=api.example.com)
 * throw #3: NetworkError: connection reset
 * throw #4: runtime_error: unknown
 *
 * Each exception is caught by its most-specific handler.
 *
 * === Bonus: slicing ===
 * Catching by value copies the exception into a std::runtime_error
 * object, slicing off the derived parts. The typeid will show
 * std::runtime_error, not TimeoutError.
 *
 * Catching by const reference preserves the dynamic type.
 *
 * RULE: Always catch exceptions by const reference, most-derived first.
 */
