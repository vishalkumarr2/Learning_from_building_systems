// =============================================================================
// Puzzle 01: Throwing in Destructors
// =============================================================================
// What happens when a destructor throws during stack unwinding?
// Answer: std::terminate() is called, killing your process.
//
// Since C++11, destructors are implicitly noexcept. Throwing from a noexcept
// function calls std::terminate(). This puzzle demonstrates the problem and
// shows the safe alternative.
//
// Compile: g++ -std=c++2a -Wall -Wextra -o puzzle01 puzzle01_destructor_throw.cpp
//
// WARNING: One test case intentionally triggers std::terminate. That path is
// disabled by default — uncomment the marked section to see it happen.
// =============================================================================

#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
#include <functional>

// =============================================================================
// WRONG: Destructor that throws
// =============================================================================
// Since C++11, destructors are implicitly noexcept.
// Throwing from a noexcept function → std::terminate().
namespace bad {

class Resource {
    std::string name_;
    bool should_fail_;

public:
    Resource(std::string name, bool should_fail = false)
        : name_(std::move(name)), should_fail_(should_fail)
    {
        std::cout << "  [bad] " << name_ << " constructed" << std::endl;
    }

    // This destructor is implicitly noexcept (C++11+).
    // If cleanup() throws, std::terminate() is called.
    ~Resource() {
        std::cout << "  [bad] " << name_ << " destructor called" << std::endl;
        if (should_fail_) {
            std::cout << "  [bad] " << name_
                      << " about to throw from destructor!" << std::endl;

            // DANGER: This will call std::terminate() because:
            // 1. Destructors are implicitly noexcept since C++11
            // 2. Throwing from a noexcept function → terminate()
            // 3. Even WITHOUT being noexcept, if we're in stack unwinding
            //    from another exception, having two exceptions in flight
            //    also calls terminate().
            //
            // Uncomment the next line to see std::terminate in action:
            // throw std::runtime_error("cleanup failed in " + name_);
            //
            std::cout << "  [bad] (throw commented out to avoid termination)"
                      << std::endl;
        }
    }
};

void demo() {
    std::cout << "\n=== WRONG: Throwing Destructor ===" << std::endl;
    std::cout << "  (throw is commented out to keep the program alive)" << std::endl;

    try {
        Resource r1("r1", false);
        Resource r2("r2", true);   // this one would throw in destructor
        // When r2's destructor runs (stack unwinding or normal), it would
        // call std::terminate() if it threw.
    } catch (...) {
        std::cout << "  (never reached if terminate is called)" << std::endl;
    }
}

} // namespace bad

// =============================================================================
// ALSO WRONG: noexcept(false) destructor — allows throwing, but dangerous
// =============================================================================
namespace explicit_throwing {

class Resource {
    std::string name_;
    bool should_fail_;

public:
    Resource(std::string name, bool should_fail = false)
        : name_(std::move(name)), should_fail_(should_fail)
    {
        std::cout << "  [explicit] " << name_ << " constructed" << std::endl;
    }

    // You CAN mark a destructor noexcept(false) to allow throwing.
    // But this is almost always a bad idea:
    // - If called during stack unwinding, two exceptions in flight → terminate()
    // - If the object is in a container, the container can't clean up safely
    // - Makes reasoning about exception safety nearly impossible
    ~Resource() noexcept(false) {
        std::cout << "  [explicit] " << name_ << " destructor" << std::endl;
        if (should_fail_) {
            throw std::runtime_error("cleanup failed: " + name_);
        }
    }
};

void demo() {
    std::cout << "\n=== noexcept(false) Destructor ===" << std::endl;

    // Case 1: No other exception in flight — the throw "works" (but is still bad)
    std::cout << "\n  Case 1: No other exception — destructor throw propagates:"
              << std::endl;
    try {
        Resource r("res", true);  // destructor will throw
    } catch (const std::exception& e) {
        std::cout << "  Caught from destructor: " << e.what() << std::endl;
        std::cout << "  (This 'works' but is bad practice)" << std::endl;
    }

    // Case 2: Another exception in flight — would call terminate!
    std::cout << "\n  Case 2: Exception during unwinding (commented out):" << std::endl;
    std::cout << "  If we threw inside a catch block's unwinding, terminate() would"
              << std::endl;
    std::cout << "  be called. This is why noexcept(false) destructors are dangerous."
              << std::endl;

    // Uncomment to see terminate():
    // try {
    //     Resource r("inner", true);
    //     throw std::runtime_error("outer exception");
    //     // r's destructor throws during unwinding → terminate!
    // } catch (...) {}
}

} // namespace explicit_throwing

// =============================================================================
// CORRECT: Safe destructor that logs errors without throwing
// =============================================================================
namespace safe {

// Error handler type — can be customized per-instance
using ErrorHandler = std::function<void(const std::string&)>;

void default_error_handler(const std::string& msg) {
    std::cerr << "  [safe] ERROR (swallowed): " << msg << std::endl;
}

class Resource {
    std::string name_;
    int* data_;
    ErrorHandler on_error_;

    // Private cleanup that may fail
    void cleanup_impl() {
        // Simulate a cleanup that might fail
        if (data_ && *data_ < 0) {
            throw std::runtime_error("cleanup_impl failed for " + name_);
        }
        delete data_;
        data_ = nullptr;
    }

public:
    Resource(std::string name, int value,
             ErrorHandler handler = default_error_handler)
        : name_(std::move(name))
        , data_(new int(value))
        , on_error_(std::move(handler))
    {
        std::cout << "  [safe] " << name_ << " constructed (value="
                  << value << ")" << std::endl;
    }

    // CORRECT: Destructor that NEVER throws.
    // It catches any exception from cleanup and handles it gracefully.
    ~Resource() {
        std::cout << "  [safe] " << name_ << " destructor" << std::endl;
        try {
            cleanup_impl();
            std::cout << "  [safe] " << name_ << " cleaned up OK" << std::endl;
        } catch (const std::exception& e) {
            // Log the error but DON'T rethrow
            try {
                on_error_(e.what());
            } catch (...) {
                // Even the error handler failed — nothing we can do
            }
            // Still need to release the resource!
            delete data_;
            data_ = nullptr;
        } catch (...) {
            delete data_;
            data_ = nullptr;
        }
    }

    // Explicit close() method for callers who want to handle errors
    void close() {
        cleanup_impl();  // throws if cleanup fails — caller can handle it
    }
};

void demo() {
    std::cout << "\n=== CORRECT: Safe Destructor ===" << std::endl;

    // Case 1: Normal cleanup (value >= 0)
    std::cout << "\n  Case 1: Clean resource" << std::endl;
    { Resource r("clean", 42); }

    // Case 2: Cleanup would throw (value < 0), but destructor swallows it
    std::cout << "\n  Case 2: Problematic resource (cleanup error swallowed)"
              << std::endl;
    { Resource r("problematic", -1); }

    // Case 3: Caller can use close() to detect errors
    std::cout << "\n  Case 3: Explicit close() for error detection" << std::endl;
    {
        Resource r("explicit", -1);
        try {
            r.close();
        } catch (const std::exception& e) {
            std::cout << "  Caller detected: " << e.what() << std::endl;
            std::cout << "  (Destructor will still run safely)" << std::endl;
        }
    }

    std::cout << "\n  Pattern: Use close() for callers who care about errors,"
              << std::endl;
    std::cout << "  destructor swallows exceptions for RAII safety." << std::endl;
}

} // namespace safe

// =============================================================================
// CHALLENGE: Understanding std::uncaught_exceptions() in destructors
// =============================================================================
namespace challenge {

class DiagnosticGuard {
    std::string context_;
    int exceptions_on_entry_;

public:
    DiagnosticGuard(std::string ctx)
        : context_(std::move(ctx))
        , exceptions_on_entry_(std::uncaught_exceptions())
    {
        std::cout << "  [diag] Enter '" << context_
                  << "' (uncaught=" << exceptions_on_entry_ << ")" << std::endl;
    }

    ~DiagnosticGuard() {
        int now = std::uncaught_exceptions();
        if (now > exceptions_on_entry_) {
            std::cout << "  [diag] Leave '" << context_
                      << "' — UNWINDING (uncaught: " << exceptions_on_entry_
                      << " → " << now << ")" << std::endl;
        } else {
            std::cout << "  [diag] Leave '" << context_
                      << "' — normal exit" << std::endl;
        }
    }
};

void demo() {
    std::cout << "\n=== CHALLENGE: std::uncaught_exceptions() ===" << std::endl;

    // Normal exit
    { DiagnosticGuard g("block A"); }

    // Exception exit
    try {
        DiagnosticGuard g("block B");
        throw std::runtime_error("test");
    } catch (...) {}

    // Nested: exception through multiple scopes
    try {
        DiagnosticGuard outer("outer");
        {
            DiagnosticGuard inner("inner");
            throw std::runtime_error("nested test");
            // inner's destructor sees uncaught_exceptions() > 0
        }
        // outer's destructor also sees uncaught_exceptions() > 0
    } catch (...) {}
}

} // namespace challenge

// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Puzzle: Throwing in Destructors" << std::endl;
    std::cout << "============================================" << std::endl;

    bad::demo();
    explicit_throwing::demo();
    safe::demo();
    challenge::demo();

    std::cout << "\n============================================" << std::endl;
    std::cout << " Rules:" << std::endl;
    std::cout << "  1. NEVER throw from destructors" << std::endl;
    std::cout << "  2. Catch-and-swallow is the destructor pattern" << std::endl;
    std::cout << "  3. Provide explicit close() for error-aware callers" << std::endl;
    std::cout << "  4. noexcept(false) destructors are almost always wrong" << std::endl;
    std::cout << "  5. std::uncaught_exceptions() detects unwinding context" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
