// =============================================================================
// Exercise 03: Scope Guards for Exception-Safe Cleanup
// =============================================================================
// Implements scope guards that automatically perform cleanup on scope exit.
// Shows ScopeGuard, ScopeFailure (rollback), and ScopeSuccess patterns using
// std::uncaught_exceptions() from C++17.
//
// Compile: g++ -std=c++2a -Wall -Wextra -o ex03 ex03_scope_guard.cpp
// =============================================================================

#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include <stdexcept>
#include <exception>
#include <utility>

// =============================================================================
// VERSION 1: Basic ScopeGuard using std::function
// =============================================================================
// Pros: Simple, any callable
// Cons: std::function may allocate (for large lambdas)
namespace v1 {

class ScopeGuard {
    std::function<void()> cleanup_;
    bool active_;

public:
    explicit ScopeGuard(std::function<void()> fn)
        : cleanup_(std::move(fn))
        , active_(true)
    {}

    // Non-copyable
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    // Move-constructible (for returning from factory functions)
    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup_(std::move(other.cleanup_))
        , active_(other.active_)
    {
        other.active_ = false;
    }

    ~ScopeGuard() {
        if (active_) {
            try { cleanup_(); }
            catch (...) {
                // NEVER let cleanup throw — we might be in stack unwinding
                std::cerr << "  [ScopeGuard] cleanup threw! Swallowed." << std::endl;
            }
        }
    }

    // Call dismiss() if the guarded operation succeeded and cleanup is no longer needed
    void dismiss() noexcept { active_ = false; }
};

void demo() {
    std::cout << "\n=== V1: Basic ScopeGuard (std::function) ===" << std::endl;

    // Example: file-like resource with manual open/close
    std::cout << "  Opening resource..." << std::endl;
    int resource_handle = 42;  // pretend this is a file descriptor

    ScopeGuard guard([&resource_handle]() {
        std::cout << "  Closing resource handle " << resource_handle << std::endl;
        resource_handle = -1;
    });

    std::cout << "  Using resource..." << std::endl;
    // Even if we throw here, the guard's destructor closes the resource

    std::cout << "  Done. Guard will fire on scope exit." << std::endl;
    // guard fires here — resource is closed
}

} // namespace v1

// =============================================================================
// VERSION 2: Optimized ScopeGuard with template lambda (no allocation)
// =============================================================================
// The lambda is stored directly in the object — no std::function overhead.
namespace v2 {

template<typename Func>
class ScopeGuard {
    Func cleanup_;
    bool active_;

public:
    explicit ScopeGuard(Func fn) noexcept
        : cleanup_(std::move(fn))
        , active_(true)
    {}

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup_(std::move(other.cleanup_))
        , active_(other.active_)
    {
        other.active_ = false;
    }

    ~ScopeGuard() {
        if (active_) {
            try { cleanup_(); }
            catch (...) {}
        }
    }

    void dismiss() noexcept { active_ = false; }
};

// Factory function with CTAD (class template argument deduction) fallback
template<typename Func>
ScopeGuard<Func> make_scope_guard(Func&& fn) {
    return ScopeGuard<Func>(std::forward<Func>(fn));
}

void demo() {
    std::cout << "\n=== V2: Template ScopeGuard (no allocation) ===" << std::endl;

    auto guard = make_scope_guard([]() {
        std::cout << "  Template guard fired! (zero overhead)" << std::endl;
    });

    std::cout << "  Working..." << std::endl;
    // guard fires at end of scope
}

} // namespace v2

// =============================================================================
// VERSION 3: ScopeSuccess / ScopeFailure using std::uncaught_exceptions()
// =============================================================================
// C++17's std::uncaught_exceptions() (note: plural!) returns the NUMBER of
// uncaught exceptions. By comparing the count at construction vs destruction,
// we can determine if we're in normal scope exit vs exception unwinding.
namespace v3 {

template<typename Func>
class ScopeFailure {
    Func rollback_;
    int exceptions_on_entry_;

public:
    explicit ScopeFailure(Func fn) noexcept
        : rollback_(std::move(fn))
        , exceptions_on_entry_(std::uncaught_exceptions())
    {}

    ScopeFailure(const ScopeFailure&) = delete;
    ScopeFailure& operator=(const ScopeFailure&) = delete;

    ~ScopeFailure() {
        // Fire ONLY if a NEW exception is being unwound
        if (std::uncaught_exceptions() > exceptions_on_entry_) {
            try { rollback_(); }
            catch (...) {}  // never throw from destructors
        }
    }
};

template<typename Func>
class ScopeSuccess {
    Func commit_;
    int exceptions_on_entry_;

public:
    explicit ScopeSuccess(Func fn) noexcept
        : commit_(std::move(fn))
        , exceptions_on_entry_(std::uncaught_exceptions())
    {}

    ScopeSuccess(const ScopeSuccess&) = delete;
    ScopeSuccess& operator=(const ScopeSuccess&) = delete;

    ~ScopeSuccess() {
        // Fire ONLY if NO new exception is being unwound (normal exit)
        if (std::uncaught_exceptions() <= exceptions_on_entry_) {
            commit_();  // OK to throw here — we're not unwinding
        }
    }
};

// Helpers
template<typename Func>
ScopeFailure<Func> on_failure(Func&& fn) {
    return ScopeFailure<Func>(std::forward<Func>(fn));
}

template<typename Func>
ScopeSuccess<Func> on_success(Func&& fn) {
    return ScopeSuccess<Func>(std::forward<Func>(fn));
}

void demo() {
    std::cout << "\n=== V3: ScopeSuccess / ScopeFailure ===" << std::endl;

    // --- Case 1: Successful operation ---
    std::cout << "\n  Case 1: Successful operation" << std::endl;
    {
        auto fail_guard = on_failure([]() {
            std::cout << "  [ScopeFailure] Rolling back! (should NOT fire)" << std::endl;
        });
        auto success_guard = on_success([]() {
            std::cout << "  [ScopeSuccess] Committing! (should fire)" << std::endl;
        });

        std::cout << "  Doing work (no throw)..." << std::endl;
    }

    // --- Case 2: Failed operation ---
    std::cout << "\n  Case 2: Failed operation" << std::endl;
    try {
        auto fail_guard = on_failure([]() {
            std::cout << "  [ScopeFailure] Rolling back! (should fire)" << std::endl;
        });
        auto success_guard = on_success([]() {
            std::cout << "  [ScopeSuccess] Committing! (should NOT fire)" << std::endl;
        });

        std::cout << "  Doing work... about to throw!" << std::endl;
        throw std::runtime_error("oops");
    } catch (const std::exception& e) {
        std::cout << "  Caught: " << e.what() << std::endl;
    }
}

} // namespace v3

// =============================================================================
// REAL-WORLD EXAMPLE: Multi-step transaction with auto-rollback
// =============================================================================
// This replaces the traditional pattern of nested try/catch blocks with
// clean scope guards.
namespace transaction_example {

// Simulated database operations
struct Database {
    std::vector<std::string> log;
    bool should_fail_on_step3 = false;

    void insert_order(int id) {
        log.push_back("INSERT order " + std::to_string(id));
        std::cout << "    DB: INSERT order " << id << std::endl;
    }
    void delete_order(int id) {
        log.push_back("DELETE order " + std::to_string(id));
        std::cout << "    DB: DELETE order " << id << " (rollback)" << std::endl;
    }
    void insert_payment(int order_id, int amount) {
        log.push_back("INSERT payment for order " + std::to_string(order_id));
        std::cout << "    DB: INSERT payment $" << amount
                  << " for order " << order_id << std::endl;
    }
    void delete_payment(int order_id) {
        log.push_back("DELETE payment for order " + std::to_string(order_id));
        std::cout << "    DB: DELETE payment for order " << order_id
                  << " (rollback)" << std::endl;
    }
    void update_inventory(int item_id, int delta) {
        if (should_fail_on_step3) {
            throw std::runtime_error("Inventory service unavailable!");
        }
        log.push_back("UPDATE inventory item " + std::to_string(item_id));
        std::cout << "    DB: UPDATE inventory item " << item_id
                  << " by " << delta << std::endl;
    }
};

// ---- THE OLD WAY: nested try/catch (ugly, error-prone) ----
void place_order_old_way(Database& db, int order_id, int amount, int item_id) {
    db.insert_order(order_id);
    try {
        db.insert_payment(order_id, amount);
        try {
            db.update_inventory(item_id, -1);
            // If we add more steps, nesting grows deeper and deeper...
        } catch (...) {
            db.delete_payment(order_id);  // rollback step 2
            throw;
        }
    } catch (...) {
        db.delete_order(order_id);  // rollback step 1
        throw;
    }
}

// ---- THE NEW WAY: scope guards (clean, composable) ----
void place_order_with_guards(Database& db, int order_id, int amount, int item_id) {
    // Step 1: Create order
    db.insert_order(order_id);
    auto guard1 = v3::on_failure([&]() { db.delete_order(order_id); });

    // Step 2: Record payment
    db.insert_payment(order_id, amount);
    auto guard2 = v3::on_failure([&]() { db.delete_payment(order_id); });

    // Step 3: Update inventory (may fail!)
    db.update_inventory(item_id, -1);

    // If we reach here, all steps succeeded.
    // ScopeFailure guards won't fire because no exception is in flight.
    // Adding more steps just means adding more lines — no nesting!
}

void demo() {
    std::cout << "\n=== REAL-WORLD: Multi-Step Transaction ===" << std::endl;

    // --- Successful transaction ---
    {
        std::cout << "\n  Successful transaction:" << std::endl;
        Database db;
        place_order_with_guards(db, 1001, 50, 42);
        std::cout << "  Transaction committed! (" << db.log.size()
                  << " operations)" << std::endl;
    }

    // --- Failed transaction (step 3 fails) ---
    {
        std::cout << "\n  Failed transaction (step 3 fails):" << std::endl;
        Database db;
        db.should_fail_on_step3 = true;
        try {
            place_order_with_guards(db, 1002, 75, 99);
        } catch (const std::exception& e) {
            std::cout << "  Transaction failed: " << e.what() << std::endl;
        }
        std::cout << "  Operations performed: " << db.log.size() << std::endl;
        for (auto& entry : db.log)
            std::cout << "    " << entry << std::endl;
        std::cout << "  Note: rollbacks happened automatically via ScopeFailure!"
                  << std::endl;
    }
}

} // namespace transaction_example

// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Scope Guards for Exception-Safe Cleanup" << std::endl;
    std::cout << "============================================" << std::endl;

    v1::demo();
    v2::demo();
    v3::demo();
    transaction_example::demo();

    std::cout << "\n============================================" << std::endl;
    std::cout << " Key Insight:" << std::endl;
    std::cout << "  ScopeGuard    = always cleanup on scope exit" << std::endl;
    std::cout << "  ScopeFailure  = rollback only on exception" << std::endl;
    std::cout << "  ScopeSuccess  = commit only on normal exit" << std::endl;
    std::cout << "  Replaces nested try/catch with composable guards!" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
