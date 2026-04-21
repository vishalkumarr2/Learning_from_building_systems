// =============================================================================
// Exercise 01: Exception Safety Guarantee Levels
// =============================================================================
// Demonstrates all 4 guarantee levels using a BankAccount transfer scenario.
// Each version shows a different level: No guarantee → Basic → Strong → No-throw.
//
// Compile: g++ -std=c++2a -Wall -Wextra -o ex01 ex01_guarantee_levels.cpp
// =============================================================================

#include <iostream>
#include <string>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <cassert>
#include <climits>
#include <climits>

// A flag to force failures at specific points
static bool g_force_failure = false;

// =============================================================================
// Utility: A transfer operation that may fail
// =============================================================================
void checked_credit(int& balance, int amount) {
    if (g_force_failure) {
        throw std::runtime_error("Network error: credit operation failed");
    }
    balance += amount;
}

// =============================================================================
// VERSION 1: NO GUARANTEE (The WRONG way — never do this)
// =============================================================================
// Problems:
//  - If credit throws, debit has already happened → money vanishes
//  - Raw pointer may leak if exception occurs between new and delete
//  - Invariant violated: total money in system changes
namespace no_guarantee {

struct Account {
    std::string name;
    int balance;
};

// WRONG: If checked_credit throws, 'from' is already debited.
// Money has vanished from the system. This is a data corruption bug.
void transfer(Account& from, Account& to, int amount) {
    std::cout << "  [no-guarantee] Debiting " << from.name << "..." << std::endl;
    from.balance -= amount;   // POINT OF NO RETURN — already modified!

    std::cout << "  [no-guarantee] Crediting " << to.name << "..." << std::endl;
    checked_credit(to.balance, amount);  // If this throws, 'from' is debited
                                          // but 'to' is never credited!
}

void demo() {
    std::cout << "\n=== VERSION 1: NO GUARANTEE ===" << std::endl;
    Account alice{"Alice", 1000};
    Account bob{"Bob", 500};

    std::cout << "Before: Alice=" << alice.balance << " Bob=" << bob.balance
              << " Total=" << (alice.balance + bob.balance) << std::endl;

    g_force_failure = true;
    try {
        transfer(alice, bob, 200);
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
    }

    std::cout << "After:  Alice=" << alice.balance << " Bob=" << bob.balance
              << " Total=" << (alice.balance + bob.balance) << std::endl;
    std::cout << "  BUG: $200 vanished! Total was 1500, now "
              << (alice.balance + bob.balance) << std::endl;
    g_force_failure = false;
}

} // namespace no_guarantee

// =============================================================================
// VERSION 2: BASIC GUARANTEE
// =============================================================================
// Guarantees:
//  - No resource leaks (RAII handles cleanup)
//  - Invariants preserved (total money is conserved)
//  - But state may have changed: the debit is rolled back on failure
//  - The rollback itself might not restore exact original state in general
namespace basic_guarantee {

struct Account {
    std::string name;
    int balance;

    void withdraw(int amount) {
        if (amount > balance) throw std::runtime_error("Insufficient funds");
        balance -= amount;
    }
    void deposit(int amount) {
        checked_credit(balance, amount);
    }
};

// BASIC: If credit fails, we roll back the debit.
// The rollback itself (deposit) could theoretically fail in a more complex
// scenario, but here it's simple enough to succeed. In the basic guarantee,
// we preserve invariants but the exact state might differ from the start.
void transfer(Account& from, Account& to, int amount) {
    std::cout << "  [basic] Debiting " << from.name << "..." << std::endl;
    from.withdraw(amount);

    try {
        std::cout << "  [basic] Crediting " << to.name << "..." << std::endl;
        to.deposit(amount);
    } catch (...) {
        std::cout << "  [basic] Credit failed, rolling back debit..." << std::endl;
        // Rollback: re-deposit to 'from'. This is a best-effort rollback.
        // In the basic guarantee, we ensure no invariant violation.
        from.balance += amount;  // Direct add — cannot fail for int
        throw;  // re-throw the original exception
    }
}

void demo() {
    std::cout << "\n=== VERSION 2: BASIC GUARANTEE ===" << std::endl;
    Account alice{"Alice", 1000};
    Account bob{"Bob", 500};

    std::cout << "Before: Alice=" << alice.balance << " Bob=" << bob.balance
              << " Total=" << (alice.balance + bob.balance) << std::endl;

    g_force_failure = true;
    try {
        transfer(alice, bob, 200);
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
    }

    std::cout << "After:  Alice=" << alice.balance << " Bob=" << bob.balance
              << " Total=" << (alice.balance + bob.balance) << std::endl;
    std::cout << "  OK: Money conserved. State rolled back (basic guarantee)."
              << std::endl;
    g_force_failure = false;
}

} // namespace basic_guarantee

// =============================================================================
// VERSION 3: STRONG GUARANTEE (commit-or-rollback)
// =============================================================================
// Guarantees:
//  - If exception thrown, state is EXACTLY as before the call
//  - Uses copy-and-swap: work on copies, then commit with noexcept swap
namespace strong_guarantee {

struct Account {
    std::string name;
    int balance;

    void withdraw(int amount) {
        if (amount > balance) throw std::runtime_error("Insufficient funds");
        balance -= amount;
    }
    void deposit(int amount) {
        checked_credit(balance, amount);
    }

    // noexcept swap — the commit mechanism
    friend void swap(Account& a, Account& b) noexcept {
        using std::swap;
        swap(a.name, b.name);
        swap(a.balance, b.balance);
    }
};

// STRONG: All work happens on copies. Only commit (swap) if everything succeeds.
// If any exception occurs, the originals are untouched.
void transfer(Account& from, Account& to, int amount) {
    // Phase 1: Do all throwing work on COPIES
    Account from_copy = from;
    Account to_copy = to;

    std::cout << "  [strong] Working on copies..." << std::endl;
    from_copy.withdraw(amount);
    to_copy.deposit(amount);  // If this throws, from & to are unchanged!

    // Phase 2: COMMIT — noexcept operations only
    std::cout << "  [strong] Committing (noexcept swap)..." << std::endl;
    swap(from, from_copy);   // noexcept
    swap(to, to_copy);       // noexcept
}

void demo() {
    std::cout << "\n=== VERSION 3: STRONG GUARANTEE ===" << std::endl;
    Account alice{"Alice", 1000};
    Account bob{"Bob", 500};

    std::cout << "Before: Alice=" << alice.balance << " Bob=" << bob.balance
              << " Total=" << (alice.balance + bob.balance) << std::endl;

    // First: a successful transfer
    transfer(alice, bob, 200);
    std::cout << "Success: Alice=" << alice.balance << " Bob=" << bob.balance
              << " Total=" << (alice.balance + bob.balance) << std::endl;

    // Now: a failed transfer — state must revert to post-first-transfer
    g_force_failure = true;
    try {
        transfer(alice, bob, 300);
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
    }

    std::cout << "After:  Alice=" << alice.balance << " Bob=" << bob.balance
              << " Total=" << (alice.balance + bob.balance) << std::endl;
    std::cout << "  OK: State exactly as before the failed transfer (strong guarantee)."
              << std::endl;

    // Verify: should still be Alice=800, Bob=700 from the first transfer
    assert(alice.balance == 800 && bob.balance == 700);
    g_force_failure = false;
}

} // namespace strong_guarantee

// =============================================================================
// VERSION 4: NO-THROW GUARANTEE
// =============================================================================
// Guarantees:
//  - The function NEVER throws
//  - Pre-validates all conditions before modifying state
//  - Marked noexcept — compiler can optimize
namespace nothrow_guarantee {

struct Account {
    std::string name;
    int balance;
};

enum class TransferResult {
    Success,
    InsufficientFunds,
    InvalidAmount,
    Overflow
};

// NO-THROW: Pre-validate everything, then do only noexcept operations.
// Returns an error code instead of throwing.
TransferResult transfer(Account& from, Account& to, int amount) noexcept {
    // Pre-validation — no state changes yet
    if (amount <= 0)          return TransferResult::InvalidAmount;
    if (from.balance < amount) return TransferResult::InsufficientFunds;
    if (to.balance > (INT_MAX - amount)) return TransferResult::Overflow;

    // All checks passed — safe to modify (only noexcept operations)
    from.balance -= amount;
    to.balance += amount;
    return TransferResult::Success;
}

const char* result_str(TransferResult r) noexcept {
    switch (r) {
        case TransferResult::Success:          return "Success";
        case TransferResult::InsufficientFunds: return "Insufficient funds";
        case TransferResult::InvalidAmount:    return "Invalid amount";
        case TransferResult::Overflow:         return "Overflow";
    }
    return "Unknown";
}

void demo() {
    std::cout << "\n=== VERSION 4: NO-THROW GUARANTEE ===" << std::endl;
    Account alice{"Alice", 1000};
    Account bob{"Bob", 500};

    std::cout << "Before: Alice=" << alice.balance << " Bob=" << bob.balance << std::endl;

    // Successful transfer
    auto r1 = transfer(alice, bob, 200);
    std::cout << "Transfer 200: " << result_str(r1) << std::endl;
    std::cout << "  Alice=" << alice.balance << " Bob=" << bob.balance << std::endl;

    // Failed transfer — insufficient funds
    auto r2 = transfer(alice, bob, 5000);
    std::cout << "Transfer 5000: " << result_str(r2) << std::endl;
    std::cout << "  Alice=" << alice.balance << " Bob=" << bob.balance << std::endl;

    // Negative amount
    auto r3 = transfer(alice, bob, -100);
    std::cout << "Transfer -100: " << result_str(r3) << std::endl;

    std::cout << "  All operations noexcept — no exceptions possible." << std::endl;

    // Compile-time proof that transfer is noexcept
    static_assert(noexcept(transfer(alice, bob, 100)),
                  "transfer must be noexcept");
}

} // namespace nothrow_guarantee

// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Exception Safety Guarantee Levels" << std::endl;
    std::cout << "============================================" << std::endl;

    no_guarantee::demo();
    basic_guarantee::demo();
    strong_guarantee::demo();
    nothrow_guarantee::demo();

    std::cout << "\n============================================" << std::endl;
    std::cout << " Summary:" << std::endl;
    std::cout << "  No guarantee:  Money vanished (BUG!)" << std::endl;
    std::cout << "  Basic:         Money conserved, rollback attempted" << std::endl;
    std::cout << "  Strong:        Exact state restored on failure" << std::endl;
    std::cout << "  No-throw:      Cannot fail — error codes instead" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
