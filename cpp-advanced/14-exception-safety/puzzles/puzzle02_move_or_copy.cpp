// =============================================================================
// Puzzle 02: Move or Copy? — Predicting std::vector's Behavior
// =============================================================================
// When std::vector grows, does it MOVE or COPY existing elements?
// The answer depends entirely on whether your move constructor is noexcept.
//
// This puzzle presents several Widget variants and challenges you to predict
// whether vector will move or copy each one.
//
// Compile: g++ -std=c++2a -Wall -Wextra -o puzzle02 puzzle02_move_or_copy.cpp
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

// =============================================================================
// Helper: tracks how each Widget variant is relocated
// =============================================================================
struct Stats {
    int copies = 0;
    int moves = 0;
    void reset() { copies = 0; moves = 0; }
    void report(const char* name) const {
        std::cout << "    " << name << ": " << copies << " copies, "
                  << moves << " moves → vector used "
                  << (copies > 0 ? "COPY" : "MOVE") << std::endl;
    }
};

// =============================================================================
// Widget A: Has noexcept move constructor
// =============================================================================
// Prediction: vector will MOVE
namespace widget_a {
    static Stats stats;

    struct Widget {
        int data;
        explicit Widget(int d) : data(d) {}

        Widget(const Widget& o) : data(o.data) { ++stats.copies; }

        Widget(Widget&& o) noexcept : data(o.data) {  // ← noexcept!
            o.data = -1;
            ++stats.moves;
        }

        Widget& operator=(Widget o) noexcept {
            data = o.data;
            return *this;
        }
    };
}

// =============================================================================
// Widget B: Move constructor is NOT noexcept
// =============================================================================
// Prediction: vector will COPY (to maintain strong guarantee during realloc)
namespace widget_b {
    static Stats stats;

    struct Widget {
        int data;
        explicit Widget(int d) : data(d) {}

        Widget(const Widget& o) : data(o.data) { ++stats.copies; }

        Widget(Widget&& o) : data(o.data) {  // ← NOT noexcept!
            o.data = -1;
            ++stats.moves;
        }

        Widget& operator=(Widget o) noexcept {
            data = o.data;
            return *this;
        }
    };
}

// =============================================================================
// Widget C: No move constructor at all (deleted)
// =============================================================================
// Prediction: vector will COPY (move is deleted, falls back to copy)
namespace widget_c {
    static Stats stats;

    struct Widget {
        int data;
        explicit Widget(int d) : data(d) {}

        Widget(const Widget& o) : data(o.data) { ++stats.copies; }

        Widget(Widget&&) = delete;  // ← explicitly deleted!

        Widget& operator=(const Widget& o) {
            data = o.data;
            return *this;
        }
    };
}

// =============================================================================
// Widget D: Move constructor is implicitly noexcept (compiler-generated)
// =============================================================================
// Prediction: vector will MOVE (compiler generates noexcept move for POD-like types)
namespace widget_d {
    static Stats stats;

    struct Widget {
        int data;
        double extra;
        explicit Widget(int d) : data(d), extra(0.0) {}

        // No user-declared copy/move — compiler generates them
        // Compiler-generated move ctor for POD members is noexcept
        // But we need to track copies/moves, so we add a layer:
    };

    // We can't directly track compiler-generated ops, so let's check the trait
    void verify() {
        std::cout << "    Widget D is_nothrow_move_constructible: "
                  << std::is_nothrow_move_constructible_v<Widget>
                  << " (vector will MOVE)" << std::endl;
    }
}

// =============================================================================
// Widget E: Has a std::string member — is the move still noexcept?
// =============================================================================
// Prediction: YES, std::string has noexcept move. Vector will MOVE.
namespace widget_e {
    static Stats stats;

    struct Widget {
        int data;
        std::string name;  // std::string move is noexcept

        explicit Widget(int d, std::string n = "")
            : data(d), name(std::move(n)) {}

        Widget(const Widget& o) : data(o.data), name(o.name) { ++stats.copies; }

        // noexcept because all members (int, std::string) have noexcept moves
        Widget(Widget&& o) noexcept
            : data(o.data), name(std::move(o.name))
        {
            o.data = -1;
            ++stats.moves;
        }

        Widget& operator=(Widget o) noexcept {
            data = o.data;
            name = std::move(o.name);
            return *this;
        }
    };
}

// =============================================================================
// Widget F: Has a member whose move MIGHT throw
// =============================================================================
// Prediction: if the member's move isn't noexcept, and our move isn't noexcept,
// vector will COPY.
namespace widget_f {
    static Stats stats;

    // A type with a potentially-throwing move
    struct Risky {
        int* ptr;
        Risky() : ptr(new int(0)) {}
        Risky(const Risky& o) : ptr(new int(*o.ptr)) {}
        Risky(Risky&& o) : ptr(o.ptr) {  // NOT noexcept
            o.ptr = nullptr;
        }
        ~Risky() { delete ptr; }
        Risky& operator=(Risky o) { std::swap(ptr, o.ptr); return *this; }
    };

    struct Widget {
        int data;
        Risky risky;

        explicit Widget(int d) : data(d) {}

        Widget(const Widget& o) : data(o.data), risky(o.risky) { ++stats.copies; }

        // This move is NOT noexcept because Risky's move isn't noexcept
        Widget(Widget&& o) : data(o.data), risky(std::move(o.risky)) {
            o.data = -1;
            ++stats.moves;
        }

        Widget& operator=(Widget o) {
            data = o.data;
            risky = std::move(o.risky);
            return *this;
        }
    };
}

// =============================================================================
// Test harness
// =============================================================================
template<typename Widget, typename StatsT>
void run_test(const char* name, StatsT& stats) {
    stats.reset();
    std::vector<Widget> vec;
    // Reserve 1 to force a reallocation on the 2nd push_back
    vec.reserve(1);
    vec.push_back(Widget(1));   // fits in reserved space

    stats.reset();  // only count operations during reallocation
    vec.push_back(Widget(2));   // triggers reallocation → existing element
                                 // must be relocated (moved or copied)
    stats.report(name);
}

// =============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << " Puzzle: Move or Copy?" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "\n  When vector grows, does it MOVE or COPY existing elements?"
              << std::endl;
    std::cout << "  Try to predict before looking at the output!\n" << std::endl;

    // Compile-time traits
    std::cout << "  Compile-time noexcept check:" << std::endl;
    std::cout << "    A (noexcept move):      nothrow_move="
              << std::is_nothrow_move_constructible_v<widget_a::Widget> << std::endl;
    std::cout << "    B (no noexcept move):   nothrow_move="
              << std::is_nothrow_move_constructible_v<widget_b::Widget> << std::endl;
    std::cout << "    C (deleted move):       move_constructible="
              << std::is_move_constructible_v<widget_c::Widget> << std::endl;
    std::cout << "    E (string member):      nothrow_move="
              << std::is_nothrow_move_constructible_v<widget_e::Widget> << std::endl;
    std::cout << "    F (risky member):       nothrow_move="
              << std::is_nothrow_move_constructible_v<widget_f::Widget> << std::endl;

    std::cout << "\n  --- Results ---\n" << std::endl;

    run_test<widget_a::Widget>("A (noexcept move)", widget_a::stats);
    run_test<widget_b::Widget>("B (may-throw move)", widget_b::stats);
    // widget_c can't be tested with push_back(Widget(N)) because move is deleted
    // and the temporary can't be copied directly. Use emplace instead.
    {
        widget_c::stats.reset();
        std::vector<widget_c::Widget> vec;
        vec.reserve(1);
        vec.emplace_back(1);
        widget_c::stats.reset();
        vec.emplace_back(2);  // triggers realloc
        widget_c::stats.report("C (deleted move)");
    }
    widget_d::verify();
    run_test<widget_e::Widget>("E (string member, noexcept)", widget_e::stats);
    run_test<widget_f::Widget>("F (risky member, no noexcept)", widget_f::stats);

    std::cout << "\n  --- Answer Key ---\n" << std::endl;
    std::cout << "  A: MOVE — noexcept move, vector trusts it" << std::endl;
    std::cout << "  B: COPY — move might throw, vector plays it safe" << std::endl;
    std::cout << "  C: COPY — no move at all, only copy available" << std::endl;
    std::cout << "  D: MOVE — compiler-generated noexcept move for POD members"
              << std::endl;
    std::cout << "  E: MOVE — std::string has noexcept move, so Widget does too"
              << std::endl;
    std::cout << "  F: COPY — Risky's move isn't noexcept → Widget isn't → copy!"
              << std::endl;

    std::cout << "\n============================================" << std::endl;
    std::cout << " Performance Trap:" << std::endl;
    std::cout << "  Forgetting noexcept on a move constructor" << std::endl;
    std::cout << "  silently turns O(1) moves into O(N) copies" << std::endl;
    std::cout << "  during every vector reallocation!" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  For a vector of 1M elements with 100-byte" << std::endl;
    std::cout << "  objects, each reallocation goes from swapping" << std::endl;
    std::cout << "  pointers (nanoseconds) to copying 100MB." << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
