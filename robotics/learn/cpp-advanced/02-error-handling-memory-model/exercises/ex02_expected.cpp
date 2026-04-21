// ex02_expected.cpp — Hand-rolled Expected<T,E> with monadic chaining
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread ex02_expected.cpp -o ex02
//
// Implements a minimal Expected<T,E> (like C++23 std::expected) using a tagged union.
// Demonstrates monadic composition: and_then (flatmap) and map (fmap).

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

// ── Expected<T,E> ────────────────────────────────────────────────────

template <typename T, typename E>
class Expected {
    union Storage {
        T val;
        E err;
        Storage() {}   // trivial — lifetime managed manually
        ~Storage() {}
    };

    Storage storage_;
    bool has_val_;

    void destroy() {
        if (has_val_) {
            storage_.val.~T();
        } else {
            storage_.err.~E();
        }
    }

public:
    // Value constructor
    Expected(T v) : has_val_(true) {
        new (&storage_.val) T(std::move(v));
    }

    // Error tag type for disambiguation
    struct UnexpectedTag {};
    Expected(UnexpectedTag, E e) : has_val_(false) {
        new (&storage_.err) E(std::move(e));
    }

    // Copy
    Expected(const Expected& o) : has_val_(o.has_val_) {
        if (has_val_) {
            new (&storage_.val) T(o.storage_.val);
        } else {
            new (&storage_.err) E(o.storage_.err);
        }
    }

    Expected& operator=(const Expected& o) {
        if (this != &o) {
            destroy();
            has_val_ = o.has_val_;
            if (has_val_) {
                new (&storage_.val) T(o.storage_.val);
            } else {
                new (&storage_.err) E(o.storage_.err);
            }
        }
        return *this;
    }

    // Move
    Expected(Expected&& o) noexcept : has_val_(o.has_val_) {
        if (has_val_) {
            new (&storage_.val) T(std::move(o.storage_.val));
        } else {
            new (&storage_.err) E(std::move(o.storage_.err));
        }
    }

    Expected& operator=(Expected&& o) noexcept {
        if (this != &o) {
            destroy();
            has_val_ = o.has_val_;
            if (has_val_) {
                new (&storage_.val) T(std::move(o.storage_.val));
            } else {
                new (&storage_.err) E(std::move(o.storage_.err));
            }
        }
        return *this;
    }

    ~Expected() { destroy(); }

    // Observers
    bool has_value() const { return has_val_; }
    explicit operator bool() const { return has_val_; }

    const T& value() const& { return storage_.val; }
    T& value() & { return storage_.val; }
    T&& value() && { return std::move(storage_.val); }

    const E& error() const& { return storage_.err; }
    E& error() & { return storage_.err; }

    // ── Monadic operations ───────────────────────────────────────────

    // and_then (flatmap): Expected<T,E> → (T → Expected<U,E>) → Expected<U,E>
    template <typename F>
    auto and_then(F&& f) const -> std::invoke_result_t<F, const T&> {
        using RetType = std::invoke_result_t<F, const T&>;
        if (has_val_) {
            return f(storage_.val);
        }
        return RetType(typename RetType::UnexpectedTag{}, storage_.err);
    }

    // map (fmap): Expected<T,E> → (T → U) → Expected<U,E>
    template <typename F>
    auto map(F&& f) const -> Expected<std::invoke_result_t<F, const T&>, E> {
        using U = std::invoke_result_t<F, const T&>;
        if (has_val_) {
            return Expected<U, E>(f(storage_.val));
        }
        return Expected<U, E>(typename Expected<U, E>::UnexpectedTag{}, storage_.err);
    }
};

// Convenience factory for errors
template <typename T, typename E>
Expected<T, E> make_unexpected(E e) {
    return Expected<T, E>(typename Expected<T, E>::UnexpectedTag{}, std::move(e));
}

// ── Pipeline stages ──────────────────────────────────────────────────

using Result = Expected<std::string, std::string>;
using IntResult = Expected<int, std::string>;

// Step 1: "read file" — simulates reading content from a filename
Expected<std::string, std::string> read_file(const std::string& path) {
    std::cout << "  [read_file] attempting '" << path << "'\n";
    if (path.empty()) {
        return make_unexpected<std::string, std::string>("read_file: empty path");
    }
    if (path == "missing.txt") {
        return make_unexpected<std::string, std::string>("read_file: file not found");
    }
    // Simulate file contents based on path
    if (path == "good.txt") return std::string{"42"};
    if (path == "bad.txt") return std::string{"hello"};
    if (path == "range.txt") return std::string{"9999"};
    return std::string{"0"};
}

// Step 2: parse_int — parse a string to int
Expected<int, std::string> parse_int(const std::string& s) {
    std::cout << "  [parse_int] parsing '" << s << "'\n";
    try {
        std::size_t pos = 0;
        int val = std::stoi(s, &pos);
        if (pos != s.size()) {
            return make_unexpected<int, std::string>(
                "parse_int: trailing characters in '" + s + "'");
        }
        return val;
    } catch (...) {
        return make_unexpected<int, std::string>(
            "parse_int: not a valid integer '" + s + "'");
    }
}

// Step 3: validate_range — check 0..100
Expected<int, std::string> validate_range(int v) {
    std::cout << "  [validate_range] checking " << v << "\n";
    if (v < 0 || v > 100) {
        return make_unexpected<int, std::string>(
            "validate_range: " + std::to_string(v) + " not in [0,100]");
    }
    return v;
}

// ── Full pipeline ────────────────────────────────────────────────────

void run_pipeline(const std::string& label, const std::string& path) {
    std::cout << "\n=== " << label << " (path=\"" << path << "\") ===\n";

    auto result = read_file(path)
        .and_then(parse_int)
        .and_then(validate_range);

    if (result) {
        std::cout << "  ✓ SUCCESS: value = " << result.value() << "\n";
    } else {
        std::cout << "  ✗ ERROR:   " << result.error() << "\n";
    }
}

// ── map() demo ───────────────────────────────────────────────────────

void demo_map() {
    std::cout << "\n=== map() demo ===\n";
    auto result = read_file("good.txt")
        .and_then(parse_int)
        .and_then(validate_range)
        .map([](int v) -> std::string {
            return "The answer is " + std::to_string(v);
        });

    if (result) {
        std::cout << "  ✓ " << result.value() << "\n";
    } else {
        std::cout << "  ✗ " << result.error() << "\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────

int main() {
    std::cout << "Expected<T,E> monadic chaining demo\n";
    std::cout << "Pipeline: read_file → parse_int → validate_range\n";

    // Case 1: Full success — "good.txt" contains "42"
    run_pipeline("Case 1: Success", "good.txt");

    // Case 2: Error at step 1 — file not found
    run_pipeline("Case 2: Error at read_file", "missing.txt");

    // Case 3: Error at step 2 — "bad.txt" contains "hello" (not a number)
    run_pipeline("Case 3: Error at parse_int", "bad.txt");

    // Case 4: Error at step 3 — "range.txt" contains "9999" (out of range)
    run_pipeline("Case 4: Error at validate_range", "range.txt");

    // Case 5: map() transforms the final value
    demo_map();

    return 0;
}
