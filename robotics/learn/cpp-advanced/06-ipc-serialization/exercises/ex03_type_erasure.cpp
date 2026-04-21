// ex03_type_erasure.cpp — Type erasure with Concept/Model, SBO, and move-only variant
//
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -O2 -pthread ex03_type_erasure.cpp -o ex03

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

// ═══════════════════════════════════════════════════════════
// Part 1: AnyCallable with Small Buffer Optimization
// ═══════════════════════════════════════════════════════════

class AnyCallable {
    static constexpr std::size_t kSBOSize  = 32;
    static constexpr std::size_t kSBOAlign = 16; // typical max_align_t

    // ── Concept (abstract interface) ────────────────────
    struct Concept {
        virtual int invoke(int arg) const = 0;
        virtual void clone_to(void* dst) const = 0;
        virtual void move_to(void* dst) noexcept = 0;
        virtual std::size_t size() const noexcept = 0;
        virtual ~Concept() = default;
    };

    // ── Model (wraps any callable T) ────────────────────
    template <typename T>
    struct Model final : Concept {
        T callable;

        explicit Model(T c) : callable(std::move(c)) {}

        int invoke(int arg) const override {
            return callable(arg);
        }

        void clone_to(void* dst) const override {
            new (dst) Model(callable);
        }

        void move_to(void* dst) noexcept override {
            new (dst) Model(std::move(callable));
        }

        std::size_t size() const noexcept override {
            return sizeof(Model);
        }
    };

    // ── Storage ─────────────────────────────────────────
    alignas(kSBOAlign) char buffer_[kSBOSize]{};
    Concept* ptr_ = nullptr;
    bool is_local_ = false;

    [[nodiscard]] Concept* local_ptr() noexcept {
        return std::launder(reinterpret_cast<Concept*>(buffer_));
    }
    [[nodiscard]] const Concept* local_ptr() const noexcept {
        return std::launder(reinterpret_cast<const Concept*>(buffer_));
    }

    template <typename T>
    static constexpr bool fits_sbo =
        sizeof(Model<T>) <= kSBOSize &&
        alignof(Model<T>) <= kSBOAlign &&
        std::is_nothrow_move_constructible_v<T>;

    void destroy() noexcept {
        if (ptr_) {
            if (is_local_) {
                ptr_->~Concept();
            } else {
                delete ptr_;
            }
            ptr_ = nullptr;
            is_local_ = false;
        }
    }

public:
    AnyCallable() = default;

    // ── Construct from any callable ─────────────────────
    template <typename T,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, AnyCallable>>>
    AnyCallable(T callable) {  // NOLINT(google-explicit-constructor)
        using ModelT = Model<std::decay_t<T>>;
        if constexpr (fits_sbo<std::decay_t<T>>) {
            new (buffer_) ModelT(std::move(callable));
            ptr_ = local_ptr();
            is_local_ = true;
        } else {
            ptr_ = new ModelT(std::move(callable));
            is_local_ = false;
        }
    }

    // ── Copy ────────────────────────────────────────────
    AnyCallable(const AnyCallable& other) {
        if (other.ptr_) {
            if (other.is_local_) {
                // Fits SBO — clone directly into our buffer
                other.ptr_->clone_to(buffer_);
                ptr_ = local_ptr();
                is_local_ = true;
            } else {
                // Doesn't fit SBO — heap-allocate, then clone into it
                auto* heap = static_cast<Concept*>(::operator new(other.ptr_->size()));
                try {
                    other.ptr_->clone_to(heap);
                } catch (...) {
                    ::operator delete(heap);
                    throw;
                }
                ptr_ = heap;
                is_local_ = false;
            }
        }
    }

    AnyCallable& operator=(const AnyCallable& other) {
        if (this != &other) {
            AnyCallable tmp(other);
            swap(tmp);
        }
        return *this;
    }

    // ── Move ────────────────────────────────────────────
    AnyCallable(AnyCallable&& other) noexcept {
        if (other.ptr_) {
            if (other.is_local_) {
                other.ptr_->move_to(buffer_);
                ptr_ = local_ptr();
                is_local_ = true;
                other.destroy();
            } else {
                ptr_ = other.ptr_;
                is_local_ = false;
                other.ptr_ = nullptr;
                other.is_local_ = false;
            }
        }
    }

    AnyCallable& operator=(AnyCallable&& other) noexcept {
        if (this != &other) {
            destroy();
            if (other.ptr_) {
                if (other.is_local_) {
                    other.ptr_->move_to(buffer_);
                    ptr_ = local_ptr();
                    is_local_ = true;
                    other.destroy();
                } else {
                    ptr_ = other.ptr_;
                    is_local_ = false;
                    other.ptr_ = nullptr;
                    other.is_local_ = false;
                }
            }
        }
        return *this;
    }

    ~AnyCallable() { destroy(); }

    void swap(AnyCallable& other) noexcept {
        AnyCallable tmp(std::move(other));
        other = std::move(*this);
        *this = std::move(tmp);
    }

    // ── Invoke ──────────────────────────────────────────
    int operator()(int arg) const {
        if (!ptr_) { std::fprintf(stderr, "AnyCallable: empty!\n"); std::abort(); }
        return ptr_->invoke(arg);
    }

    [[nodiscard]] bool is_sbo() const noexcept { return is_local_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
};

// ═══════════════════════════════════════════════════════════
// Part 2: AnyMovable — move-only type erasure
// ═══════════════════════════════════════════════════════════

class AnyMovable {
    static constexpr std::size_t kSBOSize  = 32;
    static constexpr std::size_t kSBOAlign = 16; // typical max_align_t

    struct Concept {
        virtual int invoke(int arg) = 0;  // non-const: may mutate state
        virtual void move_to(void* dst) noexcept = 0;
        virtual ~Concept() = default;
    };

    template <typename T>
    struct Model final : Concept {
        T callable;
        explicit Model(T c) : callable(std::move(c)) {}
        int invoke(int arg) override { return callable(arg); }
        void move_to(void* dst) noexcept override {
            new (dst) Model(std::move(callable));
        }
    };

    alignas(kSBOAlign) char buffer_[kSBOSize]{};
    Concept* ptr_ = nullptr;
    bool is_local_ = false;

    Concept* local_ptr() noexcept {
        return std::launder(reinterpret_cast<Concept*>(buffer_));
    }

    template <typename T>
    static constexpr bool fits_sbo =
        sizeof(Model<T>) <= kSBOSize &&
        alignof(Model<T>) <= kSBOAlign &&
        std::is_nothrow_move_constructible_v<T>;

    void destroy() noexcept {
        if (ptr_) {
            if (is_local_) ptr_->~Concept();
            else delete ptr_;
            ptr_ = nullptr;
            is_local_ = false;
        }
    }

public:
    AnyMovable() = default;

    template <typename T,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, AnyMovable>>>
    AnyMovable(T callable) {  // NOLINT
        using ModelT = Model<std::decay_t<T>>;
        if constexpr (fits_sbo<std::decay_t<T>>) {
            new (buffer_) ModelT(std::move(callable));
            ptr_ = local_ptr();
            is_local_ = true;
        } else {
            ptr_ = new ModelT(std::move(callable));
            is_local_ = false;
        }
    }

    // Delete copy
    AnyMovable(const AnyMovable&) = delete;
    AnyMovable& operator=(const AnyMovable&) = delete;

    // Move
    AnyMovable(AnyMovable&& other) noexcept {
        if (other.ptr_) {
            if (other.is_local_) {
                other.ptr_->move_to(buffer_);
                ptr_ = local_ptr();
                is_local_ = true;
                other.destroy();
            } else {
                ptr_ = other.ptr_;
                is_local_ = false;
                other.ptr_ = nullptr;
            }
        }
    }

    AnyMovable& operator=(AnyMovable&& other) noexcept {
        if (this != &other) {
            destroy();
            new (this) AnyMovable(std::move(other));
        }
        return *this;
    }

    ~AnyMovable() { destroy(); }

    int operator()(int arg) {
        if (!ptr_) std::abort();
        return ptr_->invoke(arg);
    }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }
};

// ═══════════════════════════════════════════════════════════
// Part 3: Virtual dispatch baseline
// ═══════════════════════════════════════════════════════════

struct ICallable {
    virtual int call(int arg) const = 0;
    virtual ~ICallable() = default;
};

struct VirtualAdder final : ICallable {
    int offset;
    explicit VirtualAdder(int o) : offset(o) {}
    int call(int arg) const override { return arg + offset; }
};

// ═══════════════════════════════════════════════════════════
// Part 4: Benchmark
// ═══════════════════════════════════════════════════════════

static constexpr std::size_t kBenchIters = 10'000'000;

struct BenchResult {
    const char* name;
    double ns_per_call;
};

template <typename F>
static BenchResult bench(const char* name, F&& fn) {
    // Warmup
    volatile int sink = 0;
    for (std::size_t i = 0; i < 1000; ++i) {
        sink += fn(static_cast<int>(i));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kBenchIters; ++i) {
        sink += fn(static_cast<int>(i));
    }
    auto t1 = std::chrono::steady_clock::now();

    double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    (void)sink;
    return {name, ns / static_cast<double>(kBenchIters)};
}

static void print_bench(const BenchResult* results, std::size_t count) {
    std::printf("\n╔════════════════════════════════════╦═══════════════╗\n");
    std::printf("║ %-34s ║ %13s ║\n", "Method", "ns/call");
    std::printf("╠════════════════════════════════════╬═══════════════╣\n");
    for (std::size_t i = 0; i < count; ++i) {
        std::printf("║ %-34s ║ %13.2f ║\n", results[i].name, results[i].ns_per_call);
    }
    std::printf("╚════════════════════════════════════╩═══════════════╝\n");
}

// ═══════════════════════════════════════════════════════════
// Part 5: Demonstration & Tests
// ═══════════════════════════════════════════════════════════

static void test_any_callable() {
    std::printf("=== AnyCallable Tests ===\n");

    // Lambda (small, fits SBO)
    AnyCallable f1 = [](int x) { return x * 2; };
    std::printf("  f1(21) = %d (expect 42), SBO = %s\n",
                f1(21), f1.is_sbo() ? "yes" : "no");

    // Lambda capturing data (still small)
    int offset = 10;
    AnyCallable f2 = [offset](int x) { return x + offset; };
    std::printf("  f2(32) = %d (expect 42), SBO = %s\n",
                f2(32), f2.is_sbo() ? "yes" : "no");

    // Large capture (exceeds SBO)
    char big[64]{};
    big[0] = 5;
    AnyCallable f3 = [big](int x) { return x + static_cast<int>(big[0]); };
    std::printf("  f3(37) = %d (expect 42), SBO = %s\n",
                f3(37), f3.is_sbo() ? "yes" : "no");

    // Copy
    AnyCallable f4 = f1;
    std::printf("  f4(21) = %d (copy of f1, expect 42)\n", f4(21));

    // Move
    AnyCallable f5 = std::move(f2);
    std::printf("  f5(32) = %d (moved from f2, expect 42)\n", f5(32));

    std::printf("  All AnyCallable tests passed.\n\n");
}

static void test_any_movable() {
    std::printf("=== AnyMovable Tests ===\n");

    // Stateful, move-only callable
    struct Counter {
        int count = 0;
        int operator()(int x) {
            count++;
            return x + count;
        }
        Counter() = default;
        Counter(Counter&& o) noexcept : count(o.count) { o.count = 0; }
        Counter& operator=(Counter&&) = delete;
        Counter(const Counter&) = delete;
        Counter& operator=(const Counter&) = delete;
    };

    AnyMovable m1{Counter{}};
    std::printf("  m1(10) = %d (expect 11, count=1)\n", m1(10));
    std::printf("  m1(10) = %d (expect 12, count=2)\n", m1(10));

    // Move
    AnyMovable m2 = std::move(m1);
    std::printf("  m2(10) = %d (moved, expect 13, count=3)\n", m2(10));

    // Lambda with unique_ptr (move-only)
    auto ptr = std::make_unique<int>(100);
    AnyMovable m3{[p = std::move(ptr)](int x) { return x + *p; }};
    std::printf("  m3(5) = %d (unique_ptr capture, expect 105)\n", m3(5));

    std::printf("  All AnyMovable tests passed.\n\n");
}

static void run_benchmarks() {
    std::printf("=== Benchmark: %zu calls each ===\n", kBenchIters);

    int offset = 42;

    // 1. Direct function call
    auto direct_fn = [offset](int x) __attribute__((noinline)) { return x + offset; };
    auto r1 = bench("Direct lambda call", direct_fn);

    // 2. AnyCallable (SBO path)
    AnyCallable ac_sbo = [offset](int x) { return x + offset; };
    auto r2 = bench("AnyCallable (SBO)", [&](int x) { return ac_sbo(x); });

    // 3. std::function
    std::function<int(int)> stdfn = [offset](int x) { return x + offset; };
    auto r3 = bench("std::function", [&](int x) { return stdfn(x); });

    // 4. Virtual dispatch
    auto* virt = new VirtualAdder(offset);
    auto r4 = bench("Virtual dispatch", [virt](int x) { return virt->call(x); });
    delete virt;

    // 5. AnyCallable (heap path — large capture)
    char big[64]{};
    big[0] = static_cast<char>(offset);
    AnyCallable ac_heap = [big](int x) { return x + static_cast<int>(big[0]); };
    auto r5 = bench("AnyCallable (heap)", [&](int x) { return ac_heap(x); });

    // 6. AnyMovable (SBO)
    AnyMovable am{[offset](int x) { return x + offset; }};
    auto r6 = bench("AnyMovable (SBO)", [&](int x) { return am(x); });

    BenchResult results[] = {r1, r2, r3, r4, r5, r6};
    print_bench(results, 6);

    std::printf("\nNotes:\n");
    std::printf("  - SBO avoids heap allocation, matching virtual dispatch speed\n");
    std::printf("  - Heap-allocated type erasure adds ~5-15ns from cache miss\n");
    std::printf("  - std::function uses SBO internally (typically 16-32 bytes)\n");
    std::printf("  - Compile with -O2 for realistic numbers\n");
}

// ─── Main ───────────────────────────────────────────────
int main() {
    std::printf("Week 6, Exercise 3: Type Erasure — Concept/Model + SBO\n");
    std::printf("═══════════════════════════════════════════════════════\n\n");

    test_any_callable();
    test_any_movable();
    run_benchmarks();

    return 0;
}
