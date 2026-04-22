# Module 14: Exception Safety Guarantees

## Overview

Exception safety is about what guarantees your code provides when an exception
is thrown. It's not about *preventing* exceptions — it's about maintaining
program correctness *despite* exceptions. This is one of the most important
and most misunderstood topics in C++.

The concept was formalized by **Dave Abrahams** in the late 1990s during the
standardization of the STL. His work defined the vocabulary we still use today.

---

## The Four Exception Safety Guarantee Levels

### Level 0: No Guarantee (NEVER acceptable)

The function may leak resources, violate invariants, or corrupt data if an
exception is thrown. This is a **bug**, not a design choice.

```cpp
void bad_transfer(Account& from, Account& to, int amount) {
    from.balance -= amount;   // Step 1: debit
    // If this throws, 'from' is debited but 'to' never credited!
    to.balance += amount;     // Step 2: credit — may throw
}
```

### Level 1: Basic Guarantee

If an exception is thrown:
- All invariants are preserved
- No resources are leaked
- Objects are in a **valid but unspecified** state

This is the **minimum acceptable** level for production code.

```cpp
void basic_transfer(Account& from, Account& to, int amount) {
    std::lock_guard lock(mutex_);
    from.withdraw(amount);   // may throw, but Account stays valid
    try {
        to.deposit(amount);  // may throw
    } catch (...) {
        from.deposit(amount); // rollback — basic guarantee
        throw;
    }
}
```

### Level 2: Strong Guarantee (commit-or-rollback)

If an exception is thrown, the program state is **exactly as it was before**
the function was called. Either the operation fully succeeds, or it's as if
it never happened.

This is the **copy-and-swap** level.

```cpp
void strong_transfer(Account& from, Account& to, int amount) {
    Account from_copy = from;  // work on copies
    Account to_copy = to;
    from_copy.withdraw(amount);
    to_copy.deposit(amount);
    // Only now commit — swap is noexcept
    swap(from, from_copy);     // noexcept
    swap(to, to_copy);         // noexcept
}
```

### Level 3: No-Throw Guarantee (`noexcept`)

The operation **never** throws an exception. Period. If something goes wrong
internally, it handles it without propagating an exception.

```cpp
void nothrow_swap(Account& a, Account& b) noexcept {
    std::swap(a.balance_, b.balance_);  // primitive swap, can't throw
}
```

Functions that **must** be noexcept:
- Destructors (implicitly noexcept since C++11)
- Move constructors and move assignment (for container efficiency)
- `swap()` functions
- Deallocation functions (`operator delete`)

---

## The `noexcept` Specifier

### When to Use It

Mark a function `noexcept` when:
1. It genuinely cannot throw (e.g., swapping integers)
2. You *want* the compiler to assume it won't throw (move operations!)
3. Failure should terminate rather than propagate

```cpp
// Unconditional noexcept
void swap(MyClass& other) noexcept;

// Conditional noexcept — noexcept if T's move ctor is noexcept
template<typename T>
void Container<T>::swap(Container& other) noexcept(
    std::is_nothrow_move_constructible_v<T>
);
```

### Impact on Optimization

The compiler can optimize `noexcept` functions more aggressively:
- No need to maintain stack unwinding information
- Move operations used instead of copies in containers
- `std::vector::push_back` will COPY instead of MOVE if the move ctor isn't `noexcept`

### The `noexcept` Operator

The `noexcept()` operator is a compile-time check:

```cpp
static_assert(noexcept(std::declval<int&>() = 5));  // int assignment is noexcept

// Conditional noexcept using the operator
template<typename T>
void wrapper(T& obj) noexcept(noexcept(obj.do_thing())) {
    obj.do_thing();
}
```

---

## Copy-and-Swap Idiom

The classic technique for providing the **strong guarantee** on assignment:

```cpp
class Widget {
    int* data_;
    size_t size_;

public:
    // Copy constructor — may throw (allocates)
    Widget(const Widget& other)
        : data_(new int[other.size_])
        , size_(other.size_)
    {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // Swap — must be noexcept
    friend void swap(Widget& a, Widget& b) noexcept {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.size_, b.size_);
    }

    // Assignment via copy-and-swap — strong guarantee
    Widget& operator=(Widget other) {  // note: pass by VALUE (copy happens here)
        swap(*this, other);            // noexcept swap
        return *this;                  // old data destroyed when 'other' dies
    }
};
```

Why the naive approach fails:

```cpp
// WRONG — not exception safe!
Widget& operator=(const Widget& other) {
    delete[] data_;                    // point of no return!
    data_ = new int[other.size_];     // THROWS? data_ is dangling!
    std::copy(...);
    size_ = other.size_;
    return *this;
}
```

---

## RAII: The Foundation of Exception Safety

**Resource Acquisition Is Initialization** — every resource is owned by an
object whose destructor releases it. This is THE mechanism that makes C++
exception safety work.

```cpp
// BAD: manual resource management
void unsafe() {
    FILE* f = fopen("data.txt", "r");
    int* buf = new int[1024];
    process(f, buf);    // if this throws, f and buf leak!
    delete[] buf;
    fclose(f);
}

// GOOD: RAII wrappers
void safe() {
    auto f = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen("data.txt", "r"), &fclose);
    auto buf = std::make_unique<int[]>(1024);
    process(f.get(), buf.get());  // if this throws, destructors clean up
}
```

---

## `std::uncaught_exceptions()` and Scope Guards

C++17 added `std::uncaught_exceptions()` (note: plural!) which returns the
*number* of uncaught exceptions currently in flight. This enables scope guards
that behave differently on success vs failure.

```cpp
class ScopeFailure {
    int exceptions_on_entry_;
    std::function<void()> rollback_;
public:
    ScopeFailure(std::function<void()> fn)
        : exceptions_on_entry_(std::uncaught_exceptions())
        , rollback_(std::move(fn)) {}

    ~ScopeFailure() {
        // Only call rollback if we're unwinding due to a NEW exception
        if (std::uncaught_exceptions() > exceptions_on_entry_) {
            rollback_();
        }
    }
};
```

---

## Exception Safety in Constructors

**The fully-constructed-subobject rule**: If a constructor throws, destructors
are called for all fully constructed base classes and members (in reverse order
of construction), but NOT for the object being constructed (its destructor
won't run).

```cpp
class Composite {
    ResourceA a_;   // constructed first
    ResourceB b_;   // constructed second
    ResourceC c_;   // constructed third — THROWS!

public:
    Composite()
        : a_()    // constructed, will be destroyed
        , b_()    // constructed, will be destroyed
        , c_()    // THROWS — b_ then a_ destructors called
    {}            // ~Composite() is NOT called
};
```

This is why RAII members are essential — raw pointers in members won't be
cleaned up if a later member's construction throws.

---

## Exception Safety in Containers

### Why `push_back` Provides the Strong Guarantee

`std::vector::push_back` either succeeds or leaves the vector unchanged:

1. If reallocation needed and copy/move of existing elements throws → old
   buffer is still intact, new buffer is discarded
2. If constructing the new element throws → size is not incremented

**Critical**: This only works efficiently if the move constructor is `noexcept`.
Otherwise, `vector` must COPY elements during reallocation (because if a move
throws halfway through, the old buffer is already partially moved-from).

### `std::move_if_noexcept`

```cpp
template<typename T>
auto move_if_noexcept(T& x) noexcept {
    // Returns T&& if T's move ctor is noexcept, otherwise const T&
    if constexpr (std::is_nothrow_move_constructible_v<T>) {
        return std::move(x);
    } else {
        return x;  // returns lvalue — will copy
    }
}
```

This is what `std::vector` uses internally during reallocation. If your move
constructor isn't `noexcept`, your "fast moves" silently become "slow copies".

---

## Designing Exception-Safe Interfaces

### The Commit-or-Rollback Pattern

```
1. Do all the work that might throw (on temporaries/copies)
2. Commit the results using only noexcept operations (swap, pointer assignment)
```

This is the generalized form of copy-and-swap.

### Multi-Step Operations

When multiple objects must be updated atomically:

```cpp
void atomic_update(A& a, B& b, C& c) {
    A a_new = compute_new_a(a);  // may throw — a unchanged
    B b_new = compute_new_b(b);  // may throw — a,b unchanged
    C c_new = compute_new_c(c);  // may throw — a,b,c unchanged
    // COMMIT PHASE — all noexcept
    swap(a, a_new);
    swap(b, b_new);
    swap(c, c_new);
}
```

---

## Exception Safety in Multi-Threaded Code

Exceptions and threads interact dangerously:

1. An exception thrown in a `std::thread` that isn't caught calls `std::terminate`
2. Use `std::promise`/`std::future` to transport exceptions across threads
3. Lock guards (RAII) ensure mutexes are released even if exceptions occur
4. `std::shared_mutex` with RAII locks provides exception-safe concurrent access

```cpp
void thread_work(std::promise<int>& p) {
    try {
        p.set_value(compute());  // success
    } catch (...) {
        p.set_exception(std::current_exception());  // propagate to caller
    }
}
```

---

## Real-World: AMR Robot Firmware and `-fno-exceptions`

The warehouse robot firmware (running on Zephyr RTOS) compiles with `-fno-exceptions`
because:

1. **Deterministic timing**: Exception handling adds unpredictable latency.
   A motor controller ISR must complete in microseconds — stack unwinding
   is not acceptable.

2. **Code size**: Exception tables and unwind info add significant binary size
   on embedded ARM Cortex-M targets.

3. **No heap**: Many firmware modules run without dynamic allocation.
   `std::bad_alloc` doesn't make sense.

**What replaces exceptions in firmware?**
- Return codes (often wrapped in `Result<T, Error>` types)
- Error callback / handler registration
- `assert()` / `__builtin_trap()` for unrecoverable errors
- `std::expected<T, E>` (C++23, or backported)
- Safety-Monitor watchdog: if a subsystem fails, the watchdog reboots it

This is a valid engineering tradeoff — not a rejection of exception safety
principles. The same guarantee levels apply, just enforced through different
mechanisms.

---

## Common Mistakes

### 1. Catching by Value (Slicing)

```cpp
try { throw DerivedError("oops"); }
catch (BaseError e) {     // SLICED! DerivedError info lost
    // e is a BaseError, not DerivedError
}

// CORRECT: catch by const reference
catch (const BaseError& e) { ... }
```

### 2. Throwing in Destructors

If a destructor throws during stack unwinding (another exception is already
in flight), `std::terminate()` is called. Since C++11, destructors are
implicitly `noexcept`.

```cpp
~Resource() {
    try { cleanup(); }    // cleanup might fail
    catch (...) {
        log_error();      // swallow — NEVER let destructors throw
    }
}
```

### 3. Exception-Unsafe Swap

If `swap()` can throw, copy-and-swap gives NO guarantee at all:

```cpp
// WRONG: swap that allocates (may throw)
void swap(BigObject& a, BigObject& b) {
    BigObject temp = a;  // COPIES! May throw!
    a = b;
    b = temp;
}

// CORRECT: swap by exchanging internals
void swap(BigObject& a, BigObject& b) noexcept {
    using std::swap;
    swap(a.ptr_, b.ptr_);    // pointer swap — noexcept
    swap(a.size_, b.size_);  // int swap — noexcept
}
```

### 4. Forgetting `noexcept` on Move Operations

This silently degrades `std::vector` performance from O(1) amortized moves
to O(N) copies during reallocation. See exercise 04.

### 5. Resource Acquisition Before RAII

```cpp
void leak() {
    Resource* r = acquire();     // raw pointer!
    do_work();                   // THROWS — r leaks!
    std::unique_ptr<Resource> p(r);  // too late
}

void safe() {
    auto p = std::unique_ptr<Resource>(acquire());  // RAII immediately
    do_work();   // if throws, p's destructor cleans up
}
```

---

## Summary Table

| Guarantee | State After Exception | Resource Leaks? | Example |
|-----------|----------------------|-----------------|---------|
| No-throw  | N/A (never throws)  | No              | `swap()`, destructors |
| Strong    | Rolled back          | No              | `vector::push_back` |
| Basic     | Valid but changed    | No              | `vector::insert` (sometimes) |
| None      | Anything             | Maybe           | **BUG** |

---

## Key Takeaways

1. Every function should provide at least the **basic guarantee**
2. Use **RAII** for all resource management — it's the foundation
3. Mark move ctors/assignment and swap as **`noexcept`** — containers depend on it
4. Use **copy-and-swap** when you need the strong guarantee on assignment
5. For multi-step operations, do all throwing work first, then commit with noexcept ops
6. Never throw in destructors
7. Catch by `const&`, never by value
8. `noexcept` is not just documentation — it changes runtime behavior
