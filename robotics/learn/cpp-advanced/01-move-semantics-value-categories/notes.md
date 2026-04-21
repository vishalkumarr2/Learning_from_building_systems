# Week 1: Move Semantics, Value Categories & Compile-Time C++

> **Goal**: Write zero-copy, compile-time-evaluated code.  
> **Prerequisite**: You know basic C++ templates, smart pointers, and have used `std::move` at least once without fully understanding it.

---

## Table of Contents

1. [Value Categories — The 5-Fold Taxonomy](#1-value-categories)
2. [Move Semantics — What Actually Happens](#2-move-semantics)
3. [Perfect Forwarding — Reference Collapsing & std::forward](#3-perfect-forwarding)
4. [constexpr / consteval / constinit Evolution](#4-compile-time-evaluation)
5. [C++20 Concepts — Replacing SFINAE](#5-concepts)
6. [ROS Tie-ins — Why This Matters for Robots](#6-ros-tie-ins)

---

## 1. Value Categories

### The Problem These Solve

Every expression in C++ has two independent properties:
- **A type** (int, std::string, const Widget&, etc.)
- **A value category** (lvalue, rvalue, etc.)

The compiler needs value categories to decide:
1. Which overload to pick (`f(T&)` vs `f(T&&)`)
2. Whether to move or copy
3. Whether you can take the address of an expression
4. Whether lifetime extension applies

### The Taxonomy

Before C++11, we had just **lvalues** and **rvalues**. C++11 added three more.
Here's the tree:

```
                    expression
                   /          \
               glvalue       rvalue
              /       \     /      \
           lvalue    xvalue      prvalue
```

Two independent axes:
- **Has identity** (glvalue) vs **no identity** (prvalue) — Can you take its address?
- **Movable** (rvalue) vs **not movable from** (lvalue) — Can you steal its guts?

| Category | Has Identity? | Movable? | Examples |
|----------|:---:|:---:|---|
| **lvalue** | ✅ | ❌ | `x`, `*p`, `a[3]`, `++i`, string literal `"hello"` |
| **xvalue** | ✅ | ✅ | `std::move(x)`, `static_cast<T&&>(x)`, member of rvalue |
| **prvalue** | ❌ | ✅ | `42`, `3.14`, `x + y`, `f()` returning by value, lambda |
| **glvalue** | ✅ | — | lvalue ∪ xvalue (anything with identity) |
| **rvalue** | — | ✅ | xvalue ∪ prvalue (anything movable) |

### Mental Model: The Identity-Movability Grid

```
                    Has Identity
                   YES        NO
              ┌──────────┬──────────┐
  Can Move  N │  lvalue  │ (empty)  │
  From      O │  int x;  │  N/A     │
              ├──────────┼──────────┤
  Can Move  Y │  xvalue  │ prvalue  │
  From      E │ move(x)  │  42      │
  (rvalue)  S │          │  x + y   │
              └──────────┴──────────┘
```

The "NO identity, NO move" cell is empty — it would mean an expression that you can't refer to AND can't move from. That doesn't exist in C++.

### Why Named Rvalue References Are Lvalues (The #1 Gotcha)

This is the most counter-intuitive rule in C++11:

```cpp
void f(int& x)  { std::cout << "lvalue\n"; }
void f(int&& x) { std::cout << "rvalue\n"; }

void g(int&& x) {
    // x is declared as int&&, but...
    f(x);   // calls f(int&) — x is an LVALUE here!
}
```

**Why?** Because `x` has a **name**. You can take its address (`&x`). It persists.
The rule: **if it has a name, it's an lvalue**. Full stop.

This is not a quirk — it's a safety feature. If `x` were treated as an rvalue inside `g()`, you could accidentally move from it twice:

```cpp
void g(int&& x) {
    use(x);           // first use — if this moved from x, then...
    use_again(x);     // ...this would use a moved-from value. BAD.
}
```

By making named rvalue references lvalues, the language forces you to be explicit:
```cpp
void g(int&& x) {
    use(x);                 // uses as lvalue (safe, just reads)
    use_again(std::move(x)); // explicit: "I'm done with x, take it"
}
```

### How To Test Value Categories

The compiler knows value categories. You can query them:

```cpp
#include <type_traits>

// decltype(expr) for expressions:
// - lvalue  → decltype gives T&
// - xvalue  → decltype gives T&&  
// - prvalue → decltype gives T

int x = 42;
static_assert(std::is_lvalue_reference_v<decltype((x))>);      // (x) is lvalue
static_assert(std::is_rvalue_reference_v<decltype((std::move(x)))>); // xvalue
static_assert(!std::is_reference_v<decltype((42))>);            // prvalue
```

**Note the double parentheses**: `decltype(x)` gives the declared type of `x` (int).
`decltype((x))` gives the value category of the expression `(x)` (int& — lvalue).

---

## 2. Move Semantics

### What "Move" Actually Means

Move is **not** a memory operation. It's a **permission**.

A move constructor/assignment says: "I'm allowed to steal the resources of the source object, because the caller promised it's going away (or they explicitly consented via `std::move`)."

For a `std::string`:
```
Before move:
  src: [ptr=0x1000, size=100, cap=128]    heap: "Hello world......"
  dst: [ptr=nullptr, size=0, cap=0]

After move:
  src: [ptr=nullptr, size=0, cap=0]        heap: "Hello world......"
  dst: [ptr=0x1000, size=100, cap=128]    ↑ same address, no copy!
```

At the assembly level, a move of a string is typically 3-4 `mov` instructions (swap pointers and sizes). A copy is `malloc` + `memcpy`.

### The Moved-From State

The standard says a moved-from object is in a **valid but unspecified state**.

"Valid" means:
- You can destroy it (dtor must work)
- You can assign to it (reset it)

"Unspecified" means:
- You can't assume it's empty
- You can't assume it still has the old value
- You can't assume anything about `size()`, `capacity()`, etc.

In practice, most STL implementations leave moved-from containers empty, but **you must not rely on this**.

The only safe operations on a moved-from object:
1. Destroy it
2. Assign a new value to it
3. Call functions with no preconditions (like `empty()`, `size()`)

### Rule of Five

If your class manages a resource (file descriptor, raw memory, GPU handle, mutex lock), you need ALL five special members:

```
┌─────────────────────────────────────────────┐
│  The Rule of Five                           │
│                                             │
│  1. Destructor         ~T()                 │
│  2. Copy constructor   T(const T&)          │
│  3. Copy assignment    T& operator=(const T&│)
│  4. Move constructor   T(T&&) noexcept      │
│  5. Move assignment    T& operator=(T&&) noexcept
│                                             │
│  If you write ANY of these, write ALL of them│
│  Or explicitly = delete the ones you don't  │
│  want.                                      │
└─────────────────────────────────────────────┘
```

**Why noexcept on moves?** `std::vector` uses move only if it's `noexcept`. If your move might throw, `vector::push_back` falls back to **copying** during reallocation. This is because if a move fails halfway through reallocating N elements, you've corrupted both the old and new storage. With copies, the old storage is untouched if one copy fails.

### Return Value Optimization (RVO/NRVO)

The compiler can **elide** copies and moves entirely:

```cpp
Widget make_widget() {
    Widget w;           // constructed directly in caller's memory
    w.configure(42);
    return w;           // NO copy, NO move — NRVO
}

Widget w = make_widget();  // just one construction
```

RVO (unnamed) is **mandatory** since C++17.  
NRVO (named) is still optional but virtually all compilers do it.

When NRVO **can't** apply:
- Returning different local variables on different paths
- Returning a function parameter
- Returning a global or member variable

---

## 3. Perfect Forwarding

### The Problem

You want to write a wrapper function that passes arguments through **exactly** as they were passed in — preserving lvalue/rvalue-ness, const-ness, everything.

```cpp
// You want: make<Widget>(1, "hello", std::move(resource))
// To call:  new Widget(1, "hello", std::move(resource))
// With ZERO extra copies or moves.
```

### Reference Collapsing Rules

When you combine references (through templates or typedef), C++ collapses them:

```
T& &   → T&     (lvalue ref to lvalue ref → lvalue ref)
T& &&  → T&     (rvalue ref to lvalue ref → lvalue ref)
T&& &  → T&     (lvalue ref to rvalue ref → lvalue ref)
T&& && → T&&    (rvalue ref to rvalue ref → rvalue ref)
```

**Simple rule**: If EITHER reference is an lvalue reference, the result is an lvalue reference.  
Rvalue reference only survives if BOTH are rvalue references.

```
       &    &&
     ┌─────┬──────┐
  &  │  &  │  &   │  ← lvalue always wins
     ├─────┼──────┤
  && │  &  │  &&  │  ← rvalue only if both &&
     └─────┴──────┘
```

### How T&& in Templates Works (Forwarding References)

When `T` is a template parameter and you write `T&&`, it's NOT an rvalue reference — it's a **forwarding reference** (formerly "universal reference"):

```cpp
template<typename T>
void f(T&& x);     // forwarding reference — binds to ANYTHING

void f(int&& x);   // plain rvalue reference — binds to rvalues only
```

Deduction rules for `f(T&& x)`:

| Argument | T deduced as | x type (after collapsing) |
|---|---|---|
| `int a; f(a)` | `int&` | `int& && → int&` |
| `const int b; f(b)` | `const int&` | `const int& && → const int&` |
| `f(42)` | `int` | `int&&` |
| `f(std::move(a))` | `int` | `int&&` |

### std::forward — The Cast That Preserves Categories

`std::forward<T>(x)` is a conditional cast:
- If T is an lvalue reference (like `int&`): returns x as lvalue (no cast)
- If T is not a reference (like `int`): returns x as rvalue (casts to `T&&`)

Implementation (simplified):
```cpp
template<typename T>
T&& forward(std::remove_reference_t<T>& t) noexcept {
    return static_cast<T&&>(t);
}
```

Walk through the cases:
```
T = int&:   static_cast<int& &&>(t) → static_cast<int&>(t)   → lvalue ✓
T = int:    static_cast<int&&>(t)                              → rvalue ✓
```

### auto&& in Range-For

`auto&&` in a range-for loop is a forwarding reference that binds to anything:

```cpp
std::vector<std::string> v = {"a", "b", "c"};

for (auto&& elem : v) {
    // elem is std::string& (lvalue ref, because v yields lvalues)
}

for (auto&& elem : get_temporary_vector()) {
    // elem is std::string&& (rvalue ref, because temporaries yield rvalues)
    // Lifetime of the temporary vector is extended to the loop
}
```

This is why `auto&&` is the "safest" range-for binding — it works for everything without extra copies.

---

## 4. constexpr / consteval / constinit Evolution

### The Timeline

```
C++11: constexpr introduced
       - Single return statement only
       - No loops, no local variables
       - constexpr functions can run at compile-time OR runtime

C++14: constexpr relaxed
       - Loops, local variables, multiple returns
       - Still no: try/catch, new/delete, reinterpret_cast

C++17: constexpr if, constexpr lambdas
       - if constexpr for compile-time branching
       - Lambdas can be constexpr

C++20: consteval, constinit, constexpr new/delete, virtual, try/catch
       - consteval = MUST be compile-time (immediate function)
       - constinit = variable initialized at compile-time (but mutable at runtime)
       - constexpr containers (std::string, std::vector in constexpr context)

C++23: constexpr <cmath>, static constexpr in constexpr functions
       - More standard library constexpr
       - constexpr-compatible static variables
```

### constexpr vs consteval vs constinit

```
┌───────────────────────────────────────────────────────────────┐
│ Keyword      │ When             │ What                        │
├──────────────┼──────────────────┼─────────────────────────────┤
│ constexpr    │ Compile OR run   │ Function CAN be evaluated   │
│              │                  │ at compile time             │
├──────────────┼──────────────────┼─────────────────────────────┤
│ consteval    │ Compile ONLY     │ Function MUST be evaluated  │
│  (C++20)     │                  │ at compile time             │
├──────────────┼──────────────────┼─────────────────────────────┤
│ constinit    │ Compile init,    │ Variable initialized at     │
│  (C++20)     │ runtime mutate   │ compile time, mutable later │
└──────────────┴──────────────────┴─────────────────────────────┘
```

### Why constinit Matters (The Static Init Order Fiasco)

In C++, the order of initialization of static variables in different translation units is **undefined**.

```cpp
// file_a.cpp
std::string global_path = "/etc/config";  // when does this init?

// file_b.cpp
extern std::string global_path;
std::string config_file = global_path + "/app.conf";  // might be EMPTY!
```

`constinit` prevents this:
```cpp
// Must be initialized at compile time — so it's ready before main()
constinit const char* global_path = "/etc/config";
```

If the initializer can't be computed at compile time, you get a compile error — catching the problem at build time, not as a mystery crash at runtime.

### Compile-Time Computation Patterns

**CRC32 table**: 256-entry lookup table, computed at compile time. Zero runtime cost.

```cpp
constexpr auto crc_table = [] {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ (0xEDB88320u & (-(crc & 1)));
        table[i] = crc;
    }
    return table;
}();
// This entire table is in .rodata — computed by the compiler
```

**String hashing for switch**: C++ can't switch on strings, but you can switch on compile-time hashes:

```cpp
constexpr uint64_t fnv1a(std::string_view sv) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (char c : sv) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// Usage:
switch (fnv1a(command)) {
    case fnv1a("start"):  start_robot(); break;
    case fnv1a("stop"):   stop_robot();  break;
    case fnv1a("pause"):  pause_robot(); break;
}
// The case labels are compile-time constants!
```

---

## 5. C++20 Concepts

### What Concepts Replace

Before C++20, constraining templates was painful:

```cpp
// SFINAE horror (C++11/14/17):
template<typename T, typename = std::enable_if_t<
    std::is_arithmetic_v<T> && !std::is_same_v<T, bool>>>
T clamp_to_range(T val, T lo, T hi);

// Error message when you pass a string:
//   "no matching function for call to 'clamp_to_range'"
//   "candidate template ignored: requirement 
//    'std::is_arithmetic_v<std::string>' was not satisfied"
//   (deep in the template instantiation stack)
```

With concepts:
```cpp
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

Arithmetic auto clamp_to_range(Arithmetic auto val, 
                                Arithmetic auto lo, 
                                Arithmetic auto hi);

// Error when you pass a string:
//   "constraints not satisfied for 'Arithmetic'"
//   Much clearer!
```

### Concept Syntax (4 Ways)

```cpp
// 1. Requires clause after template
template<typename T> requires Sortable<T>
void sort(T& container);

// 2. Trailing requires
template<typename T>
void sort(T& container) requires Sortable<T>;

// 3. Constrained template parameter
template<Sortable T>
void sort(T& container);

// 4. Abbreviated function template (terse)
void sort(Sortable auto& container);
```

All four are equivalent. Use #4 for simple cases, #1 or #2 for complex constraints.

### Writing Concepts with `requires` Expressions

```cpp
template<typename T>
concept Sensor = requires(T t) {
    // Simple requirement: expression must be valid
    t.read();
    
    // Compound requirement: valid + return type constraint
    { t.read() } -> std::convertible_to<double>;
    { t.is_valid() } -> std::same_as<bool>;
    
    // Nested requirement: a static_assert-like check
    requires sizeof(T) <= 128;
    
    // Type requirement: associated type must exist
    typename T::sample_type;
    
    // Compound with noexcept
    { t.is_valid() } noexcept -> std::same_as<bool>;
};
```

### Concept Subsumption (Overload Resolution)

When two concepts constrain an overload set, the **more constrained** concept wins:

```cpp
template<typename T> concept Integral = std::is_integral_v<T>;
template<typename T> concept SignedIntegral = Integral<T> && std::is_signed_v<T>;

void process(Integral auto x)       { /* generic integer */ }
void process(SignedIntegral auto x)  { /* signed-specific */ }

process(42);    // calls SignedIntegral version (more specific)
process(42u);   // calls Integral version (unsigned doesn't match Signed)
```

**How subsumption works**: The compiler decomposes concepts into atomic constraints.
`SignedIntegral` = `Integral<T> && is_signed_v<T>`. It **subsumes** `Integral` because it includes everything `Integral` requires, plus more. The compiler can prove that if `SignedIntegral` is satisfied, `Integral` is too.

This ONLY works with named concepts. Raw `requires` clauses don't participate in subsumption:
```cpp
// These are AMBIGUOUS — no subsumption possible
void f(auto x) requires std::is_integral_v<decltype(x)>;
void f(auto x) requires (std::is_integral_v<decltype(x)> && std::is_signed_v<decltype(x)>);
// Compiler error: ambiguous overload
```

---

## 6. ROS Tie-ins — Why This Matters for Robots

### Move Semantics in Sensor Pipelines

A typical robot sensor pipeline:

```
IMU → filter → estimator → controller → actuator
       ↑
   PointCloud → downsample → obstacle map → planner
```

Every `→` is a potential copy. With 30Hz IMU data at 48 bytes, that's fine. But a 640×480 depth image at 30Hz is **18 MB/sec**. Unnecessary copies destroy your cycle time.

**ROS1 Problem**: `ros::Publisher::publish()` takes a `const T&` — it always copies to the internal queue. Even if you're done with the message, you can't move it in.

**ROS2 Solution**: `rclcpp::Publisher::publish(std::unique_ptr<T>)` — zero-copy transport. The message is moved into shared memory, and subscribers get a loan. No copy at all.

```cpp
// ROS2 zero-copy publishing
auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
fill_pointcloud(*msg, lidar_data);
publisher->publish(std::move(msg));  // zero copy!
```

### Compile-Time Message Validation

In safety-critical robot code, you want to catch message format errors at **compile time**, not at runtime when the robot is moving:

```cpp
// CRC of message layout — computed at compile time
static_assert(message_crc<ImuMessage>() == 0xA3B7C921,
    "ImuMessage layout changed — update serialization code!");

// Message size validation
static_assert(sizeof(ImuMessage) <= MAX_DMA_BUFFER,
    "ImuMessage too large for DMA transfer");
```

### Concepts for Hardware Abstraction

Real robots have many sensor types. Concepts enforce the interface at compile time:

```cpp
template<typename T>
concept Sensor = requires(T t) {
    { t.read() } -> std::convertible_to<double>;
    { t.is_valid() } -> std::same_as<bool>;
    { T::sample_rate_hz } -> std::convertible_to<unsigned>;
};

// This function works with ANY sensor — IMU, encoder, lidar, etc.
// If someone adds a new sensor that doesn't have read(), they get a 
// compile error pointing at this concept, not a 200-line template error.
template<Sensor S>
void log_sensor(const S& sensor, Logger& log) {
    if (sensor.is_valid()) {
        log.write(sensor.read(), S::sample_rate_hz);
    }
}
```

### constinit for Static Robot Configuration

Robot parameters that are known at compile time but might be adjusted:

```cpp
// Initialized at compile time — no static init order issues
constinit double wheel_radius_m = 0.075;
constinit double wheel_base_m = 0.40;
constinit unsigned control_freq_hz = 200;
```

This is safer than `const` globals because `constinit` guarantees they're ready before `main()` — critical when ROS nodes initialize in unpredictable order.

---

## Summary — Week 1 Key Takeaways

```
┌─────────────────────────────────────────────────────────────────┐
│  1. EVERY expression has a value category (lvalue/xvalue/prvalue│)
│  2. Named rvalue references are LVALUES (most common mistake)   │
│  3. Move is a PERMISSION to steal, not a memory operation       │
│  4. std::forward preserves value category through templates     │
│  5. Reference collapsing: lvalue ref wins unless both are &&    │
│  6. constexpr = maybe compile-time; consteval = forced          │
│  7. Concepts replace SFINAE with readable constraints           │
│  8. Concept subsumption: more constrained wins in overloads     │
│  9. Mark moves noexcept or vector will copy instead             │
│ 10. ROS2 uses move semantics for zero-copy message transport    │
└─────────────────────────────────────────────────────────────────┘
```
