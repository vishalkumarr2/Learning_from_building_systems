# Module 12: C++20 Ranges

> **Compiler Requirements**: GCC 12+ or Clang 14+ recommended. GCC 10–11 has partial support (missing some views/algorithms). GCC 9.4 does NOT support ranges at all.

## Why Ranges?

The classic STL algorithm interface has a fundamental usability problem:

```cpp
// Classic: pass TWO iterators — easy to mismatch
std::sort(v.begin(), v.end());           // OK
std::sort(v.begin(), w.end());           // COMPILES! UB at runtime.
std::sort(v.begin(), v.begin() + 100);   // Out of bounds? No compile error.

// Ranges: pass the RANGE itself
std::ranges::sort(v);                    // Can't mismatch. Period.
```

Iterator pairs are verbose, error-prone, and don't compose. Ranges fix all three.

---

## Core Concepts

### What Is a Range?

A **range** is anything with `begin()` and `end()`. The `<ranges>` header defines a hierarchy of concepts:

| Concept | Meaning | Example |
|---|---|---|
| `std::ranges::range` | Has `begin()` + `end()` | Any STL container |
| `std::ranges::input_range` | Single-pass readable | `istream_view` |
| `std::ranges::forward_range` | Multi-pass readable | `forward_list` |
| `std::ranges::bidirectional_range` | Can go backward | `list`, `set` |
| `std::ranges::random_access_range` | O(1) index | `vector`, `deque` |
| `std::ranges::contiguous_range` | Elements contiguous in memory | `vector`, `array`, `span` |
| `std::ranges::sized_range` | O(1) `size()` | `vector`, `array` |
| `std::ranges::view` | Cheap to copy/move, O(1) | All view adaptors |
| `std::ranges::borrowed_range` | Safe to return iterators from temporaries | `span`, `string_view` |

Each concept refines the previous — `contiguous_range` ⊂ `random_access_range` ⊂ `bidirectional_range` ⊂ `forward_range` ⊂ `input_range` ⊂ `range`.

### Views vs Containers

| Property | Container (`vector`) | View (`filter_view`) |
|---|---|---|
| Owns elements | Yes | No (references source) |
| Construction | O(N) — allocates + copies | O(1) — stores reference + predicate |
| Evaluation | Eager | **Lazy** — computed on iteration |
| Copyable | Yes (deep copy) | Yes (shallow, O(1)) |
| Modifiable | Yes | Usually read-only |

**Key insight**: Views don't DO anything when you create them. They describe a computation. Work only happens when you iterate.

```cpp
// This does NOTHING — no filtering happens yet
auto expensive = data | views::filter(costly_predicate);

// Work happens HERE, one element at a time
for (auto& x : expensive) { ... }
```

---

## Pipe Syntax (`|`)

The pipe operator composes views left-to-right, like Unix pipes:

```cpp
// Read as: "take data, filter evens, square them, take first 5"
auto result = data
    | views::filter([](int x) { return x % 2 == 0; })
    | views::transform([](int x) { return x * x; })
    | views::take(5);
```

Without pipes (equivalent but harder to read):

```cpp
auto result = views::take(
    views::transform(
        views::filter(data, pred),
        square),
    5);
```

Pipes compose — you can store partial pipelines:

```cpp
auto clean_and_scale = views::filter(is_valid) | views::transform(scale);
auto result = sensor_data | clean_and_scale | views::take(100);
```

---

## Range Adaptors (Views)

### Filtering & Selecting

| Adaptor | What It Does |
|---|---|
| `views::filter(pred)` | Keep elements where `pred` is true |
| `views::take(N)` | First N elements |
| `views::drop(N)` | Skip first N elements |
| `views::take_while(pred)` | Take while predicate holds |
| `views::drop_while(pred)` | Drop while predicate holds |

### Transforming

| Adaptor | What It Does |
|---|---|
| `views::transform(fn)` | Apply `fn` to each element |
| `views::reverse` | Reverse order (needs bidirectional) |
| `views::elements<I>` | Extract Ith element from tuple-like |
| `views::keys` | Equivalent to `elements<0>` |
| `views::values` | Equivalent to `elements<1>` |

### Splitting & Joining

| Adaptor | What It Does |
|---|---|
| `views::split(delim)` | Split into subranges |
| `views::join` | Flatten nested ranges |
| `views::join_with(delim)` | Flatten with delimiter (C++23) |

### Identity & Casting

| Adaptor | What It Does |
|---|---|
| `views::all` | Wrap any range into a view |
| `views::common` | Force matching begin/end types |
| `views::counted(it, n)` | View of N elements from iterator |

---

## Range Factories

```cpp
// Infinite sequence: 0, 1, 2, 3, ...
auto naturals = views::iota(0);

// Bounded: 1, 2, ..., 100
auto one_to_hundred = views::iota(1, 101);

// Single element
auto just_42 = views::single(42);

// Empty
auto nothing = views::empty<int>;

// C++23: repeat N times
// auto fives = views::repeat(5, 10);  // ten 5s
```

---

## Range Algorithms

Range algorithms live in `std::ranges::` and accept ranges directly:

```cpp
std::vector<int> v = {3, 1, 4, 1, 5};

// Classic
std::sort(v.begin(), v.end());

// Ranges — cleaner, safer
std::ranges::sort(v);
```

### Key Algorithms with Projections

```cpp
struct Waypoint { double x, y; int priority; };
std::vector<Waypoint> wps = { ... };

// Sort by priority — no lambda needed
std::ranges::sort(wps, {}, &Waypoint::priority);

// Find max distance from origin
auto farthest = std::ranges::max_element(wps, {},
    [](const Waypoint& w) { return w.x*w.x + w.y*w.y; });

// Partition: high priority first
std::ranges::partition(wps, [](int p) { return p > 5; }, &Waypoint::priority);
```

---

## Projections — The Killer Feature

Projections let you "project" elements before comparison, eliminating verbose lambdas:

```cpp
// WITHOUT projections — compare by member
std::sort(robots.begin(), robots.end(),
    [](const Robot& a, const Robot& b) {
        return a.battery_level < b.battery_level;
    });

// WITH projections — just name the member
std::ranges::sort(robots, std::less{}, &Robot::battery_level);

// Projections work with ANY callable — not just member pointers
std::ranges::sort(robots, {}, [](const Robot& r) {
    return r.x * r.x + r.y * r.y;  // sort by distance from origin
});
```

Projection applies to EACH element independently before comparison. The `{}` means "default comparator" (`std::less{}`).

**Why this matters**: Projections are composable and reusable. You write the "what to compare" once, and the algorithm handles the boilerplate.

---

## Sentinels vs End Iterators

Classic STL requires `begin()` and `end()` to return the **same type**. Ranges relax this — `end()` can return a **sentinel** (a different type):

```cpp
// Sentinel: "stop when you hit null"
struct NullSentinel {
    bool operator==(const char* p) const { return *p == '\0'; }
};

// Now "end" is not a position — it's a CONDITION
// This enables null-terminated string iteration without strlen()
```

Sentinels enable infinite ranges (no end iterator — sentinel says "always continue") and lazy termination conditions.

---

## Writing Custom Views

Use `view_interface` CRTP base to get free member functions:

```cpp
template <std::ranges::input_range R>
class stride_view : public std::ranges::view_interface<stride_view<R>> {
    R base_;
    std::size_t stride_;
public:
    stride_view(R base, std::size_t stride) : base_(std::move(base)), stride_(stride) {}

    auto begin() { /* custom iterator */ }
    auto end()   { /* sentinel or iterator */ }
};
```

`view_interface` gives you `empty()`, `operator bool()`, `front()`, `back()`, `operator[]`, and `size()` — all for free — based on what your `begin()`/`end()` support.

---

## Common Gotchas

### 1. Dangling References

```cpp
auto get_data() { return std::vector{1, 2, 3}; }

// DANGER: view over a temporary
auto v = get_data() | views::filter([](int x) { return x > 1; });
// The vector is DEAD. v holds a dangling reference.

// Safe: ranges::find on a temporary returns ranges::dangling
auto it = std::ranges::find(get_data(), 42);
// it is std::ranges::dangling — won't compile if you dereference it
```

### 2. Single-Pass Views

Some views (like `istream_view`) are single-pass. Iterating twice gives nothing the second time:

```cpp
auto is = std::istringstream{"1 2 3"};
auto v = std::ranges::istream_view<int>(is);
// First loop: 1, 2, 3
// Second loop: EMPTY — stream is consumed
```

### 3. Views Don't Own Data

```cpp
std::vector<int> data = {1, 2, 3};
auto v = data | views::take(2);
data.push_back(4);  // Invalidates iterators → v is now UB
```

### 4. `views::filter` Caches `begin()`

Calling `begin()` on a `filter_view` the first time is O(N) (finds first matching element). Subsequent calls return the cached result. This means `filter_view` is **not const-iterable** after first use.

---

## Performance

Views are **zero-overhead abstractions**. The compiler inlines everything.

Godbolt proof: a pipeline like `v | filter(pred) | transform(fn) | take(5)` compiles to the SAME assembly as a hand-written loop. The view objects are optimized away entirely.

**Why**: Views store references + function objects. Iteration is just pointer chasing + function calls, all inlineable. No heap allocation. No virtual dispatch.

Micro-benchmark pattern:
```
Eager (intermediate vectors): ~3.2ms for 1M elements
Lazy (view pipeline):         ~1.1ms for 1M elements  (no alloc overhead)
```

The lazy version wins because it avoids allocating intermediate `vector`s.

---

## ROS / Robotics Tie-In

### Filtering Sensor Data Lazily

```cpp
// Filter valid laser scan ranges without copying
auto valid_ranges = scan.ranges
    | views::filter([](float r) { return r > 0.01f && r < 30.0f; })
    | views::transform([](float r) { return r * 1000.0f; });  // m → mm
// No allocation — processes lazily as you iterate
```

### Point Cloud Processing

```cpp
// Extract Z-values > ground plane from point cloud
auto above_ground = points
    | views::filter([](const Point& p) { return p.z > 0.05; })
    | views::transform(&Point::z);
```

### Sensorbar Data

```cpp
// Process only reliable sensorbar frames
auto reliable_frames = all_frames
    | views::filter(&SensorbarFrame::is_reliable)
    | views::transform([](const auto& f) { return f.reflectance; });
```

---

## Comparison: Ranges vs Boost.Range vs ranges-v3

| Feature | C++20 Ranges | ranges-v3 | Boost.Range |
|---|---|---|---|
| Standard | Yes | No (library) | No (library) |
| Views | Yes | Yes (more) | Limited |
| Pipe syntax | Yes | Yes | No |
| Projections | Yes | Yes | No |
| `ranges::to` | C++23 | Yes | No |
| Actions (eager) | No | Yes | No |
| Compiler support | GCC 10+, Clang 14+ | GCC 9+, Clang 6+ | GCC 5+, Clang 3+ |

`ranges-v3` by Eric Niebler was the prototype for C++20 Ranges. C++20 adopted a subset. ranges-v3 still has more features (actions, `views::zip` pre-C++23, etc.).

---

## Quick Reference

```cpp
#include <ranges>
#include <algorithm>
namespace views = std::views;       // shorthand
namespace ranges = std::ranges;     // shorthand

// Compose views with |
auto pipeline = data | views::filter(pred) | views::transform(fn);

// Range algorithms
ranges::sort(v);
ranges::sort(v, std::greater{});                  // descending
ranges::sort(v, {}, &Struct::member);             // projection
auto it = ranges::find_if(v, pred);
auto [min, max] = ranges::minmax(v);

// Factories
auto seq = views::iota(0, 100);                   // [0, 100)

// Materialize a view into a container
std::vector<int> vec(pipeline.begin(), pipeline.end());
// C++23: auto vec = pipeline | ranges::to<std::vector>();
```
