# Module 13 — C++20 `std::format`

> **Compiler Requirements**: GCC 13+ or Clang 17+ (with `-std=c++20`).
> GCC 9–12 do **not** ship `<format>`. For older compilers, use the `{fmt}` library as a drop-in polyfill.

---

## 1. Motivation

C has three string formatting approaches, each with serious trade-offs:

| Approach | Type-safe | Fast | Extensible | Positional args |
|---|---|---|---|---|
| `printf` | ❌ | ✅ | ❌ | ❌ (POSIX only) |
| `std::ostream` | ✅ | ❌ | ✅ | ❌ |
| `std::format` | ✅ | ✅ | ✅ | ✅ |

**`printf` problems:**
- `%d` with a `std::string` → undefined behaviour (no compiler error by default)
- Format string attacks: user-supplied format strings → stack reads / writes
- No custom type support (can't `printf` a `Vector3d`)

**`ostream` problems:**
- Stateful manipulators (`std::hex` sticks until reset — classic bug)
- Slow: virtual dispatch, locale facets, sentry overhead
- Verbose: `std::setw(8) << std::setfill('0') << std::hex << value`

**`std::format` fixes all of these:**
- Type-safe: mismatched arguments → compile-time error
- Fast: competitive with `snprintf`, 3–10× faster than `ostringstream`
- Extensible: specialise `std::formatter<T>` for any type
- Python-like syntax: `"{:08x}"` instead of `std::setw(8) << std::setfill('0') << std::hex`

---

## 2. Basic Format Strings

```cpp
#include <format>
#include <string>

std::string s1 = std::format("Hello, {}!", "world");          // "Hello, world!"
std::string s2 = std::format("{} + {} = {}", 1, 2, 3);        // "1 + 2 = 3"
std::string s3 = std::format("{1} before {0}", "B", "A");     // "A before B"
```

### Replacement field syntax

```
{[arg_id][:format_spec]}
```

- `{}` — next auto-numbered argument
- `{0}` — explicit argument index (0-based)
- `{:d}` — format as decimal integer
- `{:.3f}` — 3 decimal places, fixed notation
- `{:>20}` — right-align in 20-char field
- `{:#x}` — hex with `0x` prefix
- `{:b}` — binary representation

---

## 3. Format Specification Mini-Language

```
[[fill]align][sign][#][0][width][.precision][L][type]
```

### Fill and Align

| Char | Meaning |
|---|---|
| `<` | Left-align (default for non-numeric) |
| `>` | Right-align (default for numeric) |
| `^` | Centre-align |

```cpp
std::format("{:*>10}", 42);    // "********42"
std::format("{:*<10}", 42);    // "42********"
std::format("{:*^10}", 42);    // "****42****"
```

### Sign

| Char | Meaning |
|---|---|
| `+` | Always show sign |
| `-` | Show sign only for negatives (default) |
| ` ` | Space for positives, minus for negatives |

### `#` — Alternate Form

```cpp
std::format("{:#x}", 255);     // "0xff"
std::format("{:#b}", 10);      // "0b1010"
std::format("{:#o}", 8);       // "010"
```

### `0` — Zero-Padding

```cpp
std::format("{:08x}", 0xDEAD); // "0000dead"
```

### Width and Precision

```cpp
std::format("{:10d}", 42);     // "        42"
std::format("{:.5f}", 3.14);   // "3.14000"
std::format("{:10.3f}", 3.14); // "     3.140"
```

### Type Specifiers

**Integer types:** `b` `B` `c` `d` `o` `x` `X`
**Float types:** `a` `A` `e` `E` `f` `F` `g` `G`
**String/char:** `s` (default for strings)
**Bool:** `s` (outputs "true"/"false")

```cpp
std::format("{:d}", true);     // "1"
std::format("{:s}", true);     // "true" (C++23 guarantees this)
std::format("{:c}", 65);       // "A"
```

---

## 4. The `std::format` Family

### `std::format` — returns `std::string`
```cpp
std::string msg = std::format("x={}, y={}", x, y);
```

### `std::format_to` — writes to an output iterator
```cpp
std::string buf;
std::format_to(std::back_inserter(buf), "val={}", 42);
// Avoids extra allocation if buf already has capacity
```

### `std::format_to_n` — writes at most N characters
```cpp
char buf[64];
auto result = std::format_to_n(buf, sizeof(buf) - 1, "sensor={:.2f}", 3.14159);
*result.out = '\0';  // null-terminate
// result.size tells you how many chars would have been written
```

### `std::formatted_size` — compute output size without writing
```cpp
auto n = std::formatted_size("val={}", 42);  // returns 6
```

**Robotics use case**: `format_to_n` is ideal for fixed-size embedded buffers (CAN message descriptions, LCD output, UART debug strings).

---

## 5. Custom Formatters

Specialise `std::formatter<T>` with two methods:

```cpp
template<>
struct std::formatter<MyType> {
    // Parse the format spec (part after ':')
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();  // no custom specs
    }

    // Format the value
    auto format(const MyType& val, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "...", val.field);
    }
};
```

### Example: Vector3d

```cpp
struct Vector3d { double x, y, z; };

template<>
struct std::formatter<Vector3d> {
    char mode = 'f';  // 'f' = full, 's' = short

    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && (*it == 'f' || *it == 's')) {
            mode = *it++;
        }
        return it;
    }

    auto format(const Vector3d& v, std::format_context& ctx) const {
        if (mode == 's')
            return std::format_to(ctx.out(), "({:.1f},{:.1f},{:.1f})", v.x, v.y, v.z);
        return std::format_to(ctx.out(), "Vector3d(x={:.4f}, y={:.4f}, z={:.4f})",
                              v.x, v.y, v.z);
    }
};
```

Usage: `std::format("pos={:s}", pos)` → `"pos=(1.2,3.4,5.6)"`

---

## 6. Compile-Time Format String Checking

One of the biggest wins: `std::format` validates the format string **at compile time**.

```cpp
// COMPILE ERROR — too few arguments:
std::format("{} {} {}", 1, 2);

// COMPILE ERROR — bad format spec for string:
std::format("{:d}", "hello");
```

This is implemented via `std::basic_format_string` (a consteval-checked wrapper). The format string must be a compile-time constant — passing a runtime `std::string` as the format string won't compile (by design, to prevent format string attacks).

```cpp
// OK:
std::format("{}", 42);

// ERROR — runtime format strings not allowed by default:
std::string fmt_str = "{}";
std::format(fmt_str, 42);  // won't compile

// Use std::vformat for runtime format strings (opt-in, unsafe):
std::vformat(fmt_str, std::make_format_args(42));
```

---

## 7. `std::print` / `std::println` (C++23)

C++23 adds direct-to-stdout formatted output:

```cpp
#include <print>
std::print("x={}, y={}\n", x, y);   // no \n auto-added
std::println("x={}, y={}", x, y);   // adds \n
```

These bypass `std::cout` and write directly to the file descriptor — faster and avoids sync issues. GCC 14+ and Clang 18+ support these.

---

## 8. Performance

Typical benchmark results (100K iterations, formatting a mix of ints/floats/strings):

| Method | Time (ms) | Relative |
|---|---|---|
| `snprintf` | ~12 | 1.0× |
| `std::format` | ~14 | 1.2× |
| `fmt::format` | ~13 | 1.1× |
| `std::ostringstream` | ~45 | 3.8× |
| `operator+` concat | ~35 | 2.9× |

Key takeaways:
- `std::format` is within 20% of `snprintf`
- `std::format` is 3–4× faster than `ostringstream`
- `std::format_to` with a pre-allocated buffer approaches `snprintf` speed
- The `{fmt}` library (`fmt::format`) is essentially identical — `std::format` was adopted from it

---

## 9. Locale Support and the `L` Flag

By default, `std::format` does **not** use locale-specific formatting (unlike `printf` with `'` flag or `ostream` with `imbue`). This is intentional — locale-free by default is faster and more predictable.

Use the `L` flag to enable locale-dependent formatting:

```cpp
// Default: no thousand separators
std::format("{}", 1000000);       // "1000000"

// With locale:
std::format("{:L}", 1000000);     // "1,000,000" (en_US locale)
```

For robotics: almost always use the default (locale-free). You don't want sensor readings changing format based on the operator's locale.

---

## 10. Using `{fmt}` as a Polyfill

The [`{fmt}` library](https://github.com/fmtlib/fmt) is the reference implementation that `std::format` was adopted from. Use it when:
- Your compiler doesn't have `<format>` (GCC < 13, Clang < 17)
- You want `fmt::print` before C++23 `std::print` is available

### Installation

```bash
# Ubuntu/Debian
sudo apt install libfmt-dev

# Or via CMake FetchContent
```

### CMake integration

```cmake
find_package(fmt REQUIRED)
target_link_libraries(myapp PRIVATE fmt::fmt)
```

### Code migration path

```cpp
// {fmt} library:
#include <fmt/format.h>
std::string s = fmt::format("x={}", 42);

// Standard C++20 (GCC 13+):
#include <format>
std::string s = std::format("x={}", 42);
```

The APIs are nearly identical. Migration is mostly `s/fmt::format/std::format/g` and `s/<fmt\/format.h>/<format>/g`.

---

## 11. Safety Properties

| Vulnerability | `printf` | `std::format` |
|---|---|---|
| Buffer overflow | ✅ (`sprintf`) | ❌ (returns `std::string`) |
| Type mismatch UB | ✅ (`%d` + string) | ❌ (compile-time check) |
| Format string attack | ✅ (user-supplied `%n`) | ❌ (format string must be constexpr) |
| Missing arguments | ✅ (UB) | ❌ (compile-time check) |

---

## 12. ROS / Robotics Patterns

### Formatting sensor readings
```cpp
std::string msg = std::format(
    "[{}] imu: roll={:+.3f} pitch={:+.3f} yaw={:+.3f}",
    timestamp, roll, pitch, yaw);
```

### Hex dump for CAN frames
```cpp
for (auto byte : can_frame.data)
    std::format_to(std::back_inserter(out), "{:02x} ", byte);
```

### Diagnostic tables (aligned columns)
```cpp
std::format("{:<20} {:>10} {:>10}", "Motor ID", "Current", "Temp");
std::format("{:<20} {:>10.2f} {:>10.1f}", "left_wheel", 2.34, 45.2);
```

### Replacing `ROS_INFO` / `RCLCPP_INFO`
```cpp
// Old (printf-style, unsafe):
ROS_INFO("Pose: x=%.3f y=%.3f theta=%.3f", x, y, theta);

// New (type-safe):
RCLCPP_INFO(get_logger(), "%s",
    std::format("Pose: x={:.3f} y={:.3f} theta={:.3f}", x, y, theta).c_str());

// Or with a thin wrapper (see ex04_safe_logging.cpp)
```

---

## 13. Common Patterns

### Timestamps (ISO 8601)
```cpp
std::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z",
            year, month, day, hour, min, sec, ms);
```

### Hex dump (xxd-style)
```cpp
for (size_t i = 0; i < data.size(); i += 16) {
    std::format_to(out, "{:08x}: ", i);
    for (size_t j = 0; j < 16 && i + j < data.size(); ++j)
        std::format_to(out, "{:02x} ", data[i + j]);
}
```

### Aligned table output
```cpp
std::format("{:<15} {:>8} {:>8} {:>8}", "Sensor", "Min", "Max", "Mean");
```

### Zero-padded frame IDs
```cpp
std::format("frame_{:06d}.png", frame_number);
```

---

## References

- [cppreference: std::format](https://en.cppreference.com/w/cpp/utility/format/format)
- [P0645R10 — Text Formatting](https://wg21.link/P0645)
- [{fmt} library](https://github.com/fmtlib/fmt)
- [Victor Zverovich — std::format in C++20](https://fmt.dev/latest/index.html)
