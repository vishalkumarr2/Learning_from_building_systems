# Exercise 4 — Fixed-Point Arithmetic

**Covers:** Lesson 06
**Difficulty:** Hard

---

## Problem 1: Q-Format Conversions

a) Represent 3.14159 in Q16.16 format. Show the hex value.

b) Represent -0.001 in Q16.16. Show two's complement.

c) What is the maximum representable value in Q16.16? What is the resolution (smallest positive value)?

d) What range of values can Q8.24 represent? When would you prefer Q8.24 over Q16.16?

---

## Problem 2: Multiplication Overflow

Two Q16.16 values: $a = 300.0$, $b = 200.0$.

a) What is $a \times b$ mathematically?

b) Show the Q16.16 multiplication using a 64-bit intermediate. Write the C code.

c) What happens if you use a 32-bit intermediate? Show the binary overflow.

---

## Problem 3: Division

Implement Q16.16 division in C without floating point.

```c
// Divide a by b, both in Q16.16
int32_t q16_div(int32_t a, int32_t b) {
    // Your implementation here
}
```

Handle:
- Division by zero
- Overflow when a is large
- Sign handling

---

## Problem 4: Fixed-Point PID

Implement the velocity-form PID in Q16.16:

$$\Delta u = K_p(e[n] - e[n-1]) + K_i \cdot T_s \cdot e[n] + K_d \cdot \frac{e[n] - 2e[n-1] + e[n-2]}{T_s}$$

Parameters (as floats): $K_p = 2.5$, $K_i = 100.0$, $K_d = 0.005$, $T_s = 0.001$ s

a) Convert each parameter to Q16.16. Can all values fit?

b) Which product ($K_i \times T_s$ or $K_d / T_s$) should you precompute? Why?

c) Write the complete C struct and update function.

---

## Problem 5: Overflow Detection

Write a function that performs Q16.16 addition with saturation:

```c
int32_t q16_add_sat(int32_t a, int32_t b) {
    // Add a + b with saturation (no wrapping)
    // Return Q16_MAX on positive overflow
    // Return Q16_MIN on negative overflow
}
```

Then test with:
- a = 30000.0 (Q16.16), b = 30000.0 → should saturate at 32767.99998
- a = -30000.0, b = -30000.0 → should saturate at -32768.0

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1a:**

$3.14159 \times 65536 = 205887.33$ → Rounded to 205887 = `0x000323FF`

Verify: $205887 / 65536 = 3.14158630...\approx 3.14159$ ✓

**Problem 1c:**

Max: $(2^{31} - 1) / 2^{16} = 32767.99998$
Min: $-2^{31} / 2^{16} = -32768.0$
Resolution: $1 / 2^{16} = 0.0000153$

**Problem 2b:**

```c
int32_t q16_mul(int32_t a, int32_t b) {
    int64_t temp = (int64_t)a * (int64_t)b;
    return (int32_t)(temp >> 16);
}
// a = 300 * 65536 = 19660800
// b = 200 * 65536 = 13107200
// temp = 19660800 * 13107200 = 257698037760000 (fits in int64!)
// result = 257698037760000 >> 16 = 3932160000
// As Q16.16: 3932160000 / 65536 = 60000.0 ✓
```

**Problem 4b:**

$K_i \times T_s = 100.0 \times 0.001 = 0.1$ → Fits in Q16.16 ✓
$K_d / T_s = 0.005 / 0.001 = 5.0$ → Fits in Q16.16 ✓

Precompute both to avoid division in the ISR.

**Problem 5:**

```c
#define Q16_MAX  ((int32_t)0x7FFFFFFF)
#define Q16_MIN  ((int32_t)0x80000000)

int32_t q16_add_sat(int32_t a, int32_t b) {
    int32_t result = a + b;
    // Overflow if both same sign and result is different sign
    if (a > 0 && b > 0 && result < 0) return Q16_MAX;
    if (a < 0 && b < 0 && result > 0) return Q16_MIN;
    return result;
}
```

</details>
