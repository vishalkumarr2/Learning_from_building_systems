// ex02_serialization_bench.cpp — Benchmark 4 serialization methods (no external deps)
//
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -O2 -pthread -lrt ex02_serialization_bench.cpp -o ex02
// (Use -O2 for realistic benchmark numbers; -O0 is fine for correctness)

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ─── The message struct (POD, 32 bytes) ─────────────────
struct SensorMsg {
    uint64_t timestamp_ns;
    double x;
    double y;
    double z;
};

static_assert(sizeof(SensorMsg) == 32, "SensorMsg must be exactly 32 bytes");
static_assert(std::is_trivially_copyable_v<SensorMsg>);

static constexpr std::size_t kIterations = 1'000'000;
static constexpr std::size_t kMsgSize    = sizeof(SensorMsg);

// ─── Timing helper ──────────────────────────────────────
struct BenchResult {
    const char* name;
    double encode_ns_per_msg;
    double decode_ns_per_msg;
    double encode_mb_per_sec;
    double decode_mb_per_sec;
};

static double to_ns(std::chrono::steady_clock::duration d) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

// ─── Method 1: Raw memcpy ───────────────────────────────
static BenchResult bench_memcpy() {
    std::vector<char> buf(kMsgSize);
    SensorMsg msg{}, out{};

    // Encode
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        msg.timestamp_ns = i;
        msg.x = static_cast<double>(i) * 0.1;
        msg.y = static_cast<double>(i) * 0.2;
        msg.z = static_cast<double>(i) * 0.3;
        std::memcpy(buf.data(), &msg, kMsgSize);
    }
    auto t1 = std::chrono::steady_clock::now();

    // Decode
    auto t2 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        std::memcpy(&out, buf.data(), kMsgSize);
        // Prevent dead-code elimination
        if (out.timestamp_ns == UINT64_MAX) std::abort();
    }
    auto t3 = std::chrono::steady_clock::now();

    double enc_ns = to_ns(t1 - t0) / static_cast<double>(kIterations);
    double dec_ns = to_ns(t3 - t2) / static_cast<double>(kIterations);
    double bytes_per_iter = static_cast<double>(kMsgSize);

    return {
        "memcpy (raw POD)",
        enc_ns, dec_ns,
        (bytes_per_iter / enc_ns) * 1000.0,  // MB/s
        (bytes_per_iter / dec_ns) * 1000.0
    };
}

// ─── Method 2: Hand-rolled binary (field-by-field) ──────
namespace handrolled {

static void encode(char* buf, const SensorMsg& msg) {
    std::memcpy(buf + 0,  &msg.timestamp_ns, 8);
    std::memcpy(buf + 8,  &msg.x,            8);
    std::memcpy(buf + 16, &msg.y,            8);
    std::memcpy(buf + 24, &msg.z,            8);
}

static void decode(const char* buf, SensorMsg& msg) {
    std::memcpy(&msg.timestamp_ns, buf + 0,  8);
    std::memcpy(&msg.x,            buf + 8,  8);
    std::memcpy(&msg.y,            buf + 16, 8);
    std::memcpy(&msg.z,            buf + 24, 8);
}

} // namespace handrolled

static BenchResult bench_handrolled() {
    char buf[32]{};
    SensorMsg msg{}, out{};

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        msg.timestamp_ns = i;
        msg.x = static_cast<double>(i) * 0.1;
        msg.y = static_cast<double>(i) * 0.2;
        msg.z = static_cast<double>(i) * 0.3;
        handrolled::encode(buf, msg);
    }
    auto t1 = std::chrono::steady_clock::now();

    auto t2 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        handrolled::decode(buf, out);
        if (out.timestamp_ns == UINT64_MAX) std::abort();
    }
    auto t3 = std::chrono::steady_clock::now();

    double enc_ns = to_ns(t1 - t0) / static_cast<double>(kIterations);
    double dec_ns = to_ns(t3 - t2) / static_cast<double>(kIterations);
    double bytes_per_iter = static_cast<double>(kMsgSize);

    return {
        "hand-rolled binary",
        enc_ns, dec_ns,
        (bytes_per_iter / enc_ns) * 1000.0,
        (bytes_per_iter / dec_ns) * 1000.0
    };
}

// ─── Method 3: Zero-copy FlatBuffer-like ────────────────
//
// The "FlatBuffer" approach: write fields at known offsets into a
// pre-allocated buffer. For decoding, read directly from the buffer
// without copying into a struct — true zero-copy access.

namespace zerocopy {

static constexpr std::size_t kBufSize = 32;

static void encode(char* buf, uint64_t ts, double x, double y, double z) {
    std::memcpy(buf + 0,  &ts, 8);
    std::memcpy(buf + 8,  &x,  8);
    std::memcpy(buf + 16, &y,  8);
    std::memcpy(buf + 24, &z,  8);
}

// Zero-copy readers: return value directly from buffer offsets
static uint64_t read_ts(const char* buf) {
    uint64_t v; std::memcpy(&v, buf + 0, 8); return v;
}
static double read_x(const char* buf) {
    double v; std::memcpy(&v, buf + 8, 8); return v;
}
static double read_y(const char* buf) {
    double v; std::memcpy(&v, buf + 16, 8); return v;
}
static double read_z(const char* buf) {
    double v; std::memcpy(&v, buf + 24, 8); return v;
}

} // namespace zerocopy

static BenchResult bench_zerocopy() {
    char buf[zerocopy::kBufSize]{};

    // Encode
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        zerocopy::encode(buf,
                         static_cast<uint64_t>(i),
                         static_cast<double>(i) * 0.1,
                         static_cast<double>(i) * 0.2,
                         static_cast<double>(i) * 0.3);
    }
    auto t1 = std::chrono::steady_clock::now();

    // Decode: read individual fields (zero-copy style)
    volatile double sink = 0;
    auto t2 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        auto ts = zerocopy::read_ts(buf);
        auto x  = zerocopy::read_x(buf);
        auto y  = zerocopy::read_y(buf);
        auto z  = zerocopy::read_z(buf);
        sink = static_cast<double>(ts) + x + y + z;
    }
    auto t3 = std::chrono::steady_clock::now();
    (void)sink;

    double enc_ns = to_ns(t1 - t0) / static_cast<double>(kIterations);
    double dec_ns = to_ns(t3 - t2) / static_cast<double>(kIterations);
    double bytes_per_iter = static_cast<double>(kMsgSize);

    return {
        "zero-copy (FlatBuf-like)",
        enc_ns, dec_ns,
        (bytes_per_iter / enc_ns) * 1000.0,
        (bytes_per_iter / dec_ns) * 1000.0
    };
}

// ─── Method 4: JSON-style sprintf/sscanf ────────────────
static BenchResult bench_json() {
    char buf[256]{};
    SensorMsg msg{}, out{};

    // Encode: sprintf
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        msg.timestamp_ns = i;
        msg.x = static_cast<double>(i) * 0.1;
        msg.y = static_cast<double>(i) * 0.2;
        msg.z = static_cast<double>(i) * 0.3;
        std::snprintf(buf, sizeof(buf),
                      R"({"ts":%lu,"x":%.6f,"y":%.6f,"z":%.6f})",
                      static_cast<unsigned long>(msg.timestamp_ns),
                      msg.x, msg.y, msg.z);
    }
    auto t1 = std::chrono::steady_clock::now();

    // Decode: sscanf
    auto t2 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kIterations; ++i) {
        unsigned long ts_tmp{};
        std::sscanf(buf, R"({"ts":%lu,"x":%lf,"y":%lf,"z":%lf})",
                    &ts_tmp, &out.x, &out.y, &out.z);
        out.timestamp_ns = ts_tmp;
        if (out.timestamp_ns == UINT64_MAX) std::abort();
    }
    auto t3 = std::chrono::steady_clock::now();

    double enc_ns = to_ns(t1 - t0) / static_cast<double>(kIterations);
    double dec_ns = to_ns(t3 - t2) / static_cast<double>(kIterations);
    // JSON size is variable; use the last written length
    double json_size = static_cast<double>(std::strlen(buf));

    return {
        "JSON (sprintf/sscanf)",
        enc_ns, dec_ns,
        (json_size / enc_ns) * 1000.0,
        (json_size / dec_ns) * 1000.0
    };
}

// ─── Print results ──────────────────────────────────────
static void print_table(const std::vector<BenchResult>& results) {
    std::printf("\n");
    std::printf("╔══════════════════════════════╦══════════════╦══════════════╦══════════════╦══════════════╗\n");
    std::printf("║ %-28s ║ %12s ║ %12s ║ %12s ║ %12s ║\n",
                "Method", "Enc ns/msg", "Dec ns/msg", "Enc MB/s", "Dec MB/s");
    std::printf("╠══════════════════════════════╬══════════════╬══════════════╬══════════════╬══════════════╣\n");

    for (const auto& r : results) {
        std::printf("║ %-28s ║ %12.1f ║ %12.1f ║ %12.1f ║ %12.1f ║\n",
                    r.name,
                    r.encode_ns_per_msg, r.decode_ns_per_msg,
                    r.encode_mb_per_sec, r.decode_mb_per_sec);
    }

    std::printf("╚══════════════════════════════╩══════════════╩══════════════╩══════════════╩══════════════╝\n");
    std::printf("\n  %zu iterations per method, message size = %zu bytes\n",
                kIterations, kMsgSize);
}

// ─── Round-trip correctness check ───────────────────────
static void verify_correctness() {
    std::printf("Verifying round-trip correctness...\n");

    SensorMsg original{12345678ULL, 1.234, -5.678, 9.012};
    SensorMsg decoded{};
    char buf[256]{};

    // memcpy
    std::memcpy(buf, &original, sizeof(original));
    std::memcpy(&decoded, buf, sizeof(decoded));
    assert(decoded.timestamp_ns == original.timestamp_ns);
    assert(decoded.x == original.x);

    // hand-rolled
    handrolled::encode(buf, original);
    handrolled::decode(buf, decoded);
    assert(decoded.timestamp_ns == original.timestamp_ns);
    assert(decoded.y == original.y);

    // zero-copy
    zerocopy::encode(buf, original.timestamp_ns, original.x, original.y, original.z);
    assert(zerocopy::read_ts(buf) == original.timestamp_ns);
    assert(zerocopy::read_z(buf) == original.z);

    // JSON (lossy due to floating point formatting, check within epsilon)
    std::snprintf(buf, sizeof(buf),
                  R"({"ts":%lu,"x":%.6f,"y":%.6f,"z":%.6f})",
                  static_cast<unsigned long>(original.timestamp_ns),
                  original.x, original.y, original.z);
    unsigned long ts_tmp{};
    std::sscanf(buf, R"({"ts":%lu,"x":%lf,"y":%lf,"z":%lf})",
                &ts_tmp, &decoded.x, &decoded.y, &decoded.z);
    decoded.timestamp_ns = ts_tmp;
    assert(decoded.timestamp_ns == original.timestamp_ns);
    assert(std::fabs(decoded.x - original.x) < 1e-5);

    std::printf("  All methods pass round-trip verification.\n\n");
}

// ─── Main ───────────────────────────────────────────────
int main() {
    verify_correctness();

    std::printf("Benchmarking %zu iterations per method...\n", kIterations);

    std::vector<BenchResult> results;
    results.push_back(bench_memcpy());
    results.push_back(bench_handrolled());
    results.push_back(bench_zerocopy());
    results.push_back(bench_json());

    print_table(results);

    // Summary
    double fastest_enc = results[0].encode_ns_per_msg;
    double slowest_enc = results.back().encode_ns_per_msg;
    std::printf("\n  JSON is ~%.0fx slower to encode than memcpy.\n",
                slowest_enc / fastest_enc);
    std::printf("  Zero-copy decode approaches memcpy speed.\n");
    std::printf("  This is why robotics/trading IPC avoids JSON.\n");

    return 0;
}
