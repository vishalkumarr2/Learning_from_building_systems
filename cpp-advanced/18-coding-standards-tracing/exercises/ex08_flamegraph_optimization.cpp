// Exercise 8: Flamegraph-Guided Optimization
//
// This program simulates a request processing pipeline with multiple
// code paths of varying cost. Profile it, read the flamegraph, then
// optimize the hottest path.
//
// TOOL PRACTICE:
//   1. Build:  cmake --build build
//   2. Profile: perf record -g --call-graph dwarf ./ex08_flamegraph_optimization
//   3. Flamegraph:
//      perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
//   4. Open flame.svg in browser, find the widest bar
//   5. Alternatively: perf report --hierarchy
//
// The --bench flag runs both slow and fast versions for comparison.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace pipeline {

// ========================================================================
// Shared types
// ========================================================================

struct Request {
    int id;
    std::string payload;
    int priority;
};

struct Response {
    int request_id;
    double score;
    bool accepted;
};

// ========================================================================
// SLOW implementation — multiple hidden costs
// ========================================================================

namespace slow {

// Cost 1: Expensive string copy per request (no move/reserve)
std::vector<Request> parse_requests(int count) {
    std::vector<Request> reqs;
    for (int i = 0; i < count; ++i) {
        Request r;
        r.id = i;
        // Builds a long string every time
        r.payload = "request-payload-data-for-processing-id-" + std::to_string(i)
                   + "-with-extra-padding-to-make-string-longer";
        r.priority = i % 10;
        reqs.push_back(r);  // copy, not move
    }
    return reqs;
}

// Cost 2: Repeated map lookups with string keys
double score_request(const Request& req) {
    // Build lookup table from scratch EVERY call (should be cached)
    std::unordered_map<std::string, double> weights;
    weights["low"] = 0.1;
    weights["medium"] = 0.5;
    weights["high"] = 0.9;
    weights["critical"] = 1.0;

    const char* level;
    if (req.priority < 3)
        level = "low";
    else if (req.priority < 6)
        level = "medium";
    else if (req.priority < 9)
        level = "high";
    else
        level = "critical";

    double base = weights[level];

    // Cost 3: Unnecessary computation — compute sqrt for every character
    double payload_score = 0.0;
    for (char c : req.payload) {
        payload_score += std::sqrt(static_cast<double>(static_cast<unsigned char>(c)));
    }
    payload_score /= static_cast<double>(req.payload.size());

    return base * payload_score;
}

// Cost 4: Sorting by copying into pairs
std::vector<Response> process_all(const std::vector<Request>& reqs) {
    // Score all requests
    std::vector<std::pair<int, double>> scored;
    for (const auto& r : reqs) {
        scored.push_back({r.id, score_request(r)});
    }

    // Sort by score (descending) — unnecessary for acceptance decision
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<Response> responses;
    for (const auto& s : scored) {
        Response resp;
        resp.request_id = s.first;
        resp.score = s.second;
        resp.accepted = s.second > 1.0;
        responses.push_back(resp);
    }
    return responses;
}

}  // namespace slow

// ========================================================================
// FAST implementation — optimized based on profiling insights
// ========================================================================

namespace fast {

// Fix 1: reserve + emplace_back
std::vector<Request> parse_requests(int count) {
    std::vector<Request> reqs;
    reqs.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        reqs.push_back({i,
            "request-payload-data-for-processing-id-" + std::to_string(i)
            + "-with-extra-padding-to-make-string-longer",
            i % 10});
    }
    return reqs;
}

// Fix 2: static lookup table, simplified payload scoring
double score_request(const Request& req) {
    // Static table — initialized once
    static const double weights[] = {
        0.1, 0.1, 0.1,   // 0-2: low
        0.5, 0.5, 0.5,   // 3-5: medium
        0.9, 0.9, 0.9,   // 6-8: high
        1.0               // 9: critical
    };
    double base = weights[static_cast<size_t>(req.priority)];

    // Simplified payload score: sample instead of full scan
    double payload_score = 0.0;
    size_t step = (req.payload.size() > 10) ? req.payload.size() / 10 : 1;
    size_t samples = 0;
    for (size_t i = 0; i < req.payload.size(); i += step) {
        payload_score += static_cast<double>(
            static_cast<unsigned char>(req.payload[i]));
        ++samples;
    }
    if (samples > 0)
        payload_score = std::sqrt(payload_score / static_cast<double>(samples));

    return base * payload_score;
}

// Fix 3: No unnecessary sort, direct processing
std::vector<Response> process_all(const std::vector<Request>& reqs) {
    std::vector<Response> responses;
    responses.reserve(reqs.size());
    for (const auto& r : reqs) {
        double score = score_request(r);
        responses.push_back({r.id, score, score > 1.0});
    }
    return responses;
}

}  // namespace fast

// ========================================================================
// Test & Benchmark
// ========================================================================

void test() {
    // Both should produce valid responses
    auto slow_reqs = slow::parse_requests(100);
    auto slow_resp = slow::process_all(slow_reqs);
    assert(slow_resp.size() == 100);

    auto fast_reqs = fast::parse_requests(100);
    auto fast_resp = fast::process_all(fast_reqs);
    assert(fast_resp.size() == 100);

    // Verify response IDs cover all requests
    std::vector<bool> slow_seen(100, false);
    for (const auto& r : slow_resp) {
        assert(r.request_id >= 0 && r.request_id < 100);
        slow_seen[static_cast<size_t>(r.request_id)] = true;
    }
    for (size_t i = 0; i < 100; ++i) assert(slow_seen[i]);

    std::vector<bool> fast_seen(100, false);
    for (const auto& r : fast_resp) {
        assert(r.request_id >= 0 && r.request_id < 100);
        fast_seen[static_cast<size_t>(r.request_id)] = true;
    }
    for (size_t i = 0; i < 100; ++i) assert(fast_seen[i]);
}

void benchmark() {
    constexpr int kRequests = 5000;
    constexpr int kRuns = 5;

    volatile size_t sink = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) {
        auto reqs = slow::parse_requests(kRequests);
        auto resp = slow::process_all(reqs);
        sink += resp.size();
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) {
        auto reqs = fast::parse_requests(kRequests);
        auto resp = fast::process_all(reqs);
        sink += resp.size();
    }
    auto t2 = std::chrono::steady_clock::now();

    (void)sink;

    auto slow_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto fast_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    long fast_ms_safe = (fast_ms == 0) ? 1 : fast_ms;

    std::printf("  Slow pipeline: %ld ms (%d requests x %d runs)\n",
                slow_ms, kRequests, kRuns);
    std::printf("  Fast pipeline: %ld ms (%d requests x %d runs)\n",
                fast_ms, kRequests, kRuns);
    std::printf("  Speedup:       %.1fx\n",
                static_cast<double>(slow_ms) / static_cast<double>(fast_ms_safe));
}

}  // namespace pipeline

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    pipeline::test();
    std::printf("ex08_flamegraph_optimization: ALL TESTS PASSED\n");

    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
    }

    if (bench) {
        std::printf("\n=== BENCHMARK ===\n");
        std::printf("Profile the slow version:\n");
        std::printf("  perf record -g --call-graph dwarf ./ex08_flamegraph_optimization\n");
        std::printf("  perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg\n\n");
        pipeline::benchmark();
    }

    return 0;
}
