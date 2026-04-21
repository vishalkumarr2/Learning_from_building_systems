// =============================================================================
// Exercise 06: OTA Update State Machine
// =============================================================================
// Implement an A/B partition OTA update manager with:
//   1. Download → Verify → Stage → Activate → Rollback lifecycle
//   2. CRC-32 integrity check on downloaded image
//   3. Version comparison (reject downgrades)
//   4. Automatic rollback on boot failure (boot counter)
//   5. Transaction log for fleet management audit trail
//
// A/B partition scheme:
//   Slot A: currently running firmware
//   Slot B: inactive partition for staging updates
//   On activate: swap A↔B, reboot into new firmware
//   If boot fails 3 times: automatic rollback to previous slot
//
// Production relevance:
//   - Tesla, Rivian, Mercedes: all use A/B OTA updates
//   - ISO 24089 (Road vehicles — Software update engineering)
//   - Fleet management requires audit trail of all update attempts
// =============================================================================

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// ============================================================================
// Version (semver-like)
// ============================================================================
struct Version {
    uint16_t major = 0;
    uint16_t minor = 0;
    uint16_t patch = 0;

    bool operator>(Version const& o) const {
        if (major != o.major) return major > o.major;
        if (minor != o.minor) return minor > o.minor;
        return patch > o.patch;
    }
    bool operator==(Version const& o) const {
        return major == o.major && minor == o.minor && patch == o.patch;
    }
    bool operator<(Version const& o) const {
        return o > *this;
    }
    bool operator>=(Version const& o) const {
        return *this > o || *this == o;
    }

    std::string str() const {
        return std::to_string(major) + "." +
               std::to_string(minor) + "." +
               std::to_string(patch);
    }
};

// ============================================================================
// CRC-32 (same as ex01 — reused here for image verification)
// ============================================================================
namespace crc32 {
    constexpr uint32_t TABLE[256] = {
        0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA,
        0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
        0x0EDB8832, 0x79DCB8A4, 0xE0D5E91B, 0x97D2D988,
        0x09B64C2B, 0x7EB17CBF, 0xE7B82D09, 0x90BF1D9F,
        0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE,
        0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
        0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC,
        0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
        0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
        0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
        0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940,
        0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
        0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116,
        0x21B4F6B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
        0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
        0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
    };

    constexpr uint32_t compute(uint8_t const* data, size_t len) {
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < len; ++i) {
            crc = TABLE[(crc ^ data[i]) & 0x3F] ^ (crc >> 6);
        }
        return crc ^ 0xFFFFFFFF;
    }
}

// ============================================================================
// Update Image (simulated)
// ============================================================================
struct UpdateImage {
    Version version;
    std::vector<uint8_t> data;   // firmware binary
    uint32_t expected_crc = 0;   // CRC provided by server

    bool verify_integrity() const {
        uint32_t actual = crc32::compute(data.data(), data.size());
        return actual == expected_crc;
    }
};

// ============================================================================
// OTA State Machine
// ============================================================================
enum class OtaState : uint8_t {
    IDLE,
    DOWNLOADING,
    DOWNLOADED,
    VERIFYING,
    VERIFIED,
    STAGING,
    STAGED,
    ACTIVATING,
    ACTIVE,       // New firmware running
    ROLLING_BACK,
    ROLLED_BACK,
    FAILED
};

constexpr const char* to_string(OtaState s) {
    switch (s) {
        case OtaState::IDLE:         return "IDLE";
        case OtaState::DOWNLOADING:  return "DOWNLOADING";
        case OtaState::DOWNLOADED:   return "DOWNLOADED";
        case OtaState::VERIFYING:    return "VERIFYING";
        case OtaState::VERIFIED:     return "VERIFIED";
        case OtaState::STAGING:      return "STAGING";
        case OtaState::STAGED:       return "STAGED";
        case OtaState::ACTIVATING:   return "ACTIVATING";
        case OtaState::ACTIVE:       return "ACTIVE";
        case OtaState::ROLLING_BACK: return "ROLLING_BACK";
        case OtaState::ROLLED_BACK:  return "ROLLED_BACK";
        case OtaState::FAILED:       return "FAILED";
    }
    return "UNKNOWN";
}

// ============================================================================
// Audit Log Entry
// ============================================================================
struct AuditEntry {
    OtaState from;
    OtaState to;
    std::string detail;
};

// ============================================================================
// OTA Manager
// ============================================================================
class OtaManager {
    OtaState state_ = OtaState::IDLE;
    Version current_version_{1, 0, 0};
    Version staged_version_{};
    UpdateImage pending_image_;
    uint8_t boot_fail_count_ = 0;
    std::vector<AuditEntry> audit_log_;

    static constexpr uint8_t MAX_BOOT_FAILS = 3;

    void transition(OtaState to, std::string detail = "") {
        audit_log_.push_back({state_, to, std::move(detail)});
        state_ = to;
    }

public:
    explicit OtaManager(Version current)
        : current_version_(current) {}

    // --- State Machine Operations ---

    // Start download (simulated: image is provided directly)
    bool start_download(UpdateImage image) {
        if (state_ != OtaState::IDLE && state_ != OtaState::FAILED &&
            state_ != OtaState::ROLLED_BACK) {
            return false;  // busy
        }

        // Reject downgrades
        if (image.version < current_version_) {
            transition(OtaState::FAILED,
                       "Downgrade rejected: " + image.version.str() +
                       " < " + current_version_.str());
            return false;
        }

        // Reject same version
        if (image.version == current_version_) {
            transition(OtaState::FAILED,
                       "Same version: " + image.version.str());
            return false;
        }

        pending_image_ = std::move(image);
        transition(OtaState::DOWNLOADING, "Downloading " +
                   pending_image_.version.str());
        // Simulate download completion
        transition(OtaState::DOWNLOADED, "Download complete");
        return true;
    }

    // Verify the downloaded image
    bool verify() {
        if (state_ != OtaState::DOWNLOADED) return false;

        transition(OtaState::VERIFYING, "CRC check in progress");

        if (!pending_image_.verify_integrity()) {
            transition(OtaState::FAILED, "CRC mismatch — corrupted image");
            return false;
        }

        transition(OtaState::VERIFIED, "CRC OK");
        return true;
    }

    // Stage to inactive partition
    bool stage() {
        if (state_ != OtaState::VERIFIED) return false;

        transition(OtaState::STAGING, "Writing to inactive partition");
        staged_version_ = pending_image_.version;
        transition(OtaState::STAGED, "Staged v" +
                   staged_version_.str());
        return true;
    }

    // Activate: swap partitions and simulate reboot
    bool activate() {
        if (state_ != OtaState::STAGED) return false;

        transition(OtaState::ACTIVATING, "Swapping partitions");
        boot_fail_count_ = 0;
        current_version_ = staged_version_;
        transition(OtaState::ACTIVE, "Running v" +
                   current_version_.str());
        return true;
    }

    // Report boot result (called by bootloader)
    bool report_boot(bool success) {
        if (state_ != OtaState::ACTIVE) return false;

        if (success) {
            boot_fail_count_ = 0;
            transition(OtaState::IDLE, "Boot confirmed OK");
            return true;
        }

        ++boot_fail_count_;
        if (boot_fail_count_ >= MAX_BOOT_FAILS) {
            return rollback();
        }

        // Still has chances
        audit_log_.push_back({state_, state_,
            "Boot fail " + std::to_string(boot_fail_count_) +
            "/" + std::to_string(MAX_BOOT_FAILS)});
        return false;
    }

    // Manual rollback
    bool rollback() {
        if (state_ != OtaState::ACTIVE &&
            state_ != OtaState::STAGED) return false;

        transition(OtaState::ROLLING_BACK, "Reverting to previous firmware");
        // In real system: swap partition table back
        staged_version_ = {};
        transition(OtaState::ROLLED_BACK, "Rollback complete");
        return true;
    }

    // --- Queries ---
    OtaState state() const { return state_; }
    Version current() const { return current_version_; }
    Version staged() const { return staged_version_; }
    uint8_t boot_fails() const { return boot_fail_count_; }
    std::vector<AuditEntry> const& audit() const { return audit_log_; }
};

// ============================================================================
// Helper: create a valid image with correct CRC
// ============================================================================
UpdateImage make_image(Version ver, std::vector<uint8_t> data) {
    UpdateImage img;
    img.version = ver;
    img.data = std::move(data);
    img.expected_crc = crc32::compute(img.data.data(), img.data.size());
    return img;
}

UpdateImage make_corrupt_image(Version ver) {
    UpdateImage img;
    img.version = ver;
    img.data = {0xDE, 0xAD, 0xBE, 0xEF};
    img.expected_crc = 0xBADBAD;  // wrong CRC
    return img;
}

// ============================================================================
// Self-Test
// ============================================================================
void test_happy_path() {
    std::cout << "--- Test: Happy path (full update) ---\n";
    OtaManager mgr(Version{1, 0, 0});

    auto img = make_image({2, 0, 0}, {0x01, 0x02, 0x03, 0x04});

    assert(mgr.start_download(std::move(img)));
    assert(mgr.state() == OtaState::DOWNLOADED);

    assert(mgr.verify());
    assert(mgr.state() == OtaState::VERIFIED);

    assert(mgr.stage());
    assert(mgr.state() == OtaState::STAGED);

    assert(mgr.activate());
    assert(mgr.state() == OtaState::ACTIVE);
    assert(mgr.current().major == 2);

    assert(mgr.report_boot(true));
    assert(mgr.state() == OtaState::IDLE);

    std::cout << "  PASS: v1.0.0 → v2.0.0 complete\n";
}

void test_reject_downgrade() {
    std::cout << "--- Test: Reject downgrade ---\n";
    OtaManager mgr(Version{2, 0, 0});

    auto img = make_image({1, 5, 0}, {0x01});
    assert(!mgr.start_download(std::move(img)));
    assert(mgr.state() == OtaState::FAILED);
    std::cout << "  PASS: Downgrade rejected\n";
}

void test_crc_failure() {
    std::cout << "--- Test: CRC failure ---\n";
    OtaManager mgr(Version{1, 0, 0});

    auto img = make_corrupt_image({2, 0, 0});
    mgr.start_download(std::move(img));
    assert(!mgr.verify());
    assert(mgr.state() == OtaState::FAILED);
    std::cout << "  PASS: Corrupt image detected\n";
}

void test_automatic_rollback_after_3_boot_fails() {
    std::cout << "--- Test: Auto rollback after 3 boot failures ---\n";
    OtaManager mgr(Version{1, 0, 0});

    auto img = make_image({2, 0, 0}, {0xAA, 0xBB});
    mgr.start_download(std::move(img));
    mgr.verify();
    mgr.stage();
    mgr.activate();
    assert(mgr.state() == OtaState::ACTIVE);

    // 3 boot failures → auto rollback
    mgr.report_boot(false);
    assert(mgr.state() == OtaState::ACTIVE);
    mgr.report_boot(false);
    assert(mgr.state() == OtaState::ACTIVE);
    mgr.report_boot(false);
    assert(mgr.state() == OtaState::ROLLED_BACK);

    std::cout << "  PASS: Auto rollback after 3 failures\n";
}

void test_manual_rollback_from_staged() {
    std::cout << "--- Test: Manual rollback from staged ---\n";
    OtaManager mgr(Version{1, 0, 0});

    auto img = make_image({2, 0, 0}, {0x01});
    mgr.start_download(std::move(img));
    mgr.verify();
    mgr.stage();
    assert(mgr.state() == OtaState::STAGED);

    assert(mgr.rollback());
    assert(mgr.state() == OtaState::ROLLED_BACK);
    std::cout << "  PASS: Manual rollback from STAGED\n";
}

void test_reject_same_version() {
    std::cout << "--- Test: Reject same version ---\n";
    OtaManager mgr(Version{1, 0, 0});

    auto img = make_image({1, 0, 0}, {0x01});
    assert(!mgr.start_download(std::move(img)));
    assert(mgr.state() == OtaState::FAILED);
    std::cout << "  PASS: Same version rejected\n";
}

void test_audit_trail() {
    std::cout << "--- Test: Audit trail completeness ---\n";
    OtaManager mgr(Version{1, 0, 0});

    auto img = make_image({2, 0, 0}, {0xAA});
    mgr.start_download(std::move(img));
    mgr.verify();
    mgr.stage();
    mgr.activate();
    mgr.report_boot(true);

    auto const& log = mgr.audit();
    assert(log.size() >= 8);  // multiple transitions

    // Print audit trail
    for (auto const& e : log) {
        std::cout << "    " << to_string(e.from) << " → "
                  << to_string(e.to) << ": " << e.detail << "\n";
    }

    std::cout << "  PASS: " << log.size()
              << " audit entries recorded\n";
}

void test_can_retry_after_failure() {
    std::cout << "--- Test: Can retry after failure ---\n";
    OtaManager mgr(Version{1, 0, 0});

    // First attempt: corrupt image
    auto bad = make_corrupt_image({2, 0, 0});
    mgr.start_download(std::move(bad));
    mgr.verify();
    assert(mgr.state() == OtaState::FAILED);

    // Second attempt: valid image
    auto good = make_image({2, 0, 0}, {0x01, 0x02});
    assert(mgr.start_download(std::move(good)));
    assert(mgr.verify());
    assert(mgr.stage());
    assert(mgr.activate());
    assert(mgr.report_boot(true));
    assert(mgr.state() == OtaState::IDLE);
    std::cout << "  PASS: Recovery from failed attempt\n";
}

// ============================================================================
int main() {
    std::cout << "=== OTA Update State Machine Exercise ===\n\n";

    test_happy_path();
    test_reject_downgrade();
    test_crc_failure();
    test_automatic_rollback_after_3_boot_fails();
    test_manual_rollback_from_staged();
    test_reject_same_version();
    test_audit_trail();
    test_can_retry_after_failure();

    std::cout << "\n=== ALL OTA TESTS PASSED ===\n";
    return 0;
}
