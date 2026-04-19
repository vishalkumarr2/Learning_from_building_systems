// =============================================================================
// Exercise 2: Rule of Five — MappedFile
// =============================================================================
// Implement a RAII class that owns an mmap'd file descriptor.
// You MUST implement all 5 special members correctly.
//
// Move semantics: moved-from objects must be in a valid-but-empty state.
// The destructor must handle both "live" and "moved-from" objects.
//
// DELIBERATE BUGS: The "BuggyMappedFile" class below has 4 bugs.
// Find them all. The correct version is "MappedFile".
//
// Build: g++ -std=c++2a -Wall -Wextra -Wpedantic ex02_rule_of_five.cpp
// =============================================================================

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

// =============================================================================
// BUGGY VERSION — Find the 4 bugs!
// =============================================================================
// (Read this first, identify the bugs, then look at the correct version below.)
//
// Bug 1: Copy constructor doesn't dup the fd — both objects close the same fd.
// Bug 2: Move assignment doesn't release current resources before stealing.
// Bug 3: Move constructor is missing noexcept — std::vector will copy instead.
// Bug 4: Copy assignment doesn't handle self-assignment (a = a crashes).
//
// class BuggyMappedFile {
//     int fd_ = -1;
//     void* addr_ = MAP_FAILED;
//     size_t size_ = 0;
//     std::string path_;
// public:
//     explicit BuggyMappedFile(const char* path) { /* correct */ }
//     ~BuggyMappedFile() { /* correct */ }
//
//     // Bug 1: shallow copy of fd — double close
//     BuggyMappedFile(const BuggyMappedFile& other)
//         : fd_(other.fd_), addr_(other.addr_), size_(other.size_), path_(other.path_) {}
//
//     // Bug 4: no self-assignment check
//     BuggyMappedFile& operator=(const BuggyMappedFile& other) {
//         close();  // release THEN check... if &other == this, data is gone
//         fd_ = ::dup(other.fd_);
//         // ...
//     }
//
//     // Bug 3: missing noexcept
//     BuggyMappedFile(BuggyMappedFile&& other)
//         : fd_(other.fd_), addr_(other.addr_), size_(other.size_),
//           path_(std::move(other.path_)) {
//         other.fd_ = -1; other.addr_ = MAP_FAILED; other.size_ = 0;
//     }
//
//     // Bug 2: doesn't release current resources
//     BuggyMappedFile& operator=(BuggyMappedFile&& other) {
//         // Missing: munmap + close current!
//         fd_ = other.fd_; addr_ = other.addr_; size_ = other.size_;
//         path_ = std::move(other.path_);
//         other.fd_ = -1; other.addr_ = MAP_FAILED; other.size_ = 0;
//         return *this;
//     }
// };

// =============================================================================
// CORRECT VERSION
// =============================================================================

class MappedFile {
    int fd_ = -1;
    void* addr_ = MAP_FAILED;
    size_t size_ = 0;
    std::string path_;

    void release() noexcept {
        if (addr_ != MAP_FAILED && addr_ != nullptr) {
            ::munmap(addr_, size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = -1;
        addr_ = MAP_FAILED;
        size_ = 0;
        // path_ is left as-is for copy to clean up via assignment
    }

public:
    // --- Constructor ---
    explicit MappedFile(const char* path) : path_(path ? path : "") {
        if (!path) return;

        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) return;  // leave in empty state

        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            return;
        }

        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) {
            // Can't mmap empty file, but fd is valid
            addr_ = nullptr;
            return;
        }

        addr_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (addr_ == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            size_ = 0;
        }
    }

    // --- Destructor ---
    ~MappedFile() {
        release();
    }

    // --- Copy constructor (deep copy via dup + re-mmap) ---
    MappedFile(const MappedFile& other)
        : size_(other.size_), path_(other.path_) {
        if (other.fd_ < 0) return;

        fd_ = ::dup(other.fd_);
        if (fd_ < 0) {
            size_ = 0;
            return;
        }

        if (size_ > 0) {
            addr_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (addr_ == MAP_FAILED) {
                ::close(fd_);
                fd_ = -1;
                size_ = 0;
            }
        } else {
            addr_ = nullptr;
        }
    }

    // --- Copy assignment (copy-and-swap idiom) ---
    MappedFile& operator=(const MappedFile& other) {
        if (this == &other) return *this;  // self-assignment guard
        MappedFile tmp(other);             // copy
        swap(*this, tmp);                  // swap (tmp now holds old resources)
        return *this;                      // tmp destructor releases old resources
    }

    // --- Move constructor (noexcept is CRITICAL for vector reallocation) ---
    MappedFile(MappedFile&& other) noexcept
        : fd_(other.fd_),
          addr_(other.addr_),
          size_(other.size_),
          path_(std::move(other.path_)) {
        // Leave source in valid-but-empty state
        other.fd_ = -1;
        other.addr_ = MAP_FAILED;
        other.size_ = 0;
    }

    // --- Move assignment ---
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this == &other) return *this;
        release();  // free OUR current resources first
        fd_ = other.fd_;
        addr_ = other.addr_;
        size_ = other.size_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
        other.addr_ = MAP_FAILED;
        other.size_ = 0;
        return *this;
    }

    // --- Swap (used by copy-and-swap) ---
    friend void swap(MappedFile& a, MappedFile& b) noexcept {
        using std::swap;
        swap(a.fd_, b.fd_);
        swap(a.addr_, b.addr_);
        swap(a.size_, b.size_);
        swap(a.path_, b.path_);
    }

    // --- Accessors ---
    [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }
    [[nodiscard]] bool is_mapped() const noexcept {
        return addr_ != MAP_FAILED && addr_ != nullptr;
    }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

    [[nodiscard]] const void* data() const noexcept {
        return is_mapped() ? addr_ : nullptr;
    }

    [[nodiscard]] std::string_view as_string_view() const noexcept {
        if (!is_mapped()) return {};
        return {static_cast<const char*>(addr_), size_};
    }
};

// =============================================================================
// Tests
// =============================================================================

void test_basic_open() {
    std::cout << "test_basic_open: ";
    // Use /etc/hostname or /etc/os-release — should exist on any Linux
    MappedFile f("/etc/hostname");
    if (f.is_open()) {
        assert(f.size() > 0);
        assert(f.is_mapped());
        assert(!f.path().empty());
        std::cout << "PASS (size=" << f.size() << ")\n";
    } else {
        // Fallback: try /etc/os-release
        MappedFile f2("/etc/os-release");
        assert(f2.is_open());
        std::cout << "PASS (used /etc/os-release, size=" << f2.size() << ")\n";
    }
}

void test_invalid_path() {
    std::cout << "test_invalid_path: ";
    MappedFile f("/nonexistent/path/that/does/not/exist");
    assert(!f.is_open());
    assert(!f.is_mapped());
    assert(f.size() == 0);
    assert(f.data() == nullptr);
    std::cout << "PASS\n";
}

void test_move_constructor() {
    std::cout << "test_move_constructor: ";
    MappedFile f1("/etc/hostname");
    if (!f1.is_open()) { std::cout << "SKIP (can't open file)\n"; return; }

    size_t orig_size = f1.size();
    MappedFile f2(std::move(f1));

    // f2 should have the resources
    assert(f2.is_open());
    assert(f2.size() == orig_size);

    // f1 should be empty but valid
    assert(!f1.is_open());
    assert(f1.size() == 0);
    assert(f1.data() == nullptr);

    // Verify f1 is destructible (happens automatically at scope end)
    std::cout << "PASS\n";
}

void test_move_assignment() {
    std::cout << "test_move_assignment: ";
    MappedFile f1("/etc/hostname");
    if (!f1.is_open()) { std::cout << "SKIP\n"; return; }

    size_t orig_size = f1.size();

    MappedFile f2("/etc/os-release");  // open a DIFFERENT file
    f2 = std::move(f1);               // should release f2's old file, take f1's

    assert(f2.is_open());
    assert(f2.size() == orig_size);
    assert(!f1.is_open());
    assert(f1.size() == 0);
    std::cout << "PASS\n";
}

void test_copy_constructor() {
    std::cout << "test_copy_constructor: ";
    MappedFile f1("/etc/hostname");
    if (!f1.is_open()) { std::cout << "SKIP\n"; return; }

    MappedFile f2(f1);  // deep copy

    // BOTH should be open
    assert(f1.is_open());
    assert(f2.is_open());
    assert(f1.size() == f2.size());
    // Data pointers should be DIFFERENT (independent mmaps)
    assert(f1.data() != f2.data());
    // But contents should be the same
    assert(f1.as_string_view() == f2.as_string_view());

    std::cout << "PASS\n";
}

void test_self_assignment() {
    std::cout << "test_self_assignment: ";
    MappedFile f("/etc/hostname");
    if (!f.is_open()) { std::cout << "SKIP\n"; return; }

    size_t orig_size = f.size();
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpragmas"
    #pragma GCC diagnostic ignored "-Wself-move"
    f = f;  // must not crash or corrupt
    #pragma GCC diagnostic pop

    assert(f.is_open());
    assert(f.size() == orig_size);
    std::cout << "PASS\n";
}

void test_moved_from_is_assignable() {
    std::cout << "test_moved_from_is_assignable: ";
    MappedFile f1("/etc/hostname");
    if (!f1.is_open()) { std::cout << "SKIP\n"; return; }

    MappedFile f2(std::move(f1));  // f1 is now moved-from

    // Key test: can we assign to a moved-from object?
    f1 = MappedFile("/etc/os-release");
    if (f1.is_open()) {
        assert(f1.size() > 0);
        std::cout << "PASS (re-assigned to moved-from object)\n";
    } else {
        std::cout << "PASS (file not available, but no crash)\n";
    }
}

void test_noexcept_move() {
    std::cout << "test_noexcept_move: ";
    // This is what std::vector checks before deciding to move vs copy
    static_assert(std::is_nothrow_move_constructible_v<MappedFile>,
        "Move constructor MUST be noexcept for vector optimization");
    static_assert(std::is_nothrow_move_assignable_v<MappedFile>,
        "Move assignment MUST be noexcept");
    std::cout << "PASS (noexcept verified at compile time)\n";
}

int main() {
    std::cout << "=== Rule of Five: MappedFile Tests ===\n\n";

    test_basic_open();
    test_invalid_path();
    test_move_constructor();
    test_move_assignment();
    test_copy_constructor();
    test_self_assignment();
    test_moved_from_is_assignable();
    test_noexcept_move();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
