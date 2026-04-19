// puzzle01_placement_new.cpp — The placement-new-in-shared-memory gotcha
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -lrt -ldl puzzle01_placement_new.cpp -o puzzle01
//
// KEY INSIGHT:
//   Placement new constructs an object at a specific memory address, but the
//   object's INTERNAL allocations (like std::string's heap buffer) still happen
//   on the process-local heap. Another process mapping the same shared memory
//   will see the std::string metadata pointing to an address in YOUR process's
//   heap — which is garbage in THEIR address space.
//
// This is the #1 bug in shared-memory programming and the reason ROS2 uses
// only trivially_copyable types (or custom allocators) in shared memory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>         // placement new
#include <string>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------- A non-trivial type (BAD for shared memory) ----------

struct SensorConfig {
    int sensor_id;
    double calibration_offset;
    std::string name;  // <-- THIS IS THE PROBLEM

    void print() const {
        std::printf("  sensor_id=%d, offset=%.2f, name='%s'\n",
                    sensor_id, calibration_offset, name.c_str());
    }
};

// ---------- A trivially-copyable type (SAFE for shared memory) ----------

struct SensorConfigSafe {
    int sensor_id;
    double calibration_offset;
    char name[64];  // fixed-size buffer — no heap allocation

    void print() const {
        std::printf("  sensor_id=%d, offset=%.2f, name='%s'\n",
                    sensor_id, calibration_offset, name);
    }
};

// static_assert proves the point at compile time
static_assert(!std::is_trivially_copyable_v<SensorConfig>,
              "SensorConfig has std::string — NOT trivially copyable");

static_assert(std::is_trivially_copyable_v<SensorConfigSafe>,
              "SensorConfigSafe uses char[] — IS trivially copyable");

// ---------- Demonstrate the problem (single process, but explains the bug) ----------

static void demonstrate_bad_pattern() {
    std::printf("=== BAD PATTERN: placement new with std::string ===\n\n");

    constexpr std::size_t SHM_SIZE = 4096;
    const char* SHM_NAME = "/puzzle01_bad";

    // Create shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { std::perror("shm_open"); return; }
    ftruncate(fd, SHM_SIZE);

    void* ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) { std::perror("mmap"); shm_unlink(SHM_NAME); return; }

    std::printf("Shared memory mapped at: %p\n", ptr);

    // Placement new — constructs SensorConfig in shared memory
    auto* config = new (ptr) SensorConfig{};
    config->sensor_id = 42;
    config->calibration_offset = 3.14;
    config->name = "IMU_LEFT_FRONT";  // Allocates on PROCESS heap!

    std::printf("Object constructed in shared memory:\n");
    config->print();

    // Show where the string's internal buffer lives
    std::printf("\n  Object address (in shm): %p\n", static_cast<void*>(config));
    std::printf("  String data address:     %p\n",
                static_cast<const void*>(config->name.c_str()));
    std::printf("  String data is %s shared memory!\n",
                (reinterpret_cast<uintptr_t>(config->name.c_str()) >=
                     reinterpret_cast<uintptr_t>(ptr) &&
                 reinterpret_cast<uintptr_t>(config->name.c_str()) <
                     reinterpret_cast<uintptr_t>(ptr) + SHM_SIZE)
                    ? "INSIDE" : "OUTSIDE");

    std::printf("\n  ** The string buffer is on the PROCESS heap, not in shared memory! **\n");
    std::printf("  ** Another process mapping this shm would see a dangling pointer. **\n");
    std::printf("  ** Calling the destructor from another process would try to free **\n");
    std::printf("  ** memory at an address that doesn't belong to it — CRASH! **\n\n");

    // Must manually call destructor since we used placement new
    config->~SensorConfig();

    munmap(ptr, SHM_SIZE);
    shm_unlink(SHM_NAME);
}

// ---------- Demonstrate the correct pattern ----------

static void demonstrate_good_pattern() {
    std::printf("=== GOOD PATTERN: trivially_copyable types only ===\n\n");

    constexpr std::size_t SHM_SIZE = 4096;
    const char* SHM_NAME = "/puzzle01_good";

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { std::perror("shm_open"); return; }
    ftruncate(fd, SHM_SIZE);

    void* ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) { std::perror("mmap"); shm_unlink(SHM_NAME); return; }

    // Placement new with a safe type — or just memcpy, since it's trivially copyable
    auto* config = new (ptr) SensorConfigSafe{};
    config->sensor_id = 42;
    config->calibration_offset = 3.14;
    std::strncpy(config->name, "IMU_LEFT_FRONT", sizeof(config->name) - 1);
    config->name[sizeof(config->name) - 1] = '\0';

    std::printf("Object constructed in shared memory:\n");
    config->print();

    std::printf("\n  Object address (in shm): %p\n", static_cast<void*>(config));
    std::printf("  Name buffer address:     %p\n", static_cast<void*>(config->name));
    std::printf("  Name buffer is %s shared memory.\n",
                (reinterpret_cast<uintptr_t>(config->name) >=
                     reinterpret_cast<uintptr_t>(ptr) &&
                 reinterpret_cast<uintptr_t>(config->name) <
                     reinterpret_cast<uintptr_t>(ptr) + SHM_SIZE)
                    ? "INSIDE" : "OUTSIDE");

    std::printf("\n  ** Fixed-size char[] lives entirely within the shared memory region. **\n");
    std::printf("  ** Another process can safely read it via mmap. **\n");
    std::printf("  ** No destructor issues — trivially_copyable means no cleanup needed. **\n");

    // No destructor call needed for trivially copyable types
    // (but it's harmless to call since it's the default destructor)

    munmap(ptr, SHM_SIZE);
    shm_unlink(SHM_NAME);
}

// ---------- Compile-time guard you should use in production ----------

template <typename T, typename... Args>
void* safe_shm_construct(void* shm_ptr, Args&&... args) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Only trivially_copyable types are safe in shared memory! "
                  "std::string, std::vector, etc. allocate on the process heap.");
    return new (shm_ptr) T(std::forward<Args>(args)...);
}

int main() {
    demonstrate_bad_pattern();
    std::printf("\n");
    demonstrate_good_pattern();

    std::printf("\n=== Compile-time Guard Demo ===\n\n");
    // This would compile:
    std::printf("  safe_shm_construct<SensorConfigSafe>() — compiles OK\n");

    // Uncomment the next line to see the static_assert fire:
    // char buf[4096];
    // safe_shm_construct<SensorConfig>(buf);  // COMPILE ERROR!

    std::printf("  safe_shm_construct<SensorConfig>() — would fail static_assert\n");
    std::printf("  (uncomment line in main() to verify)\n");

    std::printf("\n=== Rules for Shared Memory Types ===\n");
    std::printf("  1. Use static_assert(is_trivially_copyable_v<T>) on every shared type\n");
    std::printf("  2. No std::string, std::vector, std::map, or any heap-allocating member\n");
    std::printf("  3. Use fixed-size arrays (char[64]) instead of std::string\n");
    std::printf("  4. Use C-style arrays or std::array instead of std::vector\n");
    std::printf("  5. Alternative: use a shared-memory allocator (Boost.Interprocess)\n");
    std::printf("     but that's complex — prefer flat structures when possible\n");

    return 0;
}
