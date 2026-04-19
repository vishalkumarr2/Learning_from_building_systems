// puzzle02_dlopen_leak.cpp — dlopen static variable gotcha
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -lrt -ldl -rdynamic puzzle02_dlopen_leak.cpp -o puzzle02
//
// KEY INSIGHT:
//   When you dlclose() a shared library and dlopen() it again, static variables
//   inside that library may or may not be reinitialized. This is IMPLEMENTATION-
//   DEFINED behavior. In practice:
//
//   - glibc with RTLD_NODELETE or reference counting > 0: statics survive dlclose
//   - Some platforms refcount dlopen/dlclose: only truly unload on last dlclose
//   - Static destructors may or may not run on dlclose
//   - If a static object allocated heap memory, dlclose without destructor = LEAK
//
//   This demo uses dlopen(NULL) to load from the main executable (since we can't
//   easily create a .so in an exercise), so the "plugin" is never truly unloaded.
//   Comments explain what would happen with a real .so.

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

// ---------- Simulated "plugin" state ----------
// In a real .so, these would be file-scope statics inside the shared library.
// With dlopen(NULL) they're in the main executable, so dlclose is a no-op.

static int g_call_count = 0;
static bool g_initialized = false;

extern "C" __attribute__((visibility("default")))
void plugin_init() {
    if (!g_initialized) {
        g_call_count = 0;
        g_initialized = true;
        std::printf("  [plugin] Initialized (first time)\n");
    } else {
        std::printf("  [plugin] Already initialized (statics survived!)\n");
    }
}

extern "C" __attribute__((visibility("default")))
int plugin_increment() {
    return ++g_call_count;
}

extern "C" __attribute__((visibility("default")))
int plugin_get_count() {
    return g_call_count;
}

extern "C" __attribute__((visibility("default")))
void plugin_cleanup() {
    std::printf("  [plugin] Cleanup called, count was: %d\n", g_call_count);
    g_initialized = false;
    g_call_count = 0;
}

// ---------- Loader that demonstrates the problem ----------

struct PluginAPI {
    void  (*init)();
    int   (*increment)();
    int   (*get_count)();
    void  (*cleanup)();
};

static bool load_plugin(void*& handle, PluginAPI& api, int flags) {
    handle = dlopen(nullptr, flags);  // Load self
    if (!handle) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return false;
    }

    dlerror();  // clear
    api.init      = reinterpret_cast<void(*)()>(dlsym(handle, "plugin_init"));
    api.increment = reinterpret_cast<int(*)()>(dlsym(handle, "plugin_increment"));
    api.get_count = reinterpret_cast<int(*)()>(dlsym(handle, "plugin_get_count"));
    api.cleanup   = reinterpret_cast<void(*)()>(dlsym(handle, "plugin_cleanup"));

    const char* err = dlerror();
    if (err) {
        std::fprintf(stderr, "dlsym: %s\n", err);
        dlclose(handle);
        return false;
    }
    return true;
}

// ---------- Scenario 1: dlclose + dlopen — statics may persist ----------

static void scenario_statics_persist() {
    std::printf("=== Scenario 1: dlclose + dlopen — do statics reset? ===\n\n");

    void* handle = nullptr;
    PluginAPI api{};

    // First load
    std::printf("--- First dlopen ---\n");
    if (!load_plugin(handle, api, RTLD_NOW)) return;
    api.init();
    std::printf("  Incrementing 5 times...\n");
    for (int i = 0; i < 5; ++i) {
        api.increment();
    }
    std::printf("  Count after 5 increments: %d\n\n", api.get_count());

    // Close (with dlopen(NULL), this is effectively a no-op on the main exe)
    std::printf("--- dlclose ---\n");
    // NOTE: We intentionally do NOT call plugin_cleanup() — simulating a leak
    dlclose(handle);
    handle = nullptr;
    std::printf("  Handle closed (but did statics reset?)\n\n");

    // Reopen
    std::printf("--- Second dlopen ---\n");
    if (!load_plugin(handle, api, RTLD_NOW)) return;
    api.init();
    std::printf("  Count after reopen: %d\n", api.get_count());

    std::printf("\n  ** With dlopen(NULL): statics always persist (main exe never unloads) **\n");
    std::printf("  ** With a real .so: behavior is implementation-defined! **\n");
    std::printf("     - glibc may keep the .so loaded due to refcounting\n");
    std::printf("     - atexit handlers from the .so may or may not fire\n");
    std::printf("     - Static destructors may or may not run\n");

    dlclose(handle);
    api.cleanup();  // reset for next scenario
    std::printf("\n");
}

// ---------- Scenario 2: RTLD_NODELETE flag ----------

static void scenario_nodelete() {
    std::printf("=== Scenario 2: RTLD_NODELETE — intentionally keep statics ===\n\n");

    void* handle = nullptr;
    PluginAPI api{};

    // Load with RTLD_NODELETE — the library will NOT be unloaded on dlclose
    std::printf("--- dlopen with RTLD_NODELETE ---\n");
    if (!load_plugin(handle, api, RTLD_NOW | RTLD_NODELETE)) return;
    api.init();
    for (int i = 0; i < 3; ++i) api.increment();
    std::printf("  Count after 3 increments: %d\n\n", api.get_count());

    // dlclose — handle is released but library stays loaded
    std::printf("--- dlclose (library stays loaded due to RTLD_NODELETE) ---\n");
    dlclose(handle);
    handle = nullptr;

    // Reopen — guaranteed to get the same statics
    std::printf("--- dlopen again ---\n");
    if (!load_plugin(handle, api, RTLD_NOW)) return;
    api.init();
    std::printf("  Count after reopen: %d (statics guaranteed to survive)\n\n", api.get_count());

    std::printf("  ** RTLD_NODELETE makes the behavior deterministic: **\n");
    std::printf("  ** statics always survive across dlclose/dlopen cycles. **\n");
    std::printf("  ** Use this when you need singleton-like plugin state. **\n");

    dlclose(handle);
    api.cleanup();
    std::printf("\n");
}

// ---------- Scenario 3: The resource leak ----------

static void scenario_resource_leak() {
    std::printf("=== Scenario 3: The resource leak problem ===\n\n");

    std::printf("  When a plugin has statics that own heap memory:\n\n");
    std::printf("    static std::vector<DataPoint> history;  // grows over time\n\n");
    std::printf("  If dlclose() doesn't truly unload the library:\n");
    std::printf("    - The vector's destructor never runs\n");
    std::printf("    - The heap memory it allocated is leaked\n");
    std::printf("    - Reopening the plugin finds the old, full vector\n\n");
    std::printf("  If dlclose() DOES unload the library:\n");
    std::printf("    - Static destructors run → vector freed\n");
    std::printf("    - BUT atexit() handlers registered by the plugin may crash\n");
    std::printf("    - Other threads still holding function pointers → use-after-free!\n\n");

    std::printf("  Neither outcome is great. Solutions:\n\n");
    std::printf("  1. RTLD_NODELETE: accept that plugins live forever.\n");
    std::printf("     Provide an explicit cleanup() function in the plugin API.\n\n");
    std::printf("  2. Plugin lifecycle protocol:\n");
    std::printf("       init() → use → cleanup() → dlclose()\n");
    std::printf("     cleanup() releases ALL owned resources before unload.\n\n");
    std::printf("  3. Process isolation: run each plugin in its own process.\n");
    std::printf("     Process exit guarantees full cleanup. Communicate via IPC.\n");
    std::printf("     This is what Chrome does with renderer processes.\n\n");
    std::printf("  4. ROS pluginlib approach: class_loader manages lifecycle,\n");
    std::printf("     calls destructors before unloading, uses RTLD_LAZY.\n");
}

// ---------- main ----------

int main() {
    std::printf("=== dlopen Static Variable Gotcha ===\n\n");

    scenario_statics_persist();
    scenario_nodelete();
    scenario_resource_leak();

    std::printf("\n=== Summary ===\n");
    std::printf("  - Static variables in dlopen'd libraries have ambiguous lifetime\n");
    std::printf("  - dlclose may or may not run destructors (implementation-defined)\n");
    std::printf("  - RTLD_NODELETE makes persistence explicit and deterministic\n");
    std::printf("  - Always provide explicit init()/cleanup() in plugin APIs\n");
    std::printf("  - Best practice: treat plugins as permanent once loaded\n");

    return 0;
}
