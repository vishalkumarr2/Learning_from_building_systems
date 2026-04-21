// ex04_plugin_loader.cpp — Plugin system using dlopen/dlsym
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -lrt -ldl -rdynamic ex04_plugin_loader.cpp -o ex04
//
// NOTE: -rdynamic is critical — it exports symbols from the main executable so
// dlopen(NULL) + dlsym can find them. Without it, dlsym returns nullptr.
//
// Real-world pattern:
//   1. Define IController in a shared header (controller_interface.h)
//   2. Build each controller as a .so:
//        g++ -shared -fPIC -o libpid_controller.so pid_controller.cpp
//   3. Each .so exports:  extern "C" IController* create_controller();
//   4. Host loads .so at runtime:  dlopen("libpid_controller.so", RTLD_NOW)
//
// ROS pluginlib does exactly this but adds:
//   - XML manifest for discovery (plugin_description.xml)
//   - Class loader that manages lifetime and avoids manual dlopen
//   - PLUGINLIB_EXPORT_CLASS macro that registers the factory
//   - base_class / derived_class type erasure via templates

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <vector>
#include <string>

// ---------- Interface definition (would be in a shared header) ----------

class IController {
public:
    virtual ~IController() = default;
    virtual double compute(double input) = 0;
    virtual const char* name() const = 0;
};

// ---------- A concrete controller (would live in a separate .so) ----------

class PIDController : public IController {
    double kp_, ki_, kd_;
    double integral_ = 0.0;
    double prev_error_ = 0.0;

public:
    PIDController(double kp, double ki, double kd)
        : kp_(kp), ki_(ki), kd_(kd) {}

    double compute(double error) override {
        integral_ += error;
        double derivative = error - prev_error_;
        prev_error_ = error;
        return kp_ * error + ki_ * integral_ + kd_ * derivative;
    }

    const char* name() const override { return "PIDController(kp=1.0, ki=0.1, kd=0.01)"; }
};

// ---------- Factory function — exported from main executable ----------
// In a real .so, this would be the only exported symbol.
// extern "C" prevents name mangling so dlsym can find it by string name.

extern "C" __attribute__((visibility("default")))
IController* create_controller() {
    return new PIDController(1.0, 0.1, 0.01);
}

// ---------- PluginLoader — generic loader for any IController .so ----------

class PluginLoader {
    using FactoryFn = IController* (*)();

    void* handle_ = nullptr;
    FactoryFn factory_ = nullptr;
    std::string path_;

public:
    // path: path to .so, or NULL string to load from main executable
    bool load(const char* path) {
        path_ = path ? path : "(self)";

        // dlopen(NULL) loads the main executable's symbol table
        // For a real plugin: dlopen("libfoo.so", RTLD_NOW)
        handle_ = dlopen(path, RTLD_NOW);
        if (!handle_) {
            std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
            return false;
        }

        // Clear any prior error
        dlerror();

        // Look up the factory function
        void* sym = dlsym(handle_, "create_controller");
        const char* err = dlerror();
        if (err) {
            std::fprintf(stderr, "dlsym failed: %s\n", err);
            dlclose(handle_);
            handle_ = nullptr;
            return false;
        }

        // Cast void* to function pointer (C-style cast required by POSIX)
        factory_ = reinterpret_cast<FactoryFn>(sym);
        return true;
    }

    // Create an instance via the factory
    std::unique_ptr<IController> create() const {
        if (!factory_) {
            std::fprintf(stderr, "No factory loaded\n");
            return nullptr;
        }
        return std::unique_ptr<IController>(factory_());
    }

    const std::string& path() const { return path_; }

    ~PluginLoader() {
        if (handle_ && handle_ != dlopen(nullptr, RTLD_NOW)) {
            // Only dlclose real .so handles, not the self-handle
            dlclose(handle_);
        }
    }

    // Non-copyable
    PluginLoader(const PluginLoader&) = delete;
    PluginLoader& operator=(const PluginLoader&) = delete;
    PluginLoader() = default;
};

// ---------- main ----------

int main() {
    std::printf("=== Plugin Loader Demo ===\n\n");

    PluginLoader loader;

    // Load from main executable (dlopen(NULL))
    // In production: loader.load("/opt/ros/plugins/libpid_controller.so")
    if (!loader.load(nullptr)) {
        std::fprintf(stderr, "Failed to load plugin\n");
        return 1;
    }

    std::printf("Loaded plugin from: %s\n", loader.path().c_str());

    auto controller = loader.create();
    if (!controller) {
        std::fprintf(stderr, "Failed to create controller\n");
        return 1;
    }

    std::printf("Controller name: %s\n\n", controller->name());

    // Simulate a control loop — error decreasing over time
    std::printf("  Step | Error  | Output\n");
    std::printf("  -----+--------+--------\n");

    double setpoint = 10.0;
    double position = 0.0;

    for (int i = 0; i < 10; ++i) {
        double error = setpoint - position;
        double output = controller->compute(error);
        std::printf("  %4d | %6.2f | %6.2f\n", i, error, output);
        position += output * 0.1;  // simple integration
    }

    std::printf("\n--- How to create a real .so plugin ---\n");
    std::printf("1. Put PIDController in pid_controller.cpp with the factory:\n");
    std::printf("     extern \"C\" IController* create_controller() { return new PIDController(...); }\n");
    std::printf("2. Compile: g++ -shared -fPIC -o libpid.so pid_controller.cpp\n");
    std::printf("3. Load:    loader.load(\"./libpid.so\")\n");
    std::printf("4. ROS equivalent: PLUGINLIB_EXPORT_CLASS(PIDController, IController)\n");

    return 0;
}
