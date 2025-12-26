/* SPDX-License-Identifier: MIT
 *
 * Example: APX Hot Path Integration Demo
 *
 * This example demonstrates the APX hot path optimization pipeline:
 * 1. Detects APX CPU features
 * 2. Registers code regions for hot path monitoring
 * 3. Shows how code gets lifted, APX-optimized, and cached
 * 4. Demonstrates transparent routing to APX-optimized code
 *
 * Build:
 *   g++ -std=c++20 -O2 -I. -I../include -o example_apx_hotpath \
 *       example_apx_hotpath.cpp -L../../build -lbpftime_apx_jit -lspdlog -lpthread
 *
 * Run with SDE for APX emulation:
 *   sde64 -future -- ./example_apx_hotpath
 */

#include "apx_cpu_features.hpp"
#include "apx_hotpath_manager.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>

using namespace bpftime::vm::apx;

// Example function to be APX-optimized
// In real usage, this would be JIT-compiled eBPF code
uint64_t example_compute(void* mem, size_t mem_len) {
    // Simple computation that could benefit from more registers
    uint64_t* data = static_cast<uint64_t*>(mem);
    size_t count = mem_len / sizeof(uint64_t);

    uint64_t sum = 0;
    uint64_t prod = 1;
    uint64_t xor_val = 0;
    uint64_t min_val = UINT64_MAX;
    uint64_t max_val = 0;

    for (size_t i = 0; i < count; i++) {
        uint64_t val = data[i];
        sum += val;
        prod *= (val | 1);  // Avoid zero
        xor_val ^= val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    return sum ^ prod ^ xor_val ^ min_val ^ max_val;
}

void print_apx_features(const APXFeatures& features) {
    std::cout << "\n=== APX CPU Feature Detection ===" << std::endl;
    std::cout << "APX_F (Foundation):     " << (features.has_apx_f ? "YES" : "NO") << std::endl;
    std::cout << "Extended GPRs (R16-31): " << (features.has_egpr ? "YES" : "NO") << std::endl;
    std::cout << "NDD (3-operand):        " << (features.has_ndd ? "YES" : "NO") << std::endl;
    std::cout << "NF (Flag suppression):  " << (features.has_nf ? "YES" : "NO") << std::endl;
    std::cout << "PUSH2/POP2:            " << (features.has_push2pop2 ? "YES" : "NO") << std::endl;
    std::cout << "AVX-512:               " << (features.has_avx512 ? "YES" : "NO") << std::endl;
    std::cout << "AVX10:                 " << (features.has_avx10 ? "YES" : "NO") << std::endl;
    std::cout << "XCR0:                  0x" << std::hex << features.xcr0 << std::dec << std::endl;
    std::cout << "Can use APX:           " << (features.can_use_apx() ? "YES" : "NO") << std::endl;
}

void print_stats(const APXHotPathStats& stats) {
    std::cout << "\n=== APX Hot Path Statistics ===" << std::endl;
    std::cout << "Total executions:       " << stats.total_executions << std::endl;
    std::cout << "Hot path hits:          " << stats.hot_path_hits << std::endl;
    std::cout << "Cold path hits:         " << stats.cold_path_hits << std::endl;
    std::cout << "Translations:           " << stats.translations << std::endl;
    std::cout << "Translation failures:   " << stats.translation_failures << std::endl;
    std::cout << "APX optimizations:      " << stats.apx_optimizations_applied << std::endl;
    std::cout << "Registers saved:        " << stats.registers_saved_by_apx << std::endl;
    std::cout << "XSAVE calls:            " << stats.xsave_calls << std::endl;
    std::cout << "Translation time (ns):  " << stats.total_translation_time_ns << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "    APX Hot Path Integration Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Detect APX features
    auto features = detect_apx_features();
    print_apx_features(features);

    if (!features.can_use_apx()) {
        std::cout << "\nNote: APX not available. Run with 'sde64 -future' to emulate.\n";
        std::cout << "Demo will continue with fallback path.\n";
    }

    // 2. Configure and initialize hot path manager
    APXHotPathConfig config;
    config.hot_threshold = 10;           // Low threshold for demo
    config.enable_apx = true;
    config.enable_ndd = true;
    config.enable_nf = true;
    config.selective_xsave = true;
    config.code_cache_size = 4 * 1024 * 1024;  // 4MB cache
    config.collect_stats = true;
    config.verbose_logging = true;

    APXHotPathManager manager(config);

    if (!manager.initialize()) {
        std::cerr << "Failed to initialize APX hot path manager" << std::endl;
        return 1;
    }

    std::cout << "\n=== APX Hot Path Manager Initialized ===" << std::endl;
    std::cout << "APX available: " << (manager.is_apx_available() ? "YES" : "NO") << std::endl;

    // 3. Register a code region
    uint64_t func_addr = reinterpret_cast<uint64_t>(&example_compute);
    size_t func_size = 256;  // Estimate

    uint64_t region_id = manager.register_region(func_addr, func_size, "example_compute");
    std::cout << "Registered region " << region_id << " at 0x"
              << std::hex << func_addr << std::dec << std::endl;

    // 4. Set up code read callback (reads from our process memory)
    manager.set_code_read_callback([](uint64_t addr, uint8_t* buf, size_t len) -> bool {
        memcpy(buf, reinterpret_cast<void*>(addr), len);
        return true;
    });

    // 5. Prepare test data
    constexpr size_t DATA_SIZE = 1024;
    uint64_t data[DATA_SIZE];
    for (size_t i = 0; i < DATA_SIZE; i++) {
        data[i] = i * 7 + 13;
    }

    // 6. Execute multiple times to trigger hot path detection
    std::cout << "\n=== Executing to Trigger Hot Path Detection ===" << std::endl;

    uint64_t baseline_result = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iteration = 0; iteration < 50; iteration++) {
        // Call on_execute to track and potentially use APX version
        void* exec_ptr = manager.on_execute(region_id);

        if (exec_ptr && exec_ptr != reinterpret_cast<void*>(func_addr)) {
            // We have an APX-optimized version
            using Func = uint64_t(*)(void*, size_t);
            auto apx_func = reinterpret_cast<Func>(exec_ptr);
            baseline_result = apx_func(data, sizeof(data));

            if (iteration == config.hot_threshold + 1) {
                std::cout << "  [Iteration " << iteration << "] Using APX-optimized code!" << std::endl;
            }
        } else {
            // Use original function
            baseline_result = example_compute(data, sizeof(data));
        }

        if (iteration < 5 || (iteration + 1) % 10 == 0) {
            std::cout << "  [Iteration " << std::setw(2) << iteration
                      << "] Result: " << baseline_result << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "\nTotal execution time: " << duration.count() << " us" << std::endl;

    // 7. Print statistics
    auto stats = manager.get_stats();
    print_stats(stats);

    // 8. Get region info
    const CodeRegion* region_info = manager.get_region_info(region_id);
    if (region_info) {
        std::cout << "\n=== Region Info ===" << std::endl;
        std::cout << "Start address:   0x" << std::hex << region_info->start_addr << std::dec << std::endl;
        std::cout << "Execution count: " << region_info->execution_count.load() << std::endl;
        std::cout << "State:           ";
        switch (region_info->state.load()) {
            case CodeRegion::State::COLD: std::cout << "COLD"; break;
            case CodeRegion::State::PROFILING: std::cout << "PROFILING"; break;
            case CodeRegion::State::TRANSLATING: std::cout << "TRANSLATING"; break;
            case CodeRegion::State::HOT: std::cout << "HOT"; break;
            case CodeRegion::State::FAILED: std::cout << "FAILED"; break;
        }
        std::cout << std::endl;

        if (region_info->apx_code_ptr) {
            std::cout << "APX code ptr:    0x" << std::hex
                      << reinterpret_cast<uint64_t>(region_info->apx_code_ptr)
                      << std::dec << std::endl;
            std::cout << "APX code size:   " << region_info->apx_code_size << " bytes" << std::endl;
            std::cout << "APX regs used:   0x" << std::hex << region_info->apx_regs_used
                      << std::dec << " (" << __builtin_popcount(region_info->apx_regs_used)
                      << " regs)" << std::endl;
            std::cout << "XSAVE mask:      0x" << std::hex << region_info->xsave_mask
                      << std::dec << std::endl;
        }
    }

    // 9. Cleanup
    manager.unregister_region(region_id);

    std::cout << "\n========================================" << std::endl;
    std::cout << "    Demo Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
