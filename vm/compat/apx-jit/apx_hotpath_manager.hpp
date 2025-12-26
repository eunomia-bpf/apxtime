/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * APX Hot Path Manager
 *
 * This module provides transparent APX optimization for hot code paths:
 * 1. Profiles code execution to detect hot paths
 * 2. Lifts hot paths to LLVM IR (via Rellume or from eBPF JIT output)
 * 3. Applies APX-specific optimizations (R16-R31, NDD, NF)
 * 4. Emits APX machine code to code cache
 * 5. Transparently routes execution through APX-optimized version
 */

#ifndef _BPFTIME_APX_HOTPATH_MANAGER_HPP
#define _BPFTIME_APX_HOTPATH_MANAGER_HPP

#include <cstdint>
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <atomic>
#include <unordered_map>

namespace bpftime::vm::apx {

// Forward declarations
struct APXFeatures;
class RellumeLifter;

/**
 * @brief Configuration for hot path detection and APX optimization
 */
struct APXHotPathConfig {
    // Profiling thresholds
    uint64_t hot_threshold = 1000;        // Executions before considering "hot"
    uint64_t recompile_threshold = 10000; // Executions before re-optimization

    // APX optimization settings
    bool enable_apx = true;               // Use APX if available
    bool enable_ndd = true;               // Use 3-operand NDD forms
    bool enable_nf = true;                // Use flag suppression
    bool enable_push2pop2 = true;         // Use PUSH2/POP2

    // Register allocation hints
    uint32_t preferred_apx_regs = 8;      // How many R16-R31 to prefer
    bool spill_to_apx_first = true;       // Spill to R16-R31 before stack

    // Code cache settings
    size_t code_cache_size = 64 * 1024 * 1024;  // 64MB default
    bool enable_code_cache = true;

    // XSAVE optimization
    bool selective_xsave = true;          // Only save modified APX regs

    // Debug/profiling
    bool collect_stats = true;
    bool verbose_logging = false;
};

/**
 * @brief Statistics for APX hot path optimization
 *
 * Note: These are plain integers for copyability. The Impl class
 * uses atomic operations internally for thread-safe updates.
 */
struct APXHotPathStats {
    uint64_t total_executions = 0;
    uint64_t hot_path_hits = 0;
    uint64_t cold_path_hits = 0;
    uint64_t translations = 0;
    uint64_t translation_failures = 0;
    uint64_t apx_optimizations_applied = 0;
    uint64_t registers_saved_by_apx = 0;  // Estimated spills avoided
    uint64_t xsave_calls = 0;
    uint64_t xsave_bytes_saved = 0;       // Bytes not saved due to selective XSAVE

    // Timing (in nanoseconds)
    uint64_t total_translation_time_ns = 0;
    uint64_t total_execution_time_ns = 0;
};

/**
 * @brief Code region descriptor for hot path tracking
 */
struct CodeRegion {
    uint64_t start_addr;
    uint64_t end_addr;
    size_t size;

    // Profiling data
    std::atomic<uint64_t> execution_count{0};
    std::atomic<uint64_t> last_execution_time{0};

    // Translation state
    enum class State {
        COLD,           // Not yet hot
        PROFILING,      // Collecting profile data
        TRANSLATING,    // Currently being translated
        HOT,            // APX version available
        FAILED          // Translation failed
    };
    std::atomic<State> state{State::COLD};

    // APX translation result
    void* apx_code_ptr = nullptr;
    size_t apx_code_size = 0;
    uint64_t xsave_mask = 0;
    uint32_t apx_regs_used = 0;  // Bitmask of R16-R31 used

    // Original code backup for patching
    std::vector<uint8_t> original_prologue;
    bool is_patched = false;
};

/**
 * @brief Trampoline for routing execution to APX code
 *
 * When a hot path is detected, we patch the original code's prologue
 * to jump to this trampoline, which:
 * 1. Saves APX state (selective XSAVE)
 * 2. Calls APX-optimized code
 * 3. Restores APX state (selective XRSTOR)
 * 4. Returns to caller
 */
struct APXTrampoline {
    void* trampoline_code;
    size_t trampoline_size;

    // State save area (64-byte aligned for XSAVE)
    alignas(64) uint8_t xsave_area[8192];

    // Back-pointer to code region
    CodeRegion* region;
};

/**
 * @brief Callback for reading original code bytes
 */
using CodeReadCallback = std::function<bool(uint64_t addr, uint8_t* buf, size_t len)>;

/**
 * @brief Callback for patching original code
 */
using CodePatchCallback = std::function<bool(uint64_t addr, const uint8_t* patch, size_t len)>;

/**
 * @brief APX Hot Path Manager
 *
 * Main integration point for transparent APX optimization.
 *
 * Usage:
 *   1. Create manager with config
 *   2. Register code regions to monitor
 *   3. Call on_execute() at region entry points
 *   4. Manager automatically:
 *      - Profiles execution frequency
 *      - Lifts hot paths to LLVM IR
 *      - Applies APX optimizations
 *      - Patches code to route through APX version
 */
class APXHotPathManager {
public:
    APXHotPathManager();
    explicit APXHotPathManager(const APXHotPathConfig& config);
    ~APXHotPathManager();

    // Non-copyable
    APXHotPathManager(const APXHotPathManager&) = delete;
    APXHotPathManager& operator=(const APXHotPathManager&) = delete;

    /**
     * @brief Initialize the manager
     *
     * Must be called before use. Detects APX features and allocates
     * code cache.
     *
     * @return true if APX is available and initialization succeeded
     */
    bool initialize();

    /**
     * @brief Check if APX optimization is available
     */
    bool is_apx_available() const;

    /**
     * @brief Register a code region for hot path monitoring
     *
     * @param start_addr Start address of code region
     * @param size Size in bytes
     * @param name Optional name for debugging
     * @return Region ID for future reference
     */
    uint64_t register_region(uint64_t start_addr, size_t size,
                             const std::string& name = "");

    /**
     * @brief Unregister a code region
     */
    void unregister_region(uint64_t region_id);

    /**
     * @brief Notify execution of a region
     *
     * Call this at the entry point of monitored code.
     * Returns the address to actually execute (original or APX).
     *
     * @param region_id Region ID from register_region
     * @return Address to execute (may be APX-optimized version)
     */
    void* on_execute(uint64_t region_id);

    /**
     * @brief Manually trigger APX translation for a region
     *
     * @param region_id Region to translate
     * @param code Code bytes to translate
     * @param code_size Size of code
     * @return true if translation succeeded
     */
    bool translate_region(uint64_t region_id,
                         const uint8_t* code, size_t code_size);

    /**
     * @brief Set callback for reading code bytes
     */
    void set_code_read_callback(CodeReadCallback callback);

    /**
     * @brief Set callback for patching code
     */
    void set_code_patch_callback(CodePatchCallback callback);

    /**
     * @brief Get current statistics
     */
    APXHotPathStats get_stats() const;

    /**
     * @brief Reset statistics
     */
    void reset_stats();

    /**
     * @brief Get configuration
     */
    const APXHotPathConfig& get_config() const;

    /**
     * @brief Update configuration (some settings may require re-init)
     */
    void update_config(const APXHotPathConfig& config);

    /**
     * @brief Invalidate APX translation for a region
     *
     * Use when original code is modified.
     */
    void invalidate_region(uint64_t region_id);

    /**
     * @brief Invalidate all translations
     */
    void invalidate_all();

    /**
     * @brief Get detailed info about a region
     */
    const CodeRegion* get_region_info(uint64_t region_id) const;

    // =========================================================================
    // Integration with bpftime execution
    // =========================================================================

    /**
     * @brief Wrap an eBPF JIT function for APX optimization
     *
     * Takes a pointer to JIT-compiled eBPF code and returns an
     * APX-optimized version (or original if APX unavailable).
     *
     * @param jit_func Pointer to JIT-compiled function
     * @param func_size Size of JIT code
     * @param prog_id eBPF program ID for tracking
     * @return Pointer to execute (APX or original)
     */
    void* wrap_ebpf_jit(void* jit_func, size_t func_size, uint32_t prog_id);

    /**
     * @brief Execute eBPF program through APX-optimized path
     *
     * @param prog_id Program ID
     * @param mem Memory pointer (R1)
     * @param mem_len Memory length (R2)
     * @param returns Output return value
     * @return 0 on success
     */
    int execute_ebpf_apx(uint32_t prog_id, void* mem, size_t mem_len,
                        uint64_t* returns);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Global APX manager instance
 *
 * For convenience when integrating with existing code.
 */
APXHotPathManager& get_global_apx_manager();

/**
 * @brief Initialize global APX manager with config
 */
bool init_global_apx_manager(const APXHotPathConfig& config);

} // namespace bpftime::vm::apx

#endif // _BPFTIME_APX_HOTPATH_MANAGER_HPP
