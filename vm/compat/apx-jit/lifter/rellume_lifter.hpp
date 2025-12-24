/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * Rellume-based x86-64 to LLVM IR Lifter for APX DBT
 *
 * This module integrates the Rellume library (https://github.com/aengelke/rellume)
 * for lifting x86-64 machine code to LLVM IR, enabling APX regeneration.
 */

#ifndef _BPFTIME_RELLUME_LIFTER_HPP
#define _BPFTIME_RELLUME_LIFTER_HPP

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <functional>
#include <unordered_set>

// Forward declarations
namespace llvm {
class Module;
class Function;
class LLVMContext;
}

namespace bpftime::vm::apx {

/**
 * @brief CPU state structure for Rellume
 *
 * Rellume operates on a virtual CPU state structure.
 * This mirrors the Rellume cpu_state but extends for APX R16-R31.
 */
struct alignas(64) APXCpuState {
    // General purpose registers (standard x86-64)
    uint64_t gpr[16];  // RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8-R15

    // APX extended GPRs
    uint64_t egpr[16]; // R16-R31

    // Flags
    uint64_t rflags;

    // Instruction pointer
    uint64_t rip;

    // Segment registers
    uint16_t cs, ss, ds, es, fs, gs;
    uint64_t fs_base, gs_base;

    // SSE/AVX state
    alignas(64) uint8_t xmm[32][64];  // XMM0-31 (512-bit for AVX-512)
    uint32_t mxcsr;

    // x87 FPU state (simplified)
    uint64_t x87_regs[8];
    uint16_t fpcw, fpsw, fptw;
};

/**
 * @brief Rellume configuration for APX-aware lifting
 */
struct RellumeConfig {
    bool enable_apx = true;           // Use APX registers in output
    bool enable_avx512 = false;       // Enable AVX-512 lifting
    bool optimize_flags = true;       // Dead flag elimination
    bool enable_ndd = true;           // Use 3-operand forms
    bool enable_nf = true;            // Use flag suppression
    uint64_t code_base = 0;           // Base address for code
    size_t max_instructions = 10000;  // Max instructions to lift
};

/**
 * @brief Result of APX-aware lifting
 */
struct APXLiftResult {
    std::unique_ptr<llvm::Module> module;
    llvm::Function* entry_function;

    // APX usage info
    std::unordered_set<int> apx_regs_used;  // Which R16-R31 used
    uint64_t xsave_mask;                     // XSAVE mask for state

    // Statistics
    size_t original_instructions;
    size_t lifted_instructions;
    size_t apx_optimizations;  // Number of APX-specific opts applied
};

/**
 * @brief Memory read callback for Rellume
 *
 * Called when Rellume needs to read instruction bytes.
 *
 * @param addr Address to read from
 * @param buf Buffer to read into
 * @param len Number of bytes to read
 * @return true on success
 */
using MemoryReadCallback = std::function<bool(uint64_t addr, uint8_t* buf, size_t len)>;

/**
 * @brief Symbol resolution callback
 *
 * Called to resolve external symbols/calls.
 *
 * @param addr Call target address
 * @return Symbol name, or empty string if unknown
 */
using SymbolCallback = std::function<std::string(uint64_t addr)>;

/**
 * @brief Rellume-based x86-64 to LLVM IR lifter with APX support
 *
 * This class wraps Rellume to provide x86-64 → LLVM IR lifting,
 * then applies APX-specific optimizations:
 *
 * 1. Register remapping to use R16-R31 for temporaries
 * 2. 3-operand form generation (NDD)
 * 3. Flag suppression (NF) when flags unused
 * 4. Selective XSAVE for modified extended registers
 */
class RellumeLifter {
public:
    RellumeLifter();
    ~RellumeLifter();

    /**
     * @brief Configure the lifter
     *
     * @param config Configuration options
     */
    void configure(const RellumeConfig& config);

    /**
     * @brief Set memory read callback
     *
     * @param callback Function to read instruction bytes
     */
    void set_memory_callback(MemoryReadCallback callback);

    /**
     * @brief Set symbol resolution callback
     *
     * @param callback Function to resolve symbols
     */
    void set_symbol_callback(SymbolCallback callback);

    /**
     * @brief Lift x86-64 code starting at address
     *
     * Follows control flow to build complete function/trace.
     *
     * @param entry_addr Entry point address
     * @param context LLVM context
     * @return Lifted LLVM module with APX optimizations
     */
    std::optional<APXLiftResult> lift(uint64_t entry_addr,
                                      llvm::LLVMContext& context);

    /**
     * @brief Lift x86-64 code from buffer
     *
     * @param code Machine code bytes
     * @param code_size Size in bytes
     * @param base_addr Virtual base address
     * @param context LLVM context
     * @return Lifted LLVM module with APX optimizations
     */
    std::optional<APXLiftResult> lift_buffer(const uint8_t* code,
                                             size_t code_size,
                                             uint64_t base_addr,
                                             llvm::LLVMContext& context);

    /**
     * @brief Apply APX optimizations to lifted module
     *
     * Post-processes Rellume output to maximize APX benefits:
     * - Remap temporaries to R16-R31
     * - Convert 2-op → 3-op forms
     * - Add flag suppression
     *
     * @param module Module to optimize (modified in place)
     * @return Set of APX registers used
     */
    std::unordered_set<int> apply_apx_optimizations(llvm::Module& module);

    /**
     * @brief Generate APX native code from LLVM module
     *
     * Compiles LLVM IR to x86-64 with APX features enabled.
     *
     * @param module LLVM module
     * @param xsave_mask Output: XSAVE mask for used APX features
     * @return Machine code bytes
     */
    std::vector<uint8_t> generate_apx_code(llvm::Module& module,
                                           uint64_t& xsave_mask);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief APX DBT (Dynamic Binary Translator)
 *
 * High-level interface that combines:
 * 1. Rellume lifting (x86-64 → LLVM IR)
 * 2. APX optimization passes
 * 3. Code generation with APX features
 * 4. Selective XSAVE management
 * 5. Code caching
 */
class APXDBT {
public:
    APXDBT();
    ~APXDBT();

    /**
     * @brief Translate and cache a code region
     *
     * @param original_addr Original x86-64 code address
     * @param code Machine code bytes
     * @param code_size Size in bytes
     * @return Pointer to APX-optimized code
     */
    void* translate(uint64_t original_addr,
                   const uint8_t* code,
                   size_t code_size);

    /**
     * @brief Execute APX-translated code
     *
     * Manages XSAVE/XRSTOR for extended register state.
     *
     * @param original_addr Original address (for cache lookup)
     * @param cpu_state CPU state to use
     * @return Execution result (e.g., next RIP)
     */
    uint64_t execute(uint64_t original_addr, APXCpuState& cpu_state);

    /**
     * @brief Invalidate translation for address
     *
     * @param original_addr Address to invalidate
     */
    void invalidate(uint64_t original_addr);

    /**
     * @brief Check if APX is available and enabled
     */
    bool is_apx_enabled() const;

    /**
     * @brief Get statistics
     */
    struct Stats {
        uint64_t translations;
        uint64_t cache_hits;
        uint64_t cache_misses;
        uint64_t apx_optimizations;
        uint64_t xsave_calls;
    };
    Stats get_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bpftime::vm::apx

#endif // _BPFTIME_RELLUME_LIFTER_HPP
