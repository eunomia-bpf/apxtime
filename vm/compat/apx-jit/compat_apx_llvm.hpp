/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * APX-aware LLVM JIT backend for bpftime
 *
 * This backend extends the standard LLVM JIT to leverage Intel APX
 * (Advanced Performance Extensions) when available. APX provides:
 * - 16 additional GPRs (R16-R31) reducing register spills
 * - 3-operand forms (NDD) shortening dependency chains
 * - Flag suppression (NF) reducing EFLAGS pressure
 * - PUSH2/POP2 for faster stack operations
 */

#ifndef _BPFTIME_VM_COMPAT_APX_LLVM_HPP
#define _BPFTIME_VM_COMPAT_APX_LLVM_HPP

#include <bpftime_vm_compat.hpp>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include "apx_cpu_features.hpp"

namespace bpftime::vm::apx {

/**
 * @brief APX-aware LLVM JIT VM implementation
 *
 * This class wraps the standard LLVM JIT to provide APX-optimized code
 * generation when running on APX-capable hardware. It:
 *
 * 1. Detects APX capabilities at initialization
 * 2. Configures LLVM TargetMachine with APX features
 * 3. Uses extended GPRs to reduce spills
 * 4. Applies 3-operand forms where beneficial
 * 5. Falls back to standard x86-64 on non-APX hardware
 */
class bpftime_apx_llvm_vm : public compat::bpftime_vm_impl {
public:
    /**
     * @brief Construct APX-aware LLVM VM
     *
     * Detects APX capabilities and configures the backend accordingly.
     * If APX is not available or disabled, falls back to standard LLVM JIT.
     */
    bpftime_apx_llvm_vm();

    virtual ~bpftime_apx_llvm_vm() = default;

    // bpftime_vm_impl interface implementations
    std::string get_error_message() override;
    int load_code(const void *code, size_t code_len) override;
    int register_external_function(size_t index, const std::string &name,
                                   void *fn) override;
    void unload_code() override;
    int exec(void *mem, size_t mem_len, uint64_t &bpf_return_value) override;
    std::vector<uint8_t> do_aot_compile(bool print_ir = false) override;
    std::optional<compat::precompiled_ebpf_function>
    load_aot_object(const std::vector<uint8_t> &object) override;
    std::optional<compat::precompiled_ebpf_function> compile() override;
    void set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
                          uint64_t (*map_by_idx)(uint32_t),
                          uint64_t (*map_val)(uint64_t),
                          uint64_t (*var_addr)(uint32_t),
                          uint64_t (*code_addr)(uint32_t)) override;
    std::optional<std::string> generate_ptx(const char *target_cpu) override;

    /**
     * @brief Check if APX optimizations are active
     * @return true if APX features are being used
     */
    bool is_apx_active() const { return apx_active_; }

    /**
     * @brief Get the detected APX features
     * @return Reference to APX feature detection results
     */
    const APXFeatures& get_apx_features() const { return apx_features_; }

    /**
     * @brief Get the LLVM feature string being used
     * @return LLVM feature string with APX features
     */
    const std::string& get_llvm_features() const { return llvm_features_; }

private:
    // APX feature detection results
    APXFeatures apx_features_;

    // Whether APX optimizations are active
    bool apx_active_ = false;

    // LLVM feature string for target machine
    std::string llvm_features_;

    // LLVM CPU target name
    std::string llvm_cpu_target_;

    // Underlying VM implementation (standard LLVM VM with APX config)
    std::unique_ptr<compat::bpftime_vm_impl> inner_vm_;

    // Error message storage
    std::string error_msg_;

    // Program ID for hot path tracking (assigned after successful APX wrap)
    uint32_t program_id_ = 0;

    /**
     * @brief Initialize the APX-configured LLVM backend
     */
    void initialize_apx_backend();

public:
    /**
     * @brief Get the program ID for hot path tracking
     * @return Program ID (0 if not APX-wrapped)
     */
    uint32_t get_program_id() const { return program_id_; }
};

/**
 * @brief Factory function for creating APX-aware LLVM VM instances
 * @return Unique pointer to new APX LLVM VM instance
 */
std::unique_ptr<compat::bpftime_vm_impl> create_apx_llvm_vm_instance();

} // namespace bpftime::vm::apx

#endif // _BPFTIME_VM_COMPAT_APX_LLVM_HPP
