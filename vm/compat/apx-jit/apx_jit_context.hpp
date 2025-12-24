/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * APX-aware LLVM JIT Context
 *
 * This provides direct LLVM JIT compilation with APX features enabled.
 * It creates an APX-optimized TargetMachine for code generation.
 */

#ifndef _BPFTIME_APX_JIT_CONTEXT_HPP
#define _BPFTIME_APX_JIT_CONTEXT_HPP

#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <functional>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Target/TargetMachine.h>

#include "apx_cpu_features.hpp"

namespace bpftime::vm::apx {

// Forward declaration
class apx_llvmbpf_vm;

/**
 * @brief APX-aware LLVM JIT Context
 *
 * This class manages LLVM JIT compilation with APX features enabled.
 * Key differences from standard LLVM JIT:
 *
 * 1. Creates TargetMachine with APX features (+egpr, +ndd, +nf, etc.)
 * 2. Uses register allocator hints favoring R16-R31 for temporaries
 * 3. Enables 3-operand form generation for shorter dependency chains
 * 4. Uses flag suppression where flags are not needed
 */
class apx_jit_context {
public:
    /**
     * @brief Construct APX JIT context
     * @param vm Reference to the owning VM for accessing code/helpers
     * @param features APX feature configuration
     */
    apx_jit_context(apx_llvmbpf_vm& vm, const APXFeatures& features);

    ~apx_jit_context();

    /**
     * @brief Perform JIT compilation with APX optimizations
     * @return Error on failure
     */
    llvm::Error do_jit_compile();

    /**
     * @brief Perform AOT compilation with APX optimizations
     * @param print_ir If true, print LLVM IR to stdout
     * @return Compiled object file bytes
     */
    std::vector<uint8_t> do_aot_compile(bool print_ir = false);

    /**
     * @brief Load pre-compiled AOT object
     * @param buf Object file bytes
     * @return Error on failure
     */
    llvm::Error load_aot_object(const std::vector<uint8_t>& buf);

    /**
     * @brief Get address of the compiled bpf_main function
     * @return Function pointer to entry point
     */
    uint64_t (*get_entry_address())(void* mem, size_t mem_len);

    /**
     * @brief Check if APX optimizations are enabled
     */
    bool is_apx_enabled() const { return apx_enabled_; }

private:
    apx_llvmbpf_vm& vm_;
    APXFeatures features_;
    bool apx_enabled_;

    std::string cpu_target_;
    std::string feature_string_;

    std::optional<std::unique_ptr<llvm::orc::LLJIT>> jit_;
    std::unique_ptr<llvm::TargetMachine> target_machine_;

    /**
     * @brief Create APX-configured target machine
     */
    std::unique_ptr<llvm::TargetMachine> create_apx_target_machine();

    /**
     * @brief Create and initialize LLJIT instance with APX configuration
     */
    std::tuple<std::unique_ptr<llvm::orc::LLJIT>,
               std::vector<std::string>,
               std::vector<std::string>>
    create_apx_lljit();

    /**
     * @brief Generate LLVM module from eBPF bytecode
     */
    llvm::Expected<llvm::orc::ThreadSafeModule>
    generate_module(const std::vector<std::string>& ext_func_names,
                   const std::vector<std::string>& lddw_helpers,
                   bool patch_map_val);
};

} // namespace bpftime::vm::apx

#endif // _BPFTIME_APX_JIT_CONTEXT_HPP
