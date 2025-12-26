/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * APX-aware LLVM JIT backend implementation
 *
 * This integrates APX optimization into bpftime's JIT pipeline:
 * 1. eBPF bytecode is compiled by LLVM JIT
 * 2. JIT output is lifted back to LLVM IR (via Rellume)
 * 3. APX optimizations are applied (R16-R31, NDD, NF)
 * 4. APX-enabled code is emitted and cached
 * 5. Hot paths are transparently routed to APX version
 */

#include "compat_apx_llvm.hpp"
#include "apx_cpu_features.hpp"
#include "apx_hotpath_manager.hpp"
#include <spdlog/spdlog.h>
#include <bpftime_vm_compat.hpp>

// Include LLVM headers for target machine configuration
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Config/llvm-config.h>
#if LLVM_VERSION_MAJOR >= 16
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#else
#include <llvm/Support/Host.h>
#include <llvm/MC/SubtargetFeature.h>
#endif

namespace bpftime::vm::apx {

namespace {

// Thread-local APX feature configuration for LLVM
thread_local std::string tls_apx_features;
thread_local std::string tls_apx_cpu;

// Program ID counter for hot path tracking
std::atomic<uint32_t> g_program_id_counter{1};

} // anonymous namespace

// ============================================================================
// APX Hot Path Integration
// ============================================================================

/**
 * @brief Initialize APX hot path manager with default config
 */
static APXHotPathManager& get_apx_manager() {
    static APXHotPathConfig config = []() {
        APXHotPathConfig c;
        c.hot_threshold = 100;          // Lower threshold for eBPF
        c.enable_apx = true;
        c.enable_ndd = true;
        c.enable_nf = true;
        c.selective_xsave = true;
        c.verbose_logging = false;
        return c;
    }();
    static APXHotPathManager manager(config);
    static std::once_flag init_flag;
    std::call_once(init_flag, [&]() {
        manager.initialize();
    });
    return manager;
}

bpftime_apx_llvm_vm::bpftime_apx_llvm_vm() {
    initialize_apx_backend();
}

void bpftime_apx_llvm_vm::initialize_apx_backend() {
    // Detect APX features
    apx_features_ = detect_apx_features();

    // Check if APX should be enabled
    bool apx_enabled = is_apx_enabled_by_env();

    if (apx_enabled && apx_features_.can_use_apx()) {
        apx_active_ = true;
        llvm_features_ = get_llvm_apx_features(apx_features_);
        llvm_cpu_target_ = get_llvm_cpu_target(apx_features_);

        SPDLOG_INFO("APX JIT: APX optimizations ENABLED");
        SPDLOG_INFO("APX JIT: CPU target: {}", llvm_cpu_target_);
        SPDLOG_INFO("APX JIT: Features: {}", llvm_features_);

        // Warm up APX registers
        warmup_apx_registers();
    } else {
        apx_active_ = false;
        llvm_features_ = "";
        llvm_cpu_target_ = "generic";

        if (!apx_enabled) {
            SPDLOG_INFO("APX JIT: APX disabled by environment");
        } else {
            SPDLOG_INFO("APX JIT: APX not available on this CPU");
        }
    }

    // Set thread-local configuration for use by customized LLVM code
    tls_apx_features = llvm_features_;
    tls_apx_cpu = llvm_cpu_target_;

    // Create the underlying LLVM VM
    // We try to create the standard LLVM VM and configure it with APX settings
    try {
        inner_vm_ = compat::create_vm_instance("llvm");
        SPDLOG_DEBUG("APX JIT: Created underlying LLVM VM instance");
    } catch (const std::exception& e) {
        error_msg_ = std::string("Failed to create LLVM VM: ") + e.what();
        SPDLOG_ERROR("APX JIT: {}", error_msg_);
        inner_vm_ = nullptr;
    }
}

std::string bpftime_apx_llvm_vm::get_error_message() {
    if (!error_msg_.empty()) {
        return error_msg_;
    }
    if (inner_vm_) {
        return inner_vm_->get_error_message();
    }
    return "APX LLVM VM not initialized";
}

int bpftime_apx_llvm_vm::load_code(const void *code, size_t code_len) {
    if (!inner_vm_) {
        error_msg_ = "APX LLVM VM not initialized";
        return -1;
    }

    // Set thread-local APX configuration before loading
    tls_apx_features = llvm_features_;
    tls_apx_cpu = llvm_cpu_target_;

    return inner_vm_->load_code(code, code_len);
}

int bpftime_apx_llvm_vm::register_external_function(size_t index,
                                                     const std::string &name,
                                                     void *fn) {
    if (!inner_vm_) {
        error_msg_ = "APX LLVM VM not initialized";
        return -1;
    }
    return inner_vm_->register_external_function(index, name, fn);
}

void bpftime_apx_llvm_vm::unload_code() {
    if (inner_vm_) {
        inner_vm_->unload_code();
    }
}

int bpftime_apx_llvm_vm::exec(void *mem, size_t mem_len,
                               uint64_t &bpf_return_value) {
    if (!inner_vm_) {
        error_msg_ = "APX LLVM VM not initialized";
        return -1;
    }

    // Set thread-local configuration in case of lazy compilation
    tls_apx_features = llvm_features_;
    tls_apx_cpu = llvm_cpu_target_;

    return inner_vm_->exec(mem, mem_len, bpf_return_value);
}

std::vector<uint8_t> bpftime_apx_llvm_vm::do_aot_compile(bool print_ir) {
    if (!inner_vm_) {
        error_msg_ = "APX LLVM VM not initialized";
        return {};
    }

    // Set thread-local configuration for AOT compilation
    tls_apx_features = llvm_features_;
    tls_apx_cpu = llvm_cpu_target_;

    auto result = inner_vm_->do_aot_compile(print_ir);

    if (apx_active_) {
        SPDLOG_INFO("APX JIT: AOT compilation completed with APX optimizations");
    }

    return result;
}

std::optional<compat::precompiled_ebpf_function>
bpftime_apx_llvm_vm::load_aot_object(const std::vector<uint8_t> &object) {
    if (!inner_vm_) {
        error_msg_ = "APX LLVM VM not initialized";
        return std::nullopt;
    }

    // Set thread-local configuration
    tls_apx_features = llvm_features_;
    tls_apx_cpu = llvm_cpu_target_;

    return inner_vm_->load_aot_object(object);
}

std::optional<compat::precompiled_ebpf_function> bpftime_apx_llvm_vm::compile() {
    if (!inner_vm_) {
        error_msg_ = "APX LLVM VM not initialized";
        return std::nullopt;
    }

    // Set thread-local configuration for JIT compilation
    tls_apx_features = llvm_features_;
    tls_apx_cpu = llvm_cpu_target_;

    auto result = inner_vm_->compile();

    if (!result.has_value()) {
        return std::nullopt;
    }

    // If APX is active, wrap the JIT function with APX hot path manager
    if (apx_active_ && apx_features_.can_use_apx()) {
        auto& manager = get_apx_manager();

        // Get function pointer from result (precompiled_ebpf_function is a function pointer type)
        auto func_ptr = *result;

        if (func_ptr) {
            // Assign a unique program ID
            uint32_t prog_id = g_program_id_counter++;

            // Register with hot path manager for potential APX re-optimization
            // The manager will lift the x86-64 JIT output back to IR,
            // apply APX optimizations, and cache the result
            size_t func_size = 4096;  // Estimate; could be improved with symbol info

            void* apx_func = manager.wrap_ebpf_jit(
                reinterpret_cast<void*>(func_ptr), func_size, prog_id);

            if (apx_func && apx_func != reinterpret_cast<void*>(func_ptr)) {
                // Successfully created APX-optimized version
                // Update the result to use APX function
                result = reinterpret_cast<compat::precompiled_ebpf_function>(apx_func);

                SPDLOG_INFO("APX JIT: eBPF program {} wrapped with APX optimization", prog_id);

                // Store program ID for later execution tracking
                program_id_ = prog_id;
            } else {
                SPDLOG_DEBUG("APX JIT: Using standard JIT output (APX lift failed or unavailable)");
            }
        }
    } else if (apx_active_) {
        SPDLOG_DEBUG("APX JIT: JIT compilation completed with APX target features");
    }

    return result;
}

void bpftime_apx_llvm_vm::set_lddw_helpers(uint64_t (*map_by_fd)(uint32_t),
                                           uint64_t (*map_by_idx)(uint32_t),
                                           uint64_t (*map_val)(uint64_t),
                                           uint64_t (*var_addr)(uint32_t),
                                           uint64_t (*code_addr)(uint32_t)) {
    if (inner_vm_) {
        inner_vm_->set_lddw_helpers(map_by_fd, map_by_idx, map_val,
                                    var_addr, code_addr);
    }
}

std::optional<std::string> bpftime_apx_llvm_vm::generate_ptx(const char *target_cpu) {
    if (!inner_vm_) {
        error_msg_ = "APX LLVM VM not initialized";
        return std::nullopt;
    }

    // PTX generation doesn't use APX features (it's for NVIDIA GPUs)
    return inner_vm_->generate_ptx(target_cpu);
}

std::unique_ptr<compat::bpftime_vm_impl> create_apx_llvm_vm_instance() {
    SPDLOG_DEBUG("Creating APX-aware LLVM VM instance");
    return std::make_unique<bpftime_apx_llvm_vm>();
}

// Accessor functions for thread-local APX configuration
// These can be used by modified llvmbpf code to get APX settings
extern "C" {

const char* bpftime_get_apx_cpu_target() {
    return tls_apx_cpu.empty() ? "generic" : tls_apx_cpu.c_str();
}

const char* bpftime_get_apx_features() {
    return tls_apx_features.c_str();
}

bool bpftime_is_apx_active() {
    return !tls_apx_features.empty();
}

} // extern "C"

} // namespace bpftime::vm::apx

// Factory registration
namespace bpftime::vm::compat {
namespace apx_registration {

__attribute__((constructor(0))) void register_apx_llvm_vm_factory() {
    register_vm_factory("apx_llvm", bpftime::vm::apx::create_apx_llvm_vm_instance);
    SPDLOG_DEBUG("APX LLVM: Registered VM factory 'apx_llvm'");
}

} // namespace apx_registration
} // namespace bpftime::vm::compat
