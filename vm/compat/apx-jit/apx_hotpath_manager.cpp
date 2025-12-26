/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2024, eunomia-bpf org
 * All rights reserved.
 *
 * APX Hot Path Manager Implementation
 */

#include "apx_hotpath_manager.hpp"
#include "apx_cpu_features.hpp"
#include "lifter/rellume_lifter.hpp"

#include <spdlog/spdlog.h>

// Include LLVM Module for complete type definition
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <chrono>
#include <cstring>
#include <sys/mman.h>

namespace bpftime::vm::apx {

// ============================================================================
// Executable Code Cache with Trampoline Support
// ============================================================================

class ExecutableCodeCache {
public:
    explicit ExecutableCodeCache(size_t max_size) : max_size_(max_size) {
        region_ = mmap(nullptr, max_size,
                       PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) {
            region_ = nullptr;
            SPDLOG_ERROR("Failed to allocate executable code cache");
        }
    }

    ~ExecutableCodeCache() {
        if (region_) {
            munmap(region_, max_size_);
        }
    }

    void* allocate(size_t size, size_t alignment = 16) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Align offset
        size_t aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + size > max_size_) {
            SPDLOG_ERROR("Code cache exhausted");
            return nullptr;
        }

        void* ptr = static_cast<uint8_t*>(region_) + aligned_offset;
        current_offset_ = aligned_offset + size;

        return ptr;
    }

    void* copy_code(const uint8_t* code, size_t size) {
        void* ptr = allocate(size);
        if (ptr) {
            memcpy(ptr, code, size);
        }
        return ptr;
    }

    bool is_valid() const { return region_ != nullptr; }

private:
    void* region_ = nullptr;
    size_t max_size_;
    size_t current_offset_ = 0;
    std::mutex mutex_;
};

// ============================================================================
// XSAVE Manager for Selective State Save/Restore
// ============================================================================

class XSaveManager {
public:
    XSaveManager() {
        // Allocate per-thread XSAVE buffer
        buffer_ = aligned_alloc(64, BUFFER_SIZE);
        if (buffer_) {
            memset(buffer_, 0, BUFFER_SIZE);
        }
    }

    ~XSaveManager() {
        if (buffer_) {
            free(buffer_);
        }
    }

    void save_selective(uint64_t mask) {
        if (!buffer_ || mask == 0) return;

#if defined(__x86_64__) && defined(__GNUC__)
        uint32_t eax = static_cast<uint32_t>(mask);
        uint32_t edx = static_cast<uint32_t>(mask >> 32);

        asm volatile(
            "xsave64 %0"
            : "=m"(*static_cast<char*>(buffer_))
            : "a"(eax), "d"(edx)
            : "memory"
        );
#endif
    }

    void restore_selective(uint64_t mask) {
        if (!buffer_ || mask == 0) return;

#if defined(__x86_64__) && defined(__GNUC__)
        uint32_t eax = static_cast<uint32_t>(mask);
        uint32_t edx = static_cast<uint32_t>(mask >> 32);

        asm volatile(
            "xrstor64 %0"
            :
            : "m"(*static_cast<const char*>(buffer_)), "a"(eax), "d"(edx)
            : "memory"
        );
#endif
    }

private:
    static constexpr size_t BUFFER_SIZE = 8192;
    void* buffer_ = nullptr;
};

// Thread-local XSAVE manager
thread_local XSaveManager tls_xsave_manager;

// ============================================================================
// APX Optimization Pass
// ============================================================================

class APXOptimizer {
public:
    explicit APXOptimizer(const APXHotPathConfig& config)
        : config_(config), features_(get_cached_apx_features()) {}

    struct OptimizationResult {
        std::vector<uint8_t> code;
        uint32_t apx_regs_used;   // Bitmask of R16-R31
        uint64_t xsave_mask;
        size_t spills_avoided;
        size_t ndd_conversions;
        size_t nf_applications;
    };

    std::optional<OptimizationResult> optimize(const uint8_t* code,
                                                size_t code_size,
                                                uint64_t base_addr) {
        if (!features_.can_use_apx()) {
            SPDLOG_DEBUG("APX not available, skipping optimization");
            return std::nullopt;
        }

        OptimizationResult result;
        result.apx_regs_used = 0;
        result.xsave_mask = 0;
        result.spills_avoided = 0;
        result.ndd_conversions = 0;
        result.nf_applications = 0;

        // Create LLVM context and lift code
        auto context = std::make_unique<llvm::LLVMContext>();

        auto lift_result = lifter_.lift_buffer(code, code_size,
                                                base_addr, *context);
        if (!lift_result) {
            SPDLOG_WARN("Failed to lift code at {:#x}", base_addr);
            return std::nullopt;
        }

        // Apply APX optimizations to the LLVM module
        auto apx_regs = apply_apx_passes(*lift_result->module);

        // Generate APX-enabled machine code
        uint64_t xsave_mask;
        result.code = lifter_.generate_apx_code(*lift_result->module, xsave_mask);

        if (result.code.empty()) {
            SPDLOG_WARN("Failed to generate APX code");
            return std::nullopt;
        }

        // Compute which APX registers are used
        result.apx_regs_used = compute_apx_reg_mask(apx_regs);
        result.xsave_mask = compute_selective_xsave_mask(result.apx_regs_used);

        SPDLOG_DEBUG("APX optimization: {} bytes, {} regs, mask {:#x}",
                     result.code.size(), __builtin_popcount(result.apx_regs_used),
                     result.xsave_mask);

        return result;
    }

private:
    std::unordered_set<int> apply_apx_passes(llvm::Module& module) {
        // Apply register remapping
        auto apx_regs = lifter_.apply_apx_optimizations(module);

        // Additional optimization passes could go here:
        // - Dead code elimination
        // - Constant propagation
        // - Loop unrolling with APX register pressure in mind

        return apx_regs;
    }

    uint32_t compute_apx_reg_mask(const std::unordered_set<int>& regs) {
        uint32_t mask = 0;
        for (int reg : regs) {
            if (reg >= 16 && reg <= 31) {
                mask |= (1U << (reg - 16));
            }
        }
        return mask;
    }

    uint64_t compute_selective_xsave_mask(uint32_t apx_regs_used) {
        if (apx_regs_used == 0) {
            return 0;
        }

        // APX state component is bit 19 in XCR0
        // We only need to save if using R16-R31
        if (config_.selective_xsave) {
            return (1ULL << 19);  // APX state component only
        }

        // Full XSAVE including all components
        return features_.xcr0;
    }

    APXHotPathConfig config_;
    APXFeatures features_;
    RellumeLifter lifter_;
};

// ============================================================================
// Trampoline Generator
// ============================================================================

class TrampolineGenerator {
public:
    explicit TrampolineGenerator(ExecutableCodeCache& cache)
        : cache_(cache) {}

    /**
     * Generate a trampoline that:
     * 1. Saves caller-saved registers
     * 2. Optionally does selective XSAVE
     * 3. Calls the APX-optimized function
     * 4. Restores state
     * 5. Returns to caller
     */
    void* generate_trampoline(void* apx_code,
                              uint64_t xsave_mask,
                              void* original_code) {
        // Trampoline structure:
        //   push rbp
        //   mov rbp, rsp
        //   sub rsp, 32                    ; Align stack
        //   [xsave64 if needed]
        //   call apx_code
        //   [xrstor64 if needed]
        //   mov rsp, rbp
        //   pop rbp
        //   ret

        std::vector<uint8_t> trampoline;

        // Prologue
        trampoline.push_back(0x55);              // push rbp
        trampoline.push_back(0x48); trampoline.push_back(0x89);
        trampoline.push_back(0xe5);              // mov rbp, rsp
        trampoline.push_back(0x48); trampoline.push_back(0x83);
        trampoline.push_back(0xec); trampoline.push_back(0x20);  // sub rsp, 32

        // If XSAVE needed, we'll handle it differently - for now just call
        // (Full XSAVE integration would require more complex codegen)

        // Call APX code (relative or absolute)
        // For simplicity, use absolute call via register
        // mov rax, imm64
        trampoline.push_back(0x48); trampoline.push_back(0xb8);
        uint64_t addr = reinterpret_cast<uint64_t>(apx_code);
        for (int i = 0; i < 8; i++) {
            trampoline.push_back(static_cast<uint8_t>(addr >> (i * 8)));
        }
        // call rax
        trampoline.push_back(0xff); trampoline.push_back(0xd0);

        // Epilogue
        trampoline.push_back(0x48); trampoline.push_back(0x89);
        trampoline.push_back(0xec);              // mov rsp, rbp
        trampoline.push_back(0x5d);              // pop rbp
        trampoline.push_back(0xc3);              // ret

        // Copy to executable memory
        return cache_.copy_code(trampoline.data(), trampoline.size());
    }

    /**
     * Generate a hot patch (jump to trampoline)
     */
    std::vector<uint8_t> generate_hot_patch(void* trampoline) {
        std::vector<uint8_t> patch;

        // jmp [rip+0]
        // .quad trampoline_address
        // This is a 14-byte absolute jump

        patch.push_back(0xff); patch.push_back(0x25);  // jmp [rip+0]
        patch.push_back(0x00); patch.push_back(0x00);
        patch.push_back(0x00); patch.push_back(0x00);

        uint64_t addr = reinterpret_cast<uint64_t>(trampoline);
        for (int i = 0; i < 8; i++) {
            patch.push_back(static_cast<uint8_t>(addr >> (i * 8)));
        }

        return patch;
    }

private:
    ExecutableCodeCache& cache_;
};

// ============================================================================
// APXHotPathManager Implementation
// ============================================================================

// Internal stats with atomics for thread-safety
struct InternalStats {
    std::atomic<uint64_t> total_executions{0};
    std::atomic<uint64_t> hot_path_hits{0};
    std::atomic<uint64_t> cold_path_hits{0};
    std::atomic<uint64_t> translations{0};
    std::atomic<uint64_t> translation_failures{0};
    std::atomic<uint64_t> apx_optimizations_applied{0};
    std::atomic<uint64_t> registers_saved_by_apx{0};
    std::atomic<uint64_t> xsave_calls{0};
    std::atomic<uint64_t> xsave_bytes_saved{0};
    std::atomic<uint64_t> total_translation_time_ns{0};
    std::atomic<uint64_t> total_execution_time_ns{0};

    APXHotPathStats to_public() const {
        APXHotPathStats result;
        result.total_executions = total_executions.load();
        result.hot_path_hits = hot_path_hits.load();
        result.cold_path_hits = cold_path_hits.load();
        result.translations = translations.load();
        result.translation_failures = translation_failures.load();
        result.apx_optimizations_applied = apx_optimizations_applied.load();
        result.registers_saved_by_apx = registers_saved_by_apx.load();
        result.xsave_calls = xsave_calls.load();
        result.xsave_bytes_saved = xsave_bytes_saved.load();
        result.total_translation_time_ns = total_translation_time_ns.load();
        result.total_execution_time_ns = total_execution_time_ns.load();
        return result;
    }

    void reset() {
        total_executions = 0;
        hot_path_hits = 0;
        cold_path_hits = 0;
        translations = 0;
        translation_failures = 0;
        apx_optimizations_applied = 0;
        registers_saved_by_apx = 0;
        xsave_calls = 0;
        xsave_bytes_saved = 0;
        total_translation_time_ns = 0;
        total_execution_time_ns = 0;
    }
};

class APXHotPathManager::Impl {
public:
    APXHotPathConfig config;
    InternalStats stats;
    APXFeatures features;

    std::unique_ptr<ExecutableCodeCache> code_cache;
    std::unique_ptr<APXOptimizer> optimizer;
    std::unique_ptr<TrampolineGenerator> trampoline_gen;

    // Region management
    std::shared_mutex regions_mutex;
    std::unordered_map<uint64_t, std::unique_ptr<CodeRegion>> regions;
    uint64_t next_region_id = 1;

    // eBPF program tracking
    std::unordered_map<uint32_t, uint64_t> ebpf_prog_to_region;

    // Callbacks
    CodeReadCallback code_read_cb;
    CodePatchCallback code_patch_cb;

    bool initialized = false;

    Impl(const APXHotPathConfig& cfg) : config(cfg) {}

    bool initialize() {
        if (initialized) return true;

        features = get_cached_apx_features();

        if (!features.can_use_apx() && config.enable_apx) {
            SPDLOG_WARN("APX requested but not available on this CPU");
        }

        // Allocate code cache
        if (config.enable_code_cache) {
            code_cache = std::make_unique<ExecutableCodeCache>(config.code_cache_size);
            if (!code_cache->is_valid()) {
                SPDLOG_ERROR("Failed to initialize code cache");
                return false;
            }
        }

        // Initialize optimizer
        optimizer = std::make_unique<APXOptimizer>(config);

        // Initialize trampoline generator
        if (code_cache) {
            trampoline_gen = std::make_unique<TrampolineGenerator>(*code_cache);
        }

        initialized = true;
        SPDLOG_INFO("APX Hot Path Manager initialized (APX: {})",
                    features.can_use_apx() ? "available" : "not available");

        return true;
    }

    uint64_t register_region(uint64_t start_addr, size_t size,
                             const std::string& name) {
        std::unique_lock lock(regions_mutex);

        auto region = std::make_unique<CodeRegion>();
        region->start_addr = start_addr;
        region->end_addr = start_addr + size;
        region->size = size;
        region->state = CodeRegion::State::COLD;

        uint64_t id = next_region_id++;
        regions[id] = std::move(region);

        SPDLOG_DEBUG("Registered region {} at {:#x}, size {}", id, start_addr, size);

        return id;
    }

    void unregister_region(uint64_t region_id) {
        std::unique_lock lock(regions_mutex);
        regions.erase(region_id);
    }

    void* on_execute(uint64_t region_id) {
        stats.total_executions++;

        std::shared_lock lock(regions_mutex);
        auto it = regions.find(region_id);
        if (it == regions.end()) {
            return nullptr;
        }

        CodeRegion* region = it->second.get();
        region->execution_count++;

        auto state = region->state.load();

        switch (state) {
            case CodeRegion::State::HOT:
                stats.hot_path_hits++;
                return region->apx_code_ptr;

            case CodeRegion::State::COLD:
                if (region->execution_count >= config.hot_threshold) {
                    // Transition to profiling/translating
                    auto expected = CodeRegion::State::COLD;
                    if (region->state.compare_exchange_strong(
                            expected, CodeRegion::State::TRANSLATING)) {
                        // Trigger async translation
                        lock.unlock();
                        trigger_translation(region_id, region);
                    }
                }
                stats.cold_path_hits++;
                return reinterpret_cast<void*>(region->start_addr);

            case CodeRegion::State::PROFILING:
            case CodeRegion::State::TRANSLATING:
            case CodeRegion::State::FAILED:
            default:
                stats.cold_path_hits++;
                return reinterpret_cast<void*>(region->start_addr);
        }
    }

    void trigger_translation(uint64_t region_id, CodeRegion* region) {
        if (!code_read_cb) {
            SPDLOG_WARN("No code read callback set, cannot translate");
            region->state = CodeRegion::State::FAILED;
            return;
        }

        // Read original code
        std::vector<uint8_t> code(region->size);
        if (!code_read_cb(region->start_addr, code.data(), code.size())) {
            SPDLOG_ERROR("Failed to read code for region {}", region_id);
            region->state = CodeRegion::State::FAILED;
            stats.translation_failures++;
            return;
        }

        // Translate
        if (!do_translation(region, code.data(), code.size())) {
            region->state = CodeRegion::State::FAILED;
            stats.translation_failures++;
            return;
        }

        // Optionally patch original code
        if (code_patch_cb && region->apx_code_ptr) {
            apply_hot_patch(region);
        }

        region->state = CodeRegion::State::HOT;
        stats.translations++;

        SPDLOG_INFO("Region {} translated to APX, {} regs used",
                    region_id, __builtin_popcount(region->apx_regs_used));
    }

    bool do_translation(CodeRegion* region,
                        const uint8_t* code, size_t code_size) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Optimize with APX
        auto result = optimizer->optimize(code, code_size, region->start_addr);

        if (!result) {
            return false;
        }

        // Copy to executable memory
        void* exec_ptr = code_cache->copy_code(result->code.data(),
                                                result->code.size());
        if (!exec_ptr) {
            return false;
        }

        region->apx_code_ptr = exec_ptr;
        region->apx_code_size = result->code.size();
        region->xsave_mask = result->xsave_mask;
        region->apx_regs_used = result->apx_regs_used;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time).count();

        stats.total_translation_time_ns += duration;
        stats.apx_optimizations_applied++;
        stats.registers_saved_by_apx += result->spills_avoided;

        return true;
    }

    void apply_hot_patch(CodeRegion* region) {
        if (!trampoline_gen || !code_patch_cb) return;

        // Generate trampoline
        void* trampoline = trampoline_gen->generate_trampoline(
            region->apx_code_ptr,
            region->xsave_mask,
            reinterpret_cast<void*>(region->start_addr));

        if (!trampoline) {
            SPDLOG_WARN("Failed to generate trampoline");
            return;
        }

        // Generate hot patch (jump to trampoline)
        auto patch = trampoline_gen->generate_hot_patch(trampoline);

        // Save original prologue
        region->original_prologue.resize(patch.size());
        if (!code_read_cb(region->start_addr,
                          region->original_prologue.data(),
                          region->original_prologue.size())) {
            return;
        }

        // Apply patch
        if (code_patch_cb(region->start_addr, patch.data(), patch.size())) {
            region->is_patched = true;
            SPDLOG_DEBUG("Applied hot patch at {:#x}", region->start_addr);
        }
    }

    void invalidate_region(uint64_t region_id) {
        std::unique_lock lock(regions_mutex);
        auto it = regions.find(region_id);
        if (it == regions.end()) return;

        CodeRegion* region = it->second.get();

        // Restore original prologue if patched
        if (region->is_patched && code_patch_cb) {
            code_patch_cb(region->start_addr,
                          region->original_prologue.data(),
                          region->original_prologue.size());
            region->is_patched = false;
        }

        region->state = CodeRegion::State::COLD;
        region->apx_code_ptr = nullptr;
        region->execution_count = 0;
    }

    void* wrap_ebpf_jit(void* jit_func, size_t func_size, uint32_t prog_id) {
        if (!features.can_use_apx() || !config.enable_apx) {
            return jit_func;
        }

        // Register as a region
        uint64_t region_id = register_region(
            reinterpret_cast<uint64_t>(jit_func), func_size,
            "ebpf_prog_" + std::to_string(prog_id));

        ebpf_prog_to_region[prog_id] = region_id;

        // Set up memory callback to read from the JIT buffer
        auto jit_ptr = static_cast<const uint8_t*>(jit_func);
        auto jit_size = func_size;

        // Immediately translate (eBPF programs are usually small)
        std::shared_lock lock(regions_mutex);
        auto it = regions.find(region_id);
        if (it == regions.end()) return jit_func;

        CodeRegion* region = it->second.get();
        region->state = CodeRegion::State::TRANSLATING;
        lock.unlock();

        if (do_translation(region, jit_ptr, jit_size)) {
            region->state = CodeRegion::State::HOT;
            SPDLOG_INFO("eBPF prog {} APX-optimized: {} -> {} bytes",
                        prog_id, func_size, region->apx_code_size);
            return region->apx_code_ptr;
        }

        region->state = CodeRegion::State::FAILED;
        return jit_func;
    }

    int execute_ebpf_apx(uint32_t prog_id, void* mem, size_t mem_len,
                         uint64_t* returns) {
        auto it = ebpf_prog_to_region.find(prog_id);
        if (it == ebpf_prog_to_region.end()) {
            return -1;
        }

        uint64_t region_id = it->second;
        void* exec_ptr = on_execute(region_id);

        if (!exec_ptr) {
            return -1;
        }

        // Get region info for XSAVE mask
        std::shared_lock lock(regions_mutex);
        auto region_it = regions.find(region_id);
        if (region_it == regions.end()) {
            return -1;
        }

        CodeRegion* region = region_it->second.get();
        uint64_t xsave_mask = region->xsave_mask;
        lock.unlock();

        // Save APX state if needed
        if (xsave_mask != 0 && config.selective_xsave) {
            tls_xsave_manager.save_selective(xsave_mask);
            stats.xsave_calls++;
        }

        // Execute
        using EbpfFunc = uint64_t (*)(void*, size_t);
        auto func = reinterpret_cast<EbpfFunc>(exec_ptr);
        *returns = func(mem, mem_len);

        // Restore APX state
        if (xsave_mask != 0 && config.selective_xsave) {
            tls_xsave_manager.restore_selective(xsave_mask);
        }

        return 0;
    }
};

// ============================================================================
// APXHotPathManager Public Interface
// ============================================================================

APXHotPathManager::APXHotPathManager()
    : impl_(std::make_unique<Impl>(APXHotPathConfig{})) {}

APXHotPathManager::APXHotPathManager(const APXHotPathConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

APXHotPathManager::~APXHotPathManager() = default;

bool APXHotPathManager::initialize() {
    return impl_->initialize();
}

bool APXHotPathManager::is_apx_available() const {
    return impl_->features.can_use_apx();
}

uint64_t APXHotPathManager::register_region(uint64_t start_addr, size_t size,
                                            const std::string& name) {
    return impl_->register_region(start_addr, size, name);
}

void APXHotPathManager::unregister_region(uint64_t region_id) {
    impl_->unregister_region(region_id);
}

void* APXHotPathManager::on_execute(uint64_t region_id) {
    return impl_->on_execute(region_id);
}

bool APXHotPathManager::translate_region(uint64_t region_id,
                                         const uint8_t* code, size_t code_size) {
    std::shared_lock lock(impl_->regions_mutex);
    auto it = impl_->regions.find(region_id);
    if (it == impl_->regions.end()) return false;

    CodeRegion* region = it->second.get();
    lock.unlock();

    return impl_->do_translation(region, code, code_size);
}

void APXHotPathManager::set_code_read_callback(CodeReadCallback callback) {
    impl_->code_read_cb = std::move(callback);
}

void APXHotPathManager::set_code_patch_callback(CodePatchCallback callback) {
    impl_->code_patch_cb = std::move(callback);
}

APXHotPathStats APXHotPathManager::get_stats() const {
    return impl_->stats.to_public();
}

void APXHotPathManager::reset_stats() {
    impl_->stats.reset();
}

const APXHotPathConfig& APXHotPathManager::get_config() const {
    return impl_->config;
}

void APXHotPathManager::update_config(const APXHotPathConfig& config) {
    impl_->config = config;
}

void APXHotPathManager::invalidate_region(uint64_t region_id) {
    impl_->invalidate_region(region_id);
}

void APXHotPathManager::invalidate_all() {
    std::unique_lock lock(impl_->regions_mutex);
    for (auto& [id, region] : impl_->regions) {
        impl_->invalidate_region(id);
    }
}

const CodeRegion* APXHotPathManager::get_region_info(uint64_t region_id) const {
    std::shared_lock lock(impl_->regions_mutex);
    auto it = impl_->regions.find(region_id);
    if (it != impl_->regions.end()) {
        return it->second.get();
    }
    return nullptr;
}

void* APXHotPathManager::wrap_ebpf_jit(void* jit_func, size_t func_size,
                                        uint32_t prog_id) {
    return impl_->wrap_ebpf_jit(jit_func, func_size, prog_id);
}

int APXHotPathManager::execute_ebpf_apx(uint32_t prog_id, void* mem,
                                         size_t mem_len, uint64_t* returns) {
    return impl_->execute_ebpf_apx(prog_id, mem, mem_len, returns);
}

// ============================================================================
// Global Instance
// ============================================================================

static std::unique_ptr<APXHotPathManager> g_apx_manager;
static std::once_flag g_apx_manager_init;

APXHotPathManager& get_global_apx_manager() {
    std::call_once(g_apx_manager_init, []() {
        g_apx_manager = std::make_unique<APXHotPathManager>();
        g_apx_manager->initialize();
    });
    return *g_apx_manager;
}

bool init_global_apx_manager(const APXHotPathConfig& config) {
    if (g_apx_manager) {
        g_apx_manager->update_config(config);
        return true;
    }

    g_apx_manager = std::make_unique<APXHotPathManager>(config);
    return g_apx_manager->initialize();
}

} // namespace bpftime::vm::apx
