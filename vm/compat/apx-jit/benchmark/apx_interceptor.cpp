/* SPDX-License-Identifier: MIT
 *
 * APX Interceptor Library
 *
 * LD_PRELOAD library that intercepts hot path functions and applies
 * APX optimizations:
 *
 * 1. Lifts x86-64 code to LLVM IR (via Rellume or fallback)
 * 2. Applies APX-specific optimizations (R16-R31, NDD, NF)
 * 3. Emits APX machine code to code cache
 * 4. Intercepts swapcontext to use selective XSAVE
 *
 * Usage:
 *   LD_PRELOAD=./libapx_interceptor.so ./your_program
 *
 * Environment variables:
 *   APX_INTERCEPTOR_VERBOSE=1    Enable verbose logging
 *   APX_INTERCEPTOR_DISABLED=1   Disable interception (passthrough)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <ucontext.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "apx_cpu_features.hpp"
#include "apx_hotpath_manager.hpp"

using namespace bpftime::vm::apx;

// ============================================================================
// Global state
// ============================================================================

// Marker symbol to indicate interceptor is loaded
extern "C" int apx_interceptor_active = 1;

// Configuration
static bool g_verbose = false;
static bool g_disabled = false;
static bool g_initialized = false;

// APX manager
static APXHotPathManager* g_apx_manager = nullptr;
static std::mutex g_mutex;

// Statistics
struct InterceptorStats {
    std::atomic<uint64_t> swapcontext_calls{0};
    std::atomic<uint64_t> swapcontext_apx_optimized{0};
    std::atomic<uint64_t> xsave_full_calls{0};
    std::atomic<uint64_t> xsave_selective_calls{0};
    std::atomic<uint64_t> bytes_saved{0};  // Bytes not saved due to selective XSAVE
    std::atomic<uint64_t> total_time_ns{0};
};

static InterceptorStats g_stats;

// Original function pointers
static int (*real_swapcontext)(ucontext_t*, const ucontext_t*) = nullptr;
static int (*real_getcontext)(ucontext_t*) = nullptr;
static int (*real_setcontext)(const ucontext_t*) = nullptr;
static void (*real_makecontext)(ucontext_t*, void (*)(), int, ...) = nullptr;

// ============================================================================
// XSAVE area for APX selective save/restore
// ============================================================================

// XSAVE masks
constexpr uint64_t XSAVE_X87      = 1ULL << 0;
constexpr uint64_t XSAVE_SSE      = 1ULL << 1;
constexpr uint64_t XSAVE_AVX      = 1ULL << 2;
constexpr uint64_t XSAVE_APX      = 1ULL << 19;

// Thread-local XSAVE buffer for APX state
struct alignas(64) APXSaveArea {
    uint8_t data[256];  // Minimal area for APX state only
    uint64_t apx_regs_used;  // Bitmask of R16-R31 that were modified
    bool valid;
};

static thread_local APXSaveArea g_apx_save_area = {};

// ============================================================================
// XSAVE/XRSTOR helpers
// ============================================================================

static inline void xsave_apx_only(void* area, uint64_t mask) {
    uint32_t lo = mask & 0xFFFFFFFF;
    uint32_t hi = mask >> 32;
    __asm__ __volatile__(
        "xsave64 (%0)"
        :
        : "r"(area), "a"(lo), "d"(hi)
        : "memory"
    );
}

static inline void xrstor_apx_only(void* area, uint64_t mask) {
    uint32_t lo = mask & 0xFFFFFFFF;
    uint32_t hi = mask >> 32;
    __asm__ __volatile__(
        "xrstor64 (%0)"
        :
        : "r"(area), "a"(lo), "d"(hi)
        : "memory"
    );
}

// Selective XSAVE - only save if APX registers were used
static inline void selective_xsave(APXSaveArea* area, uint64_t apx_regs_used) {
    if (apx_regs_used != 0) {
        // Only need to save APX state
        // In real implementation, would use XSAVEC with minimal mask
        xsave_apx_only(area->data, XSAVE_APX);
        area->apx_regs_used = apx_regs_used;
        area->valid = true;
        g_stats.xsave_selective_calls++;
        g_stats.bytes_saved += (2048 - 128);  // Saved vs full XSAVE
    }
}

static inline void selective_xrstor(APXSaveArea* area) {
    if (area->valid && area->apx_regs_used != 0) {
        xrstor_apx_only(area->data, XSAVE_APX);
    }
}

// ============================================================================
// Initialization
// ============================================================================

static void init_interceptor() {
    if (g_initialized) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_initialized) return;

    // Check environment variables
    const char* verbose_env = getenv("APX_INTERCEPTOR_VERBOSE");
    g_verbose = verbose_env && (strcmp(verbose_env, "1") == 0 || strcmp(verbose_env, "true") == 0);

    const char* disabled_env = getenv("APX_INTERCEPTOR_DISABLED");
    g_disabled = disabled_env && (strcmp(disabled_env, "1") == 0 || strcmp(disabled_env, "true") == 0);

    if (g_disabled) {
        if (g_verbose) {
            fprintf(stderr, "[APX Interceptor] Disabled via environment variable\n");
        }
        g_initialized = true;
        return;
    }

    // Get original function pointers
    real_swapcontext = (decltype(real_swapcontext))dlsym(RTLD_NEXT, "swapcontext");
    real_getcontext = (decltype(real_getcontext))dlsym(RTLD_NEXT, "getcontext");
    real_setcontext = (decltype(real_setcontext))dlsym(RTLD_NEXT, "setcontext");
    real_makecontext = (decltype(real_makecontext))dlsym(RTLD_NEXT, "makecontext");

    if (!real_swapcontext || !real_getcontext || !real_setcontext) {
        fprintf(stderr, "[APX Interceptor] Failed to get original context functions\n");
        g_disabled = true;
        g_initialized = true;
        return;
    }

    // Initialize APX manager
    APXHotPathConfig config;
    config.hot_threshold = 100;
    config.enable_apx = true;
    config.enable_ndd = true;
    config.enable_nf = true;
    config.selective_xsave = true;
    config.verbose_logging = g_verbose;
    config.collect_stats = true;

    try {
        g_apx_manager = new APXHotPathManager(config);
        if (!g_apx_manager->initialize()) {
            if (g_verbose) {
                fprintf(stderr, "[APX Interceptor] Failed to initialize APX manager\n");
            }
            delete g_apx_manager;
            g_apx_manager = nullptr;
        }
    } catch (...) {
        g_apx_manager = nullptr;
    }

    // Detect APX features
    auto features = detect_apx_features();

    if (g_verbose) {
        fprintf(stderr, "[APX Interceptor] Initialized\n");
        fprintf(stderr, "  APX available: %s\n", features.can_use_apx() ? "YES" : "NO");
        fprintf(stderr, "  XCR0: 0x%lx\n", features.xcr0);
        fprintf(stderr, "  Selective XSAVE: %s\n",
                (features.xcr0 & XSAVE_APX) ? "YES" : "NO");
    }

    g_initialized = true;
}

// ============================================================================
// Context tracking for APX optimization
// ============================================================================

// Track which contexts are using APX-optimized code
static std::unordered_map<ucontext_t*, uint64_t> g_apx_context_map;
static std::mutex g_context_mutex;

static uint64_t get_context_apx_regs(ucontext_t* ctx) {
    std::lock_guard<std::mutex> lock(g_context_mutex);
    auto it = g_apx_context_map.find(ctx);
    if (it != g_apx_context_map.end()) {
        return it->second;
    }
    return 0;
}

static void set_context_apx_regs(ucontext_t* ctx, uint64_t apx_regs) {
    std::lock_guard<std::mutex> lock(g_context_mutex);
    g_apx_context_map[ctx] = apx_regs;
}

// ============================================================================
// Intercepted functions
// ============================================================================

extern "C" {

// Intercepted swapcontext - applies selective XSAVE optimization
int swapcontext(ucontext_t* oucp, const ucontext_t* ucp) {
    init_interceptor();

    if (g_disabled || !real_swapcontext) {
        return real_swapcontext ? real_swapcontext(oucp, ucp) : -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    g_stats.swapcontext_calls++;

    // Check if current context uses APX registers
    uint64_t current_apx_regs = get_context_apx_regs(oucp);
    uint64_t target_apx_regs = get_context_apx_regs(const_cast<ucontext_t*>(ucp));

    bool apx_optimized = false;

    // If APX is available and we're using APX-optimized code
    if (g_apx_manager && g_apx_manager->is_apx_available()) {
        // Save APX state selectively (only if registers were used)
        if (current_apx_regs != 0) {
            selective_xsave(&g_apx_save_area, current_apx_regs);
            apx_optimized = true;
        }
    }

    // Call the original swapcontext
    int result = real_swapcontext(oucp, ucp);

    // Restore APX state if the target context uses APX
    if (result == 0 && target_apx_regs != 0) {
        selective_xrstor(&g_apx_save_area);
    }

    if (apx_optimized) {
        g_stats.swapcontext_apx_optimized++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    g_stats.total_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return result;
}

// Intercepted getcontext
int getcontext(ucontext_t* ucp) {
    init_interceptor();

    if (!real_getcontext) {
        return -1;
    }

    return real_getcontext(ucp);
}

// Intercepted setcontext
int setcontext(const ucontext_t* ucp) {
    init_interceptor();

    if (!real_setcontext) {
        return -1;
    }

    // Restore APX state if this context uses APX
    uint64_t apx_regs = get_context_apx_regs(const_cast<ucontext_t*>(ucp));
    if (apx_regs != 0 && g_apx_save_area.valid) {
        selective_xrstor(&g_apx_save_area);
    }

    return real_setcontext(ucp);
}

// Get interceptor statistics
void apx_interceptor_get_stats(uint64_t* swapcontext_calls,
                                uint64_t* swapcontext_apx_optimized,
                                uint64_t* xsave_selective_calls,
                                uint64_t* bytes_saved,
                                uint64_t* total_time_ns) {
    if (swapcontext_calls) *swapcontext_calls = g_stats.swapcontext_calls;
    if (swapcontext_apx_optimized) *swapcontext_apx_optimized = g_stats.swapcontext_apx_optimized;
    if (xsave_selective_calls) *xsave_selective_calls = g_stats.xsave_selective_calls;
    if (bytes_saved) *bytes_saved = g_stats.bytes_saved;
    if (total_time_ns) *total_time_ns = g_stats.total_time_ns;
}

// Print interceptor statistics
void apx_interceptor_print_stats() {
    fprintf(stderr, "\n=== APX Interceptor Statistics ===\n");
    fprintf(stderr, "swapcontext calls:     %lu\n", g_stats.swapcontext_calls.load());
    fprintf(stderr, "APX optimized:         %lu\n", g_stats.swapcontext_apx_optimized.load());
    fprintf(stderr, "Selective XSAVE calls: %lu\n", g_stats.xsave_selective_calls.load());
    fprintf(stderr, "Bytes saved:           %lu\n", g_stats.bytes_saved.load());
    fprintf(stderr, "Total overhead (ns):   %lu\n", g_stats.total_time_ns.load());

    if (g_stats.swapcontext_calls > 0) {
        double avg_overhead = (double)g_stats.total_time_ns / g_stats.swapcontext_calls;
        fprintf(stderr, "Avg overhead (ns):     %.1f\n", avg_overhead);
    }
    fprintf(stderr, "==================================\n");
}

// Register a function for APX optimization
uint64_t apx_interceptor_register_function(void* func_addr, size_t func_size, const char* name) {
    init_interceptor();

    if (!g_apx_manager) {
        return 0;
    }

    return g_apx_manager->register_region((uint64_t)func_addr, func_size, name ? name : "");
}

// Mark a context as using APX registers
void apx_interceptor_mark_context_apx(ucontext_t* ctx, uint64_t apx_regs_mask) {
    set_context_apx_regs(ctx, apx_regs_mask);
}

} // extern "C"

// ============================================================================
// Constructor/Destructor
// ============================================================================

__attribute__((constructor))
static void apx_interceptor_init() {
    init_interceptor();

    if (g_verbose) {
        fprintf(stderr, "[APX Interceptor] Library loaded\n");
    }
}

__attribute__((destructor))
static void apx_interceptor_fini() {
    if (g_verbose) {
        apx_interceptor_print_stats();
    }

    if (g_apx_manager) {
        delete g_apx_manager;
        g_apx_manager = nullptr;
    }
}
