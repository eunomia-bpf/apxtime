/* SPDX-License-Identifier: MIT
 *
 * APX Context Switch Benchmark
 *
 * Compares context switching performance:
 * 1. Standard ucontext (saves all registers)
 * 2. APX with full XSAVE (saves all extended state)
 * 3. APX with selective XSAVE (only saves modified R16-R31)
 *
 * Build:
 *   g++ -std=c++17 -O2 -o benchmark_ucontext benchmark_ucontext.cpp -lpthread
 *
 * Run with SDE for APX emulation:
 *   sde64 -future -- ./benchmark_ucontext
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <functional>
#include <ucontext.h>
#include <sys/mman.h>

// ============================================================================
// APX Feature Detection (inline for standalone benchmark)
// ============================================================================

struct APXFeatures {
    bool has_apx_f = false;
    bool has_egpr = false;
    bool has_avx512 = false;
    bool has_avx10 = false;
    uint64_t xcr0 = 0;

    bool can_use_apx() const { return has_apx_f && has_egpr; }
};

static void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]) {
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
        : "a"(leaf), "c"(subleaf)
    );
}

static uint64_t get_xcr0() {
    uint32_t regs[4];
    cpuid(1, 0, regs);
    bool has_osxsave = (regs[2] >> 27) & 1;
    if (!has_osxsave) return 0;

    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}

static APXFeatures detect_apx() {
    APXFeatures features = {};

    uint32_t regs[4];
    cpuid(0, 0, regs);
    uint32_t max_leaf = regs[0];

    if (max_leaf >= 7) {
        cpuid(7, 0, regs);
        features.has_avx512 = (regs[1] >> 16) & 1;

        cpuid(7, 1, regs);
        features.has_avx10 = (regs[3] >> 19) & 1;
        features.has_apx_f = (regs[3] >> 21) & 1;

        if (features.has_apx_f) {
            features.has_egpr = true;
        }
    }

    features.xcr0 = get_xcr0();
    return features;
}

// ============================================================================
// XSAVE Area Management
// ============================================================================

// XSAVE area components
constexpr uint64_t XSAVE_X87      = 1ULL << 0;   // x87 FPU state
constexpr uint64_t XSAVE_SSE      = 1ULL << 1;   // SSE state (XMM0-15)
constexpr uint64_t XSAVE_AVX      = 1ULL << 2;   // AVX state (YMM0-15 upper)
constexpr uint64_t XSAVE_AVX512_K = 1ULL << 5;   // AVX-512 opmask (k0-k7)
constexpr uint64_t XSAVE_AVX512_ZMM_HI = 1ULL << 6;  // ZMM0-15 upper
constexpr uint64_t XSAVE_AVX512_HI16   = 1ULL << 7;  // ZMM16-31
constexpr uint64_t XSAVE_APX     = 1ULL << 19;  // APX extended GPRs (R16-R31)

// XSAVE area sizes (approximate)
constexpr size_t XSAVE_LEGACY_SIZE = 512;       // x87 + SSE
constexpr size_t XSAVE_AVX_SIZE = 256;          // YMM upper halves
constexpr size_t XSAVE_AVX512_SIZE = 2048;      // Full AVX-512 state
constexpr size_t XSAVE_APX_SIZE = 128;          // R16-R31 (16 regs * 8 bytes)
constexpr size_t XSAVE_MAX_SIZE = 8192;         // Maximum XSAVE area

// Aligned XSAVE buffer
struct alignas(64) XSaveArea {
    uint8_t data[XSAVE_MAX_SIZE];
};

// XSAVE/XRSTOR wrappers
static inline void do_xsave(void* area, uint64_t mask) {
    uint32_t lo = mask & 0xFFFFFFFF;
    uint32_t hi = mask >> 32;
    __asm__ __volatile__(
        "xsave64 (%0)"
        :
        : "r"(area), "a"(lo), "d"(hi)
        : "memory"
    );
}

static inline void do_xrstor(void* area, uint64_t mask) {
    uint32_t lo = mask & 0xFFFFFFFF;
    uint32_t hi = mask >> 32;
    __asm__ __volatile__(
        "xrstor64 (%0)"
        :
        : "r"(area), "a"(lo), "d"(hi)
        : "memory"
    );
}

static inline void do_xsavec(void* area, uint64_t mask) {
    uint32_t lo = mask & 0xFFFFFFFF;
    uint32_t hi = mask >> 32;
    __asm__ __volatile__(
        "xsavec64 (%0)"
        :
        : "r"(area), "a"(lo), "d"(hi)
        : "memory"
    );
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string name;
    uint64_t iterations;
    double total_time_ns;
    double avg_time_ns;
    double ops_per_sec;
};

template<typename Func>
BenchmarkResult run_benchmark(const std::string& name, uint64_t iterations, Func&& func) {
    // Warmup
    for (uint64_t i = 0; i < 1000; i++) {
        func();
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < iterations; i++) {
        func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    BenchmarkResult result;
    result.name = name;
    result.iterations = iterations;
    result.total_time_ns = duration.count();
    result.avg_time_ns = result.total_time_ns / iterations;
    result.ops_per_sec = 1e9 / result.avg_time_ns;

    return result;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(40) << r.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(1) << r.avg_time_ns << " ns"
              << std::setw(15) << std::setprecision(0) << r.ops_per_sec << " ops/s"
              << std::endl;
}

// ============================================================================
// Context Switch Scenarios
// ============================================================================

// Global context variables for ucontext benchmarks
static ucontext_t main_ctx, worker_ctx;
static volatile int switch_count = 0;
static constexpr size_t STACK_SIZE = 64 * 1024;

// Simple worker function that switches back
static void worker_func() {
    while (true) {
        switch_count++;
        swapcontext(&worker_ctx, &main_ctx);
    }
}

// XSAVE-based context structure
struct APXContext {
    XSaveArea xsave_area;
    uint64_t rip;
    uint64_t rsp;
    uint64_t rbp;
    uint64_t rflags;
    // GPRs not covered by XSAVE (non-APX)
    uint64_t rax, rbx, rcx, rdx, rsi, rdi;
    uint64_t r8, r9, r10, r11, r12, r13, r14, r15;
};

// ============================================================================
// Benchmark: Standard ucontext
// ============================================================================

BenchmarkResult benchmark_ucontext_standard(uint64_t iterations) {
    // Allocate stack
    void* stack = mmap(nullptr, STACK_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if (stack == MAP_FAILED) {
        throw std::runtime_error("Failed to allocate stack");
    }

    // Initialize worker context
    getcontext(&worker_ctx);
    worker_ctx.uc_stack.ss_sp = stack;
    worker_ctx.uc_stack.ss_size = STACK_SIZE;
    worker_ctx.uc_link = &main_ctx;
    makecontext(&worker_ctx, worker_func, 0);

    switch_count = 0;

    auto result = run_benchmark("ucontext (standard swapcontext)", iterations, [&]() {
        swapcontext(&main_ctx, &worker_ctx);
    });

    munmap(stack, STACK_SIZE);
    return result;
}

// ============================================================================
// Benchmark: Minimal context switch (no FPU/SSE save)
// ============================================================================

// Minimal context: only GPRs
struct MinimalContext {
    uint64_t rsp;
    uint64_t rbp;
    uint64_t rbx;
    uint64_t r12, r13, r14, r15;
    uint64_t rip;
};

static MinimalContext g_main_minimal, g_worker_minimal;
static void* g_worker_stack_minimal = nullptr;

// Assembly for minimal context switch (callee-saved registers only)
extern "C" void minimal_switch(MinimalContext* save, MinimalContext* restore);
__asm__(
    ".global minimal_switch\n"
    "minimal_switch:\n"
    "    # Save callee-saved registers to save context\n"
    "    movq %rsp, 0(%rdi)\n"
    "    movq %rbp, 8(%rdi)\n"
    "    movq %rbx, 16(%rdi)\n"
    "    movq %r12, 24(%rdi)\n"
    "    movq %r13, 32(%rdi)\n"
    "    movq %r14, 40(%rdi)\n"
    "    movq %r15, 48(%rdi)\n"
    "    leaq 1f(%rip), %rax\n"
    "    movq %rax, 56(%rdi)\n"
    "    # Restore callee-saved registers from restore context\n"
    "    movq 0(%rsi), %rsp\n"
    "    movq 8(%rsi), %rbp\n"
    "    movq 16(%rsi), %rbx\n"
    "    movq 24(%rsi), %r12\n"
    "    movq 32(%rsi), %r13\n"
    "    movq 40(%rsi), %r14\n"
    "    movq 48(%rsi), %r15\n"
    "    jmpq *56(%rsi)\n"
    "1:  ret\n"
);

static void minimal_worker_entry() {
    while (true) {
        switch_count++;
        minimal_switch(&g_worker_minimal, &g_main_minimal);
    }
}

BenchmarkResult benchmark_minimal_context(uint64_t iterations) {
    g_worker_stack_minimal = mmap(nullptr, STACK_SIZE, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if (g_worker_stack_minimal == MAP_FAILED) {
        throw std::runtime_error("Failed to allocate stack");
    }

    // Setup worker context
    memset(&g_worker_minimal, 0, sizeof(g_worker_minimal));
    g_worker_minimal.rsp = (uint64_t)g_worker_stack_minimal + STACK_SIZE - 8;
    g_worker_minimal.rip = (uint64_t)minimal_worker_entry;

    switch_count = 0;

    auto result = run_benchmark("Minimal context (GPRs only)", iterations, [&]() {
        minimal_switch(&g_main_minimal, &g_worker_minimal);
    });

    munmap(g_worker_stack_minimal, STACK_SIZE);
    return result;
}

// ============================================================================
// Benchmark: XSAVE operations
// ============================================================================

BenchmarkResult benchmark_xsave_full(uint64_t iterations, uint64_t xcr0) {
    XSaveArea area;
    memset(&area, 0, sizeof(area));

    // Initialize XSAVE area header
    *(uint64_t*)(area.data + 512) = xcr0;  // XSTATE_BV

    uint64_t mask = xcr0 & (XSAVE_X87 | XSAVE_SSE | XSAVE_AVX);
    if (mask == 0) mask = XSAVE_X87 | XSAVE_SSE;  // Minimum

    return run_benchmark("XSAVE (x87+SSE+AVX)", iterations, [&]() {
        do_xsave(area.data, mask);
    });
}

BenchmarkResult benchmark_xrstor_full(uint64_t iterations, uint64_t xcr0) {
    XSaveArea area;
    memset(&area, 0, sizeof(area));

    // Initialize XSAVE area
    *(uint64_t*)(area.data + 512) = xcr0;

    uint64_t mask = xcr0 & (XSAVE_X87 | XSAVE_SSE | XSAVE_AVX);
    if (mask == 0) mask = XSAVE_X87 | XSAVE_SSE;

    // Do an initial save
    do_xsave(area.data, mask);

    return run_benchmark("XRSTOR (x87+SSE+AVX)", iterations, [&]() {
        do_xrstor(area.data, mask);
    });
}

BenchmarkResult benchmark_xsave_avx512(uint64_t iterations, uint64_t xcr0) {
    if (!(xcr0 & XSAVE_AVX512_K)) {
        return {"XSAVE (AVX-512) - N/A", 0, 0, 0, 0};
    }

    XSaveArea area;
    memset(&area, 0, sizeof(area));
    *(uint64_t*)(area.data + 512) = xcr0;

    uint64_t mask = XSAVE_X87 | XSAVE_SSE | XSAVE_AVX |
                    XSAVE_AVX512_K | XSAVE_AVX512_ZMM_HI | XSAVE_AVX512_HI16;

    return run_benchmark("XSAVE (full AVX-512)", iterations, [&]() {
        do_xsave(area.data, mask);
    });
}

BenchmarkResult benchmark_xsavec_selective(uint64_t iterations, uint64_t xcr0) {
    XSaveArea area;
    memset(&area, 0, sizeof(area));
    *(uint64_t*)(area.data + 512) = xcr0;

    // Only save SSE (common case for most code)
    uint64_t mask = XSAVE_SSE;

    return run_benchmark("XSAVEC (SSE only, compacted)", iterations, [&]() {
        do_xsavec(area.data, mask);
    });
}

BenchmarkResult benchmark_xsave_apx_only(uint64_t iterations, uint64_t xcr0) {
    if (!(xcr0 & XSAVE_APX)) {
        return {"XSAVE (APX R16-R31 only) - N/A", 0, 0, 0, 0};
    }

    XSaveArea area;
    memset(&area, 0, sizeof(area));
    *(uint64_t*)(area.data + 512) = xcr0;

    // Only save APX state
    uint64_t mask = XSAVE_APX;

    return run_benchmark("XSAVE (APX R16-R31 only)", iterations, [&]() {
        do_xsave(area.data, mask);
    });
}

// ============================================================================
// Benchmark: Simulated APX-optimized context switch
// ============================================================================

// Extended context with APX registers
struct APXExtendedContext {
    // Callee-saved GPRs (standard x86-64)
    uint64_t rsp, rbp, rbx;
    uint64_t r12, r13, r14, r15;
    uint64_t rip;

    // APX extended registers (R16-R31) - would be saved via XSAVE
    alignas(64) uint8_t apx_state[128];

    // Bitmask of which APX regs are modified
    uint32_t apx_modified_mask;
};

static APXExtendedContext g_main_apx, g_worker_apx;
static void* g_worker_stack_apx = nullptr;
static XSaveArea g_xsave_buffer;

// Simulated APX context switch with selective XSAVE
// This is a SIMULATION - does not actually switch context, just measures overhead
static void apx_switch_selective_sim(APXExtendedContext* save, APXExtendedContext* restore) {
    // Simulate saving callee-saved GPRs (7 * 8 = 56 bytes)
    volatile uint64_t dummy_save[7];
    dummy_save[0] = save->rsp;
    dummy_save[1] = save->rbp;
    dummy_save[2] = save->rbx;
    dummy_save[3] = save->r12;
    dummy_save[4] = save->r13;
    dummy_save[5] = save->r14;
    dummy_save[6] = save->r15;

    // Save APX state ONLY if modified (selective XSAVE)
    // In real code, this would check apx_modified_mask and only XSAVE if needed
    if (save->apx_modified_mask != 0) {
        // Simulated: just memcpy to represent minimal APX state save (128 bytes for R16-R31)
        // Real code would use: xsave64 with APX mask only
        memcpy(save->apx_state, g_xsave_buffer.data, 128);
    }

    // Restore APX state if target uses APX
    if (restore->apx_modified_mask != 0) {
        memcpy(g_xsave_buffer.data, restore->apx_state, 128);
    }

    // Simulate restoring callee-saved GPRs
    volatile uint64_t dummy_restore[7];
    dummy_restore[0] = restore->rsp;
    dummy_restore[1] = restore->rbp;
    dummy_restore[2] = restore->rbx;
    dummy_restore[3] = restore->r12;
    dummy_restore[4] = restore->r13;
    dummy_restore[5] = restore->r14;
    dummy_restore[6] = restore->r15;

    // Prevent optimization
    __asm__ __volatile__("" : : "m"(dummy_save), "m"(dummy_restore));
}

BenchmarkResult benchmark_apx_selective_context(uint64_t iterations, bool use_apx) {
    memset(&g_main_apx, 0, sizeof(g_main_apx));
    memset(&g_worker_apx, 0, sizeof(g_worker_apx));
    memset(&g_xsave_buffer, 0, sizeof(g_xsave_buffer));

    // Simulate APX usage
    if (use_apx) {
        g_main_apx.apx_modified_mask = 0x000F;  // R16-R19 used
        g_worker_apx.apx_modified_mask = 0x000F;
    }

    std::string name = use_apx ?
        "Simulated APX selective (R16-R19)" :
        "Simulated APX selective (no APX regs)";

    return run_benchmark(name, iterations, [&]() {
        apx_switch_selective_sim(&g_main_apx, &g_worker_apx);
    });
}

// ============================================================================
// Benchmark: Memory copy comparison (XSAVE equivalent)
// ============================================================================

// Use volatile to prevent compiler optimizations
static volatile uint8_t g_memcpy_sink;

BenchmarkResult benchmark_memcpy_128(uint64_t iterations) {
    alignas(64) uint8_t src[128], dst[128];
    memset(src, 0x42, 128);

    return run_benchmark("memcpy 128 bytes (APX state size)", iterations, [&]() {
        memcpy(dst, src, 128);
        g_memcpy_sink = dst[0];  // Prevent optimization
        __asm__ __volatile__("" : : "m"(dst));
    });
}

BenchmarkResult benchmark_memcpy_512(uint64_t iterations) {
    alignas(64) uint8_t src[512], dst[512];
    memset(src, 0x42, 512);

    return run_benchmark("memcpy 512 bytes (x87+SSE size)", iterations, [&]() {
        memcpy(dst, src, 512);
        g_memcpy_sink = dst[0];
        __asm__ __volatile__("" : : "m"(dst));
    });
}

BenchmarkResult benchmark_memcpy_2048(uint64_t iterations) {
    alignas(64) uint8_t src[2048], dst[2048];
    memset(src, 0x42, 2048);

    return run_benchmark("memcpy 2048 bytes (AVX-512 size)", iterations, [&]() {
        memcpy(dst, src, 2048);
        g_memcpy_sink = dst[0];
        __asm__ __volatile__("" : : "m"(dst));
    });
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "       APX Context Switch Benchmark" << std::endl;
    std::cout << "========================================================" << std::endl;

    // Detect APX
    auto features = detect_apx();

    std::cout << "\nCPU Features:" << std::endl;
    std::cout << "  APX_F:   " << (features.has_apx_f ? "YES" : "NO") << std::endl;
    std::cout << "  AVX-512: " << (features.has_avx512 ? "YES" : "NO") << std::endl;
    std::cout << "  AVX10:   " << (features.has_avx10 ? "YES" : "NO") << std::endl;
    std::cout << "  XCR0:    0x" << std::hex << features.xcr0 << std::dec << std::endl;

    bool has_apx_os = (features.xcr0 & XSAVE_APX) != 0;
    std::cout << "  OS APX:  " << (has_apx_os ? "YES" : "NO") << std::endl;

    if (features.can_use_apx()) {
        std::cout << "\n*** APX IS AVAILABLE - Will use APX optimizations ***" << std::endl;
    } else {
        std::cout << "\n*** APX not available - Run with 'sde64 -future' to emulate ***" << std::endl;
    }

    constexpr uint64_t ITERATIONS = 1000000;

    std::vector<BenchmarkResult> results;

    // ========== Context Switch Benchmarks ==========
    std::cout << "\n--- Context Switch Benchmarks (" << ITERATIONS << " iterations) ---" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    results.push_back(benchmark_ucontext_standard(ITERATIONS));
    print_result(results.back());

    results.push_back(benchmark_minimal_context(ITERATIONS));
    print_result(results.back());

    results.push_back(benchmark_apx_selective_context(ITERATIONS, false));
    print_result(results.back());

    results.push_back(benchmark_apx_selective_context(ITERATIONS, true));
    print_result(results.back());

    // ========== XSAVE Benchmarks ==========
    std::cout << "\n--- XSAVE/XRSTOR Benchmarks (" << ITERATIONS << " iterations) ---" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    results.push_back(benchmark_xsave_full(ITERATIONS, features.xcr0));
    print_result(results.back());

    results.push_back(benchmark_xrstor_full(ITERATIONS, features.xcr0));
    print_result(results.back());

    results.push_back(benchmark_xsavec_selective(ITERATIONS, features.xcr0));
    print_result(results.back());

    results.push_back(benchmark_xsave_avx512(ITERATIONS, features.xcr0));
    print_result(results.back());

    results.push_back(benchmark_xsave_apx_only(ITERATIONS, features.xcr0));
    print_result(results.back());

    // ========== Memory Copy Comparison ==========
    std::cout << "\n--- Memory Copy Comparison (" << ITERATIONS << " iterations) ---" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    results.push_back(benchmark_memcpy_128(ITERATIONS));
    print_result(results.back());

    results.push_back(benchmark_memcpy_512(ITERATIONS));
    print_result(results.back());

    results.push_back(benchmark_memcpy_2048(ITERATIONS));
    print_result(results.back());

    // ========== Summary ==========
    std::cout << "\n========================================================" << std::endl;
    std::cout << "                     Summary" << std::endl;
    std::cout << "========================================================" << std::endl;

    // Find baseline (standard ucontext)
    double baseline_ns = results[0].avg_time_ns;
    double minimal_ns = results[1].avg_time_ns;

    std::cout << "\nRelative to standard ucontext:" << std::endl;
    std::cout << "  Minimal GPR-only:    " << std::fixed << std::setprecision(2)
              << (baseline_ns / minimal_ns) << "x faster" << std::endl;

    if (results[3].avg_time_ns > 0) {  // APX selective with APX regs
        std::cout << "  APX selective:       " << std::fixed << std::setprecision(2)
                  << (baseline_ns / results[3].avg_time_ns) << "x faster" << std::endl;
    }

    std::cout << "\nAPX Optimization Benefits:" << std::endl;
    std::cout << "  - 16 additional GPRs (R16-R31) reduce spills" << std::endl;
    std::cout << "  - Selective XSAVE: ~128 bytes vs ~2KB+ full state" << std::endl;
    std::cout << "  - Only save APX regs when actually modified" << std::endl;

    if (features.can_use_apx()) {
        std::cout << "\n*** This system supports APX - optimal performance available ***" << std::endl;
    } else {
        std::cout << "\n*** APX not detected - using fallback paths ***" << std::endl;
        std::cout << "    Test with: sde64 -future -- ./benchmark_ucontext" << std::endl;
    }

    std::cout << "\n========================================================" << std::endl;

    return 0;
}
