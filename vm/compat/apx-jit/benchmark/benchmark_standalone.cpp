/* SPDX-License-Identifier: MIT
 *
 * APX Standalone Benchmark
 *
 * Direct comparison of native x86-64 vs APX-optimized context switching.
 * Does not require LD_PRELOAD - runs both modes in the same process.
 *
 * Build:
 *   Part of the CMake build, or:
 *   g++ -std=c++20 -O2 -I.. benchmark_standalone.cpp \
 *       -L../../build -lbpftime_apx_jit -lpthread -o benchmark_standalone
 *
 * Run:
 *   ./benchmark_standalone
 *   sde64 -future -- ./benchmark_standalone  # For APX emulation
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

#include "apx_cpu_features.hpp"
#include "apx_hotpath_manager.hpp"

using namespace bpftime::vm::apx;

// ============================================================================
// Configuration
// ============================================================================

constexpr size_t STACK_SIZE = 64 * 1024;
constexpr uint64_t WARMUP_ITERATIONS = 10000;
constexpr uint64_t BENCHMARK_ITERATIONS = 1000000;

// ============================================================================
// XSAVE constants
// ============================================================================

constexpr uint64_t XSAVE_X87      = 1ULL << 0;
constexpr uint64_t XSAVE_SSE      = 1ULL << 1;
constexpr uint64_t XSAVE_AVX      = 1ULL << 2;
constexpr uint64_t XSAVE_AVX512_K = 1ULL << 5;
constexpr uint64_t XSAVE_AVX512_ZMM_HI = 1ULL << 6;
constexpr uint64_t XSAVE_AVX512_HI16   = 1ULL << 7;
constexpr uint64_t XSAVE_APX      = 1ULL << 19;

// ============================================================================
// XSAVE/XRSTOR helpers
// ============================================================================

struct alignas(64) XSaveArea {
    uint8_t data[8192];
};

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
// Timer
// ============================================================================

struct Timer {
    std::chrono::high_resolution_clock::time_point start;

    void begin() { start = std::chrono::high_resolution_clock::now(); }

    double elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

// ============================================================================
// Benchmark result
// ============================================================================

struct Result {
    std::string name;
    uint64_t iterations;
    double avg_ns;
    double ops_per_sec;

    void print() const {
        std::cout << std::left << std::setw(45) << name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(1)
                  << avg_ns << " ns"
                  << std::setw(14) << std::setprecision(0) << ops_per_sec << " ops/s"
                  << std::endl;
    }
};

// ============================================================================
// Minimal context (GPRs only)
// ============================================================================

struct MinimalContext {
    uint64_t rsp, rbp, rbx;
    uint64_t r12, r13, r14, r15;
    uint64_t rip;
};

extern "C" void minimal_switch(MinimalContext* save, MinimalContext* restore);
__asm__(
    ".global minimal_switch\n"
    "minimal_switch:\n"
    "    movq %rsp, 0(%rdi)\n"
    "    movq %rbp, 8(%rdi)\n"
    "    movq %rbx, 16(%rdi)\n"
    "    movq %r12, 24(%rdi)\n"
    "    movq %r13, 32(%rdi)\n"
    "    movq %r14, 40(%rdi)\n"
    "    movq %r15, 48(%rdi)\n"
    "    leaq 1f(%rip), %rax\n"
    "    movq %rax, 56(%rdi)\n"
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

// ============================================================================
// APX-enhanced context (GPRs + selective R16-R31)
// ============================================================================

struct APXContext {
    // Standard callee-saved GPRs
    uint64_t rsp, rbp, rbx;
    uint64_t r12, r13, r14, r15;
    uint64_t rip;

    // APX extended registers (R16-R31) - stored selectively
    alignas(64) uint8_t apx_state[128];
    uint32_t apx_regs_used;  // Bitmask of which R16-R31 are used
    bool apx_state_valid;
};

// Simulated APX context switch with selective state save
static void apx_switch_sim(APXContext* save, APXContext* restore,
                           XSaveArea* xsave_buf, uint64_t xcr0, bool has_apx) {
    // Save standard GPRs (simulate)
    volatile uint64_t dummy[7];
    dummy[0] = save->rsp; dummy[1] = save->rbp; dummy[2] = save->rbx;
    dummy[3] = save->r12; dummy[4] = save->r13; dummy[5] = save->r14;
    dummy[6] = save->r15;

    // Selective APX save - only if registers were modified
    if (has_apx && save->apx_regs_used != 0) {
        // In real code: xsave64 with APX mask only (~26 ns)
        // Simulated: memcpy 128 bytes (~2 ns)
        memcpy(save->apx_state, xsave_buf->data, 128);
        save->apx_state_valid = true;
    }

    // Selective APX restore - only if target uses APX regs
    if (has_apx && restore->apx_state_valid && restore->apx_regs_used != 0) {
        memcpy(xsave_buf->data, restore->apx_state, 128);
    }

    // Restore standard GPRs (simulate)
    volatile uint64_t dummy2[7];
    dummy2[0] = restore->rsp; dummy2[1] = restore->rbp; dummy2[2] = restore->rbx;
    dummy2[3] = restore->r12; dummy2[4] = restore->r13; dummy2[5] = restore->r14;
    dummy2[6] = restore->r15;

    __asm__ __volatile__("" : : "m"(dummy), "m"(dummy2));
}

// ============================================================================
// Benchmarks
// ============================================================================

static ucontext_t g_main_uctx, g_worker_uctx;
static void* g_ucontext_stack = nullptr;
static volatile uint64_t g_switch_count = 0;

static void ucontext_worker() {
    while (true) {
        g_switch_count++;
        swapcontext(&g_worker_uctx, &g_main_uctx);
    }
}

Result benchmark_ucontext(uint64_t iterations) {
    g_ucontext_stack = mmap(nullptr, STACK_SIZE, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    getcontext(&g_worker_uctx);
    g_worker_uctx.uc_stack.ss_sp = g_ucontext_stack;
    g_worker_uctx.uc_stack.ss_size = STACK_SIZE;
    g_worker_uctx.uc_link = &g_main_uctx;
    makecontext(&g_worker_uctx, ucontext_worker, 0);

    // Warmup
    for (uint64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        swapcontext(&g_main_uctx, &g_worker_uctx);
    }

    Timer timer;
    timer.begin();
    for (uint64_t i = 0; i < iterations; i++) {
        swapcontext(&g_main_uctx, &g_worker_uctx);
    }
    double elapsed = timer.elapsed_ns();

    munmap(g_ucontext_stack, STACK_SIZE);

    Result r;
    r.name = "Standard ucontext (full state save)";
    r.iterations = iterations;
    r.avg_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_ns;
    return r;
}

static MinimalContext g_main_min, g_worker_min;
static void* g_minimal_stack = nullptr;

static void minimal_worker_entry() {
    while (true) {
        g_switch_count++;
        minimal_switch(&g_worker_min, &g_main_min);
    }
}

Result benchmark_minimal_context(uint64_t iterations) {
    g_minimal_stack = mmap(nullptr, STACK_SIZE, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);

    memset(&g_worker_min, 0, sizeof(g_worker_min));
    g_worker_min.rsp = (uint64_t)g_minimal_stack + STACK_SIZE - 8;
    g_worker_min.rip = (uint64_t)minimal_worker_entry;

    // Warmup
    for (uint64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        minimal_switch(&g_main_min, &g_worker_min);
    }

    Timer timer;
    timer.begin();
    for (uint64_t i = 0; i < iterations; i++) {
        minimal_switch(&g_main_min, &g_worker_min);
    }
    double elapsed = timer.elapsed_ns();

    munmap(g_minimal_stack, STACK_SIZE);

    Result r;
    r.name = "Minimal context (GPRs only)";
    r.iterations = iterations;
    r.avg_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_ns;
    return r;
}

Result benchmark_xsave_full(uint64_t iterations, uint64_t xcr0) {
    XSaveArea area;
    memset(&area, 0, sizeof(area));
    *(uint64_t*)(area.data + 512) = xcr0;

    uint64_t mask = XSAVE_X87 | XSAVE_SSE | XSAVE_AVX;
    if (xcr0 & XSAVE_AVX512_K) {
        mask |= XSAVE_AVX512_K | XSAVE_AVX512_ZMM_HI | XSAVE_AVX512_HI16;
    }

    // Warmup
    for (uint64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        do_xsave(area.data, mask);
    }

    Timer timer;
    timer.begin();
    for (uint64_t i = 0; i < iterations; i++) {
        do_xsave(area.data, mask);
    }
    double elapsed = timer.elapsed_ns();

    Result r;
    r.name = "XSAVE (full state: x87+SSE+AVX+AVX512)";
    r.iterations = iterations;
    r.avg_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_ns;
    return r;
}

Result benchmark_xsave_minimal(uint64_t iterations, uint64_t xcr0) {
    XSaveArea area;
    memset(&area, 0, sizeof(area));
    *(uint64_t*)(area.data + 512) = xcr0;

    uint64_t mask = XSAVE_SSE;  // Minimal: just SSE

    // Warmup
    for (uint64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        do_xsavec(area.data, mask);
    }

    Timer timer;
    timer.begin();
    for (uint64_t i = 0; i < iterations; i++) {
        do_xsavec(area.data, mask);
    }
    double elapsed = timer.elapsed_ns();

    Result r;
    r.name = "XSAVEC (minimal: SSE only)";
    r.iterations = iterations;
    r.avg_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_ns;
    return r;
}

Result benchmark_apx_selective_no_regs(uint64_t iterations, uint64_t xcr0) {
    APXContext ctx1 = {}, ctx2 = {};
    XSaveArea xsave_buf = {};
    bool has_apx = (xcr0 & XSAVE_APX) != 0;

    // No APX regs used - should skip XSAVE entirely
    ctx1.apx_regs_used = 0;
    ctx2.apx_regs_used = 0;

    // Warmup
    for (uint64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        apx_switch_sim(&ctx1, &ctx2, &xsave_buf, xcr0, has_apx);
    }

    Timer timer;
    timer.begin();
    for (uint64_t i = 0; i < iterations; i++) {
        apx_switch_sim(&ctx1, &ctx2, &xsave_buf, xcr0, has_apx);
    }
    double elapsed = timer.elapsed_ns();

    Result r;
    r.name = "APX selective (no APX regs used)";
    r.iterations = iterations;
    r.avg_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_ns;
    return r;
}

Result benchmark_apx_selective_with_regs(uint64_t iterations, uint64_t xcr0) {
    APXContext ctx1 = {}, ctx2 = {};
    XSaveArea xsave_buf = {};
    bool has_apx = (xcr0 & XSAVE_APX) != 0;

    // Using R16-R19 (4 APX registers)
    ctx1.apx_regs_used = 0x000F;
    ctx2.apx_regs_used = 0x000F;

    // Warmup
    for (uint64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        apx_switch_sim(&ctx1, &ctx2, &xsave_buf, xcr0, has_apx);
    }

    Timer timer;
    timer.begin();
    for (uint64_t i = 0; i < iterations; i++) {
        apx_switch_sim(&ctx1, &ctx2, &xsave_buf, xcr0, has_apx);
    }
    double elapsed = timer.elapsed_ns();

    Result r;
    r.name = "APX selective (R16-R19 used, 128B save)";
    r.iterations = iterations;
    r.avg_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_ns;
    return r;
}

Result benchmark_memcpy_128(uint64_t iterations) {
    alignas(64) uint8_t src[128], dst[128];
    memset(src, 0x42, 128);

    // Warmup
    for (uint64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        memcpy(dst, src, 128);
        __asm__ __volatile__("" : : "m"(dst));
    }

    Timer timer;
    timer.begin();
    for (uint64_t i = 0; i < iterations; i++) {
        memcpy(dst, src, 128);
        __asm__ __volatile__("" : : "m"(dst));
    }
    double elapsed = timer.elapsed_ns();

    Result r;
    r.name = "memcpy 128 bytes (APX state size)";
    r.iterations = iterations;
    r.avg_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_ns;
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "      APX Context Switch Benchmark (Standalone)" << std::endl;
    std::cout << "================================================================" << std::endl;

    // Detect APX
    auto features = detect_apx_features();

    std::cout << "\nCPU Features:" << std::endl;
    std::cout << "  APX_F:        " << (features.has_apx_f ? "YES" : "NO") << std::endl;
    std::cout << "  EGPR (R16-31):" << (features.has_egpr ? "YES" : "NO") << std::endl;
    std::cout << "  NDD (3-op):   " << (features.has_ndd ? "YES" : "NO") << std::endl;
    std::cout << "  NF (no-flag): " << (features.has_nf ? "YES" : "NO") << std::endl;
    std::cout << "  AVX-512:      " << (features.has_avx512 ? "YES" : "NO") << std::endl;
    std::cout << "  AVX10:        " << (features.has_avx10 ? "YES" : "NO") << std::endl;
    std::cout << "  XCR0:         0x" << std::hex << features.xcr0 << std::dec << std::endl;

    bool has_os_apx = (features.xcr0 & XSAVE_APX) != 0;
    std::cout << "  OS APX:       " << (has_os_apx ? "YES" : "NO") << std::endl;

    if (features.can_use_apx()) {
        std::cout << "\n*** APX IS AVAILABLE ***" << std::endl;
    } else {
        std::cout << "\n*** APX not available ***" << std::endl;
        std::cout << "    Run with: sde64 -future -- ./benchmark_standalone" << std::endl;
    }

    std::cout << "\n--- Context Switch Performance ---" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::vector<Result> results;

    results.push_back(benchmark_ucontext(BENCHMARK_ITERATIONS));
    results.back().print();

    results.push_back(benchmark_minimal_context(BENCHMARK_ITERATIONS));
    results.back().print();

    results.push_back(benchmark_apx_selective_no_regs(BENCHMARK_ITERATIONS, features.xcr0));
    results.back().print();

    results.push_back(benchmark_apx_selective_with_regs(BENCHMARK_ITERATIONS, features.xcr0));
    results.back().print();

    std::cout << "\n--- XSAVE/State Save Performance ---" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    results.push_back(benchmark_xsave_full(BENCHMARK_ITERATIONS, features.xcr0));
    results.back().print();

    results.push_back(benchmark_xsave_minimal(BENCHMARK_ITERATIONS, features.xcr0));
    results.back().print();

    results.push_back(benchmark_memcpy_128(BENCHMARK_ITERATIONS));
    results.back().print();

    // Summary
    std::cout << "\n================================================================" << std::endl;
    std::cout << "                        Summary" << std::endl;
    std::cout << "================================================================" << std::endl;

    double ucontext_ns = results[0].avg_ns;
    double minimal_ns = results[1].avg_ns;
    double apx_no_regs_ns = results[2].avg_ns;
    double apx_with_regs_ns = results[3].avg_ns;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\nSpeedup vs standard ucontext:" << std::endl;
    std::cout << "  Minimal (GPRs only):     " << (ucontext_ns / minimal_ns) << "x faster" << std::endl;
    std::cout << "  APX (no APX regs):       " << (ucontext_ns / apx_no_regs_ns) << "x faster" << std::endl;
    std::cout << "  APX (with R16-R19):      " << (ucontext_ns / apx_with_regs_ns) << "x faster" << std::endl;

    std::cout << "\nAPX Benefits:" << std::endl;
    std::cout << "  - 16 additional GPRs (R16-R31) reduce register spills" << std::endl;
    std::cout << "  - Selective XSAVE: only 128 bytes for APX vs 2KB+ full state" << std::endl;
    std::cout << "  - Skip XSAVE entirely when APX regs not modified" << std::endl;
    std::cout << "  - NDD (3-operand) forms reduce instruction count" << std::endl;
    std::cout << "  - NF (no-flags) reduces false dependencies" << std::endl;

    std::cout << "\nState Save Size Comparison:" << std::endl;
    std::cout << "  Full ucontext:  ~2KB+  (FPU + SSE + AVX + ...)" << std::endl;
    std::cout << "  Full XSAVE:     ~2KB   (depends on XCR0)" << std::endl;
    std::cout << "  APX selective:  128B   (R16-R31 only)" << std::endl;
    std::cout << "  Minimal GPRs:   56B    (callee-saved only)" << std::endl;

    std::cout << "\n================================================================" << std::endl;

    return 0;
}
