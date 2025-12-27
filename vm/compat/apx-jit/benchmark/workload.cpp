/* SPDX-License-Identifier: MIT
 *
 * APX Benchmark Workload
 *
 * This is a standalone workload that exercises context switching patterns
 * commonly found in eBPF/userspace applications. It can be run:
 *
 * 1. Natively (baseline):
 *    ./apx_workload
 *
 * 2. With APX interception:
 *    LD_PRELOAD=./libapx_interceptor.so ./apx_workload
 *
 * The workload includes:
 * - ucontext-based coroutines
 * - Hot path functions that benefit from APX optimization
 * - Memory-intensive operations that stress register allocation
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
#include <dlfcn.h>

// ============================================================================
// Configuration
// ============================================================================

constexpr size_t STACK_SIZE = 64 * 1024;
constexpr uint64_t DEFAULT_ITERATIONS = 1000000;

// ============================================================================
// Timing utilities
// ============================================================================

struct Timer {
    std::chrono::high_resolution_clock::time_point start;

    void begin() {
        start = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    double elapsed_us() const {
        return elapsed_ns() / 1000.0;
    }

    double elapsed_ms() const {
        return elapsed_ns() / 1000000.0;
    }
};

// ============================================================================
// Hot path functions - candidates for APX optimization
// These functions are marked with specific attributes for interception
// ============================================================================

// Attribute to mark functions for APX optimization
#define APX_HOT_PATH __attribute__((noinline, visibility("default")))

// Hot path 1: Compute-intensive function with many local variables
// Benefits from R16-R31 to avoid register spills
APX_HOT_PATH
uint64_t compute_intensive(const uint64_t* data, size_t count) {
    // Use many local variables to stress register allocation
    uint64_t sum = 0;
    uint64_t prod = 1;
    uint64_t xor_val = 0;
    uint64_t and_val = ~0ULL;
    uint64_t or_val = 0;
    uint64_t min_val = UINT64_MAX;
    uint64_t max_val = 0;
    uint64_t count_nonzero = 0;
    uint64_t running_avg = 0;
    uint64_t variance_acc = 0;
    uint64_t prev_val = 0;
    uint64_t diff_sum = 0;

    for (size_t i = 0; i < count; i++) {
        uint64_t val = data[i];

        sum += val;
        prod *= (val | 1);  // Avoid zero
        xor_val ^= val;
        and_val &= val;
        or_val |= val;

        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        if (val != 0) count_nonzero++;

        running_avg = (running_avg * i + val) / (i + 1);
        variance_acc += (val > running_avg) ? (val - running_avg) : (running_avg - val);

        if (i > 0) {
            diff_sum += (val > prev_val) ? (val - prev_val) : (prev_val - val);
        }
        prev_val = val;
    }

    return sum ^ prod ^ xor_val ^ and_val ^ or_val ^
           min_val ^ max_val ^ count_nonzero ^ running_avg ^ variance_acc ^ diff_sum;
}

// Hot path 2: Memory-intensive function
APX_HOT_PATH
void memory_intensive(uint64_t* dst, const uint64_t* src, size_t count) {
    // Interleaved read-modify-write pattern
    uint64_t carry = 0;
    uint64_t prev = 0;

    for (size_t i = 0; i < count; i++) {
        uint64_t val = src[i];
        uint64_t modified = val ^ carry ^ prev;
        dst[i] = modified;
        carry = modified >> 32;
        prev = val;
    }
}

// Hot path 3: Branch-heavy function
APX_HOT_PATH
uint64_t branch_heavy(const uint64_t* data, size_t count, uint64_t threshold) {
    uint64_t result = 0;
    uint64_t streak = 0;
    uint64_t max_streak = 0;

    for (size_t i = 0; i < count; i++) {
        uint64_t val = data[i];

        if (val > threshold) {
            result += val;
            streak++;
        } else if (val == threshold) {
            result ^= val;
            if (streak > max_streak) max_streak = streak;
            streak = 0;
        } else {
            result -= val;
            streak = 0;
        }
    }

    return result ^ max_streak;
}

// ============================================================================
// Coroutine-based workload using ucontext
// ============================================================================

struct Coroutine {
    ucontext_t ctx;
    void* stack;
    bool finished;
    uint64_t result;

    Coroutine() : stack(nullptr), finished(false), result(0) {}

    ~Coroutine() {
        if (stack) {
            munmap(stack, STACK_SIZE);
        }
    }

    bool init(void (*func)(void*), void* arg) {
        stack = mmap(nullptr, STACK_SIZE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
        if (stack == MAP_FAILED) {
            return false;
        }

        if (getcontext(&ctx) == -1) {
            return false;
        }

        ctx.uc_stack.ss_sp = stack;
        ctx.uc_stack.ss_size = STACK_SIZE;
        ctx.uc_link = nullptr;

        // Note: makecontext is limited, we use a wrapper
        makecontext(&ctx, (void(*)())func, 1, arg);
        return true;
    }
};

static ucontext_t g_main_ctx;
static Coroutine* g_current_coro = nullptr;

APX_HOT_PATH
void yield_to_main() {
    if (g_current_coro) {
        swapcontext(&g_current_coro->ctx, &g_main_ctx);
    }
}

APX_HOT_PATH
void resume_coroutine(Coroutine* coro) {
    g_current_coro = coro;
    swapcontext(&g_main_ctx, &coro->ctx);
    g_current_coro = nullptr;
}

// ============================================================================
// Benchmark scenarios
// ============================================================================

struct BenchmarkResult {
    std::string name;
    uint64_t iterations;
    double total_time_ms;
    double avg_time_ns;
    double ops_per_sec;
};

void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(40) << r.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(1)
              << r.avg_time_ns << " ns"
              << std::setw(15) << std::setprecision(0) << r.ops_per_sec << " ops/s"
              << std::endl;
}

// Benchmark 1: Pure computation
BenchmarkResult benchmark_compute(uint64_t iterations) {
    std::vector<uint64_t> data(1024);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = i * 7 + 13;
    }

    Timer timer;
    timer.begin();

    volatile uint64_t result = 0;
    for (uint64_t i = 0; i < iterations; i++) {
        result = compute_intensive(data.data(), data.size());
    }

    double elapsed = timer.elapsed_ns();

    BenchmarkResult r;
    r.name = "compute_intensive (hot path)";
    r.iterations = iterations;
    r.total_time_ms = elapsed / 1e6;
    r.avg_time_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_time_ns;
    return r;
}

// Benchmark 2: Memory operations
BenchmarkResult benchmark_memory(uint64_t iterations) {
    std::vector<uint64_t> src(1024), dst(1024);
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = i * 11 + 17;
    }

    Timer timer;
    timer.begin();

    for (uint64_t i = 0; i < iterations; i++) {
        memory_intensive(dst.data(), src.data(), src.size());
    }

    double elapsed = timer.elapsed_ns();

    BenchmarkResult r;
    r.name = "memory_intensive (hot path)";
    r.iterations = iterations;
    r.total_time_ms = elapsed / 1e6;
    r.avg_time_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_time_ns;
    return r;
}

// Benchmark 3: Branch-heavy
BenchmarkResult benchmark_branch(uint64_t iterations) {
    std::vector<uint64_t> data(1024);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = (i * 13) % 100;
    }

    Timer timer;
    timer.begin();

    volatile uint64_t result = 0;
    for (uint64_t i = 0; i < iterations; i++) {
        result = branch_heavy(data.data(), data.size(), 50);
    }

    double elapsed = timer.elapsed_ns();

    BenchmarkResult r;
    r.name = "branch_heavy (hot path)";
    r.iterations = iterations;
    r.total_time_ms = elapsed / 1e6;
    r.avg_time_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_time_ns;
    return r;
}

// Benchmark 4: Context switching
BenchmarkResult benchmark_context_switch(uint64_t iterations) {
    static volatile uint64_t switch_count = 0;

    auto worker_func = [](void* arg) {
        while (true) {
            switch_count++;
            yield_to_main();
        }
    };

    Coroutine coro;
    if (!coro.init((void(*)(void*))worker_func, nullptr)) {
        std::cerr << "Failed to create coroutine" << std::endl;
        return {"context_switch (ucontext)", 0, 0, 0, 0};
    }

    Timer timer;
    timer.begin();

    for (uint64_t i = 0; i < iterations; i++) {
        resume_coroutine(&coro);
    }

    double elapsed = timer.elapsed_ns();

    BenchmarkResult r;
    r.name = "context_switch (ucontext)";
    r.iterations = iterations;
    r.total_time_ms = elapsed / 1e6;
    r.avg_time_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_time_ns;
    return r;
}

// Benchmark 5: Mixed workload (compute + context switch)
BenchmarkResult benchmark_mixed(uint64_t iterations) {
    std::vector<uint64_t> data(256);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = i * 7 + 13;
    }

    static volatile uint64_t global_result = 0;
    static const uint64_t* global_data = nullptr;
    static size_t global_count = 0;

    global_data = data.data();
    global_count = data.size();

    auto worker_func = [](void* arg) {
        while (true) {
            // Do some computation in the coroutine
            global_result ^= compute_intensive(global_data, global_count);
            yield_to_main();
        }
    };

    Coroutine coro;
    if (!coro.init((void(*)(void*))worker_func, nullptr)) {
        return {"mixed_workload", 0, 0, 0, 0};
    }

    Timer timer;
    timer.begin();

    for (uint64_t i = 0; i < iterations; i++) {
        resume_coroutine(&coro);
    }

    double elapsed = timer.elapsed_ns();

    BenchmarkResult r;
    r.name = "mixed_workload (compute + context)";
    r.iterations = iterations;
    r.total_time_ms = elapsed / 1e6;
    r.avg_time_ns = elapsed / iterations;
    r.ops_per_sec = 1e9 / r.avg_time_ns;
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    uint64_t iterations = DEFAULT_ITERATIONS;

    if (argc > 1) {
        iterations = std::stoull(argv[1]);
    }

    // Check if we're running under APX interception
    bool has_interceptor = (dlsym(RTLD_DEFAULT, "apx_interceptor_active") != nullptr);

    std::cout << "========================================================" << std::endl;
    std::cout << "           APX Benchmark Workload" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Mode: " << (has_interceptor ? "APX INTERCEPTED" : "NATIVE") << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::endl;

    std::vector<BenchmarkResult> results;

    std::cout << "--- Hot Path Benchmarks ---" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    results.push_back(benchmark_compute(iterations));
    print_result(results.back());

    results.push_back(benchmark_memory(iterations));
    print_result(results.back());

    results.push_back(benchmark_branch(iterations));
    print_result(results.back());

    std::cout << std::endl;
    std::cout << "--- Context Switch Benchmarks ---" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    results.push_back(benchmark_context_switch(iterations));
    print_result(results.back());

    results.push_back(benchmark_mixed(iterations / 10));  // Fewer iterations for mixed
    print_result(results.back());

    std::cout << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "                     Summary" << std::endl;
    std::cout << "========================================================" << std::endl;

    double total_time = 0;
    for (const auto& r : results) {
        total_time += r.total_time_ms;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total benchmark time: " << total_time << " ms" << std::endl;
    std::cout << "Mode: " << (has_interceptor ? "APX INTERCEPTED" : "NATIVE") << std::endl;

    if (has_interceptor) {
        std::cout << "\n*** Running with APX optimizations enabled ***" << std::endl;
    } else {
        std::cout << "\nTo run with APX interception:" << std::endl;
        std::cout << "  LD_PRELOAD=./libapx_interceptor.so ./apx_workload" << std::endl;
    }

    std::cout << "========================================================" << std::endl;

    return 0;
}
