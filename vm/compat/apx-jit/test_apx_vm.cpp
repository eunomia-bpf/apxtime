/* SPDX-License-Identifier: MIT
 *
 * Test APX VM integration with bpftime
 */

#include <iostream>
#include <cstdint>
#include <cstring>

// Simple eBPF program: return R1 + 42
// r0 = r1 + 42
// exit
static const uint8_t simple_ebpf_code[] = {
    0x07, 0x01, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00,  // r1 += 42
    0xbf, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // r0 = r1
    0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // exit
};

// Forward declarations - we'll use the C API
extern "C" {

// From apx_cpu_features
struct APXFeatures {
    bool has_apx_f;
    bool has_egpr;
    bool has_push2pop2;
    bool has_ppx;
    bool has_ndd;
    bool has_ccmp;
    bool has_cf;
    bool has_nf;
    bool has_avx512;
    bool has_avx10;
    uint64_t xcr0;
};

}

// Inline CPUID
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
            features.has_ndd = true;
            features.has_nf = true;
            features.has_push2pop2 = true;
        }
    }

    features.xcr0 = get_xcr0();
    return features;
}

int main() {
    std::cout << "=== APX VM Integration Test ===" << std::endl;

    // 1. Detect APX features
    auto features = detect_apx();

    std::cout << "\nCPU Features:" << std::endl;
    std::cout << "  APX_F:   " << (features.has_apx_f ? "YES" : "NO") << std::endl;
    std::cout << "  AVX-512: " << (features.has_avx512 ? "YES" : "NO") << std::endl;
    std::cout << "  AVX10:   " << (features.has_avx10 ? "YES" : "NO") << std::endl;
    std::cout << "  XCR0:    0x" << std::hex << features.xcr0 << std::dec << std::endl;

    bool apx_available = features.has_apx_f && features.has_egpr;

    if (apx_available) {
        std::cout << "\n*** APX IS AVAILABLE ***" << std::endl;
        std::cout << "APX optimizations would be applied:" << std::endl;
        std::cout << "  - Extended GPRs R16-R31" << std::endl;
        std::cout << "  - 3-operand NDD forms" << std::endl;
        std::cout << "  - Flag suppression (NF)" << std::endl;
        std::cout << "  - PUSH2/POP2 stack ops" << std::endl;
    } else {
        std::cout << "\nAPX not available - would use standard x86-64" << std::endl;
    }

    // 2. Test simple eBPF code info
    std::cout << "\nSimple eBPF program:" << std::endl;
    std::cout << "  Size: " << sizeof(simple_ebpf_code) << " bytes" << std::endl;
    std::cout << "  Instructions: " << sizeof(simple_ebpf_code) / 8 << std::endl;
    std::cout << "  Expected: r0 = input + 42" << std::endl;

    // 3. Verify XCR0 APX bit
    bool os_apx = (features.xcr0 & (1ULL << 19)) != 0;
    std::cout << "\nOS Support:" << std::endl;
    std::cout << "  XCR0[19] (APX state): " << (os_apx ? "YES" : "NO") << std::endl;

    std::cout << "\n=== Test Complete ===" << std::endl;

    return 0;
}
