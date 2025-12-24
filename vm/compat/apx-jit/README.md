# APX-Aware JIT Backend and Dynamic Binary Translator for bpftime

This module provides an APX (Advanced Performance Extensions) aware JIT backend and Dynamic Binary Translator (DBT) for bpftime. It enables:

1. **eBPF → APX JIT**: Compile eBPF bytecode to APX-optimized native code
2. **x86-64 → APX DBT**: Lift legacy x86-64 code to LLVM IR and regenerate with APX features
3. **Selective XSAVE**: Only save/restore extended registers (R16-R31) that are modified

## Overview

Intel APX introduces several powerful features for improved code generation:

- **Extended GPRs (R16-R31)**: 16 additional general-purpose registers, reducing register spills
- **3-Operand Forms (NDD)**: New Data Destination encoding for shorter dependency chains
- **Flag Suppression (NF)**: No-Flags forms to reduce EFLAGS pressure
- **PUSH2/POP2**: Double-width push/pop for faster stack operations
- **Conditional Compare (CCMP)**: Conditional flag-setting operations
- **Conditional Faults (CF)**: Conditional memory operations

## Features

### CPU Detection

The module automatically detects APX capabilities using CPUID:
- Checks CPUID Leaf 7, Subleaf 1 for APX_F (bit 21)
- Verifies XCR0 for OS support of extended state
- Caches detection results for performance

### LLVM Integration

When APX is available:
- Configures LLVM TargetMachine with APX features
- Generates feature strings like `+egpr,+ndd,+nf,+push2pop2`
- Falls back gracefully on non-APX hardware

## Usage

### Building

```bash
# Build with APX JIT enabled (default)
cmake -Bbuild -DBPFTIME_APX_JIT=ON -DBPFTIME_LLVM_JIT=ON
cmake --build build

# Build without APX JIT
cmake -Bbuild -DBPFTIME_APX_JIT=OFF
cmake --build build
```

### Runtime Configuration

Control APX usage via environment variable:

```bash
# Enable APX (default if hardware supports)
export BPFTIME_APX_ENABLED=1

# Disable APX (use standard x86-64)
export BPFTIME_APX_ENABLED=0
```

### API Usage

```cpp
#include <bpftime_vm_compat.hpp>

// Create APX-aware VM instance
auto vm = bpftime::vm::compat::create_vm_instance("apx_llvm");

// Or use the standard VM which will auto-detect APX
auto vm = bpftime::vm::compat::create_vm_instance("llvm");

// Load and compile eBPF code
vm->load_code(code, code_len);
auto func = vm->compile();

// Execute with APX optimizations
uint64_t result;
vm->exec(memory, memory_len, result);
```

### Checking APX Status

```cpp
#include "apx_cpu_features.hpp"

using namespace bpftime::vm::apx;

// Detect APX features
APXFeatures features = detect_apx_features();

if (features.can_use_apx()) {
    std::cout << "APX optimizations enabled!" << std::endl;
    std::cout << "  Extended GPRs: " << features.has_egpr << std::endl;
    std::cout << "  3-operand forms: " << features.has_ndd << std::endl;
    std::cout << "  Flag suppression: " << features.has_nf << std::endl;
}

// Get LLVM configuration
std::string llvm_features = get_llvm_apx_features(features);
std::string cpu_target = get_llvm_cpu_target(features);
```

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Application/bpftime         │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     VM Factory (create_vm_instance) │
                    │                                     │
                    │  "llvm" → Standard LLVM JIT         │
                    │  "apx_llvm" → APX-aware LLVM JIT    │
                    │  "ubpf" → uBPF interpreter          │
                    └─────────────────────────────────────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
              ┌──────────────┐ ┌────────────┐ ┌─────────┐
              │  APX LLVM VM │ │  LLVM VM   │ │ uBPF VM │
              └──────────────┘ └────────────┘ └─────────┘
                    │                │
                    ▼                ▼
           ┌────────────────────────────────────────────┐
           │             LLVM JIT Infrastructure         │
           │                                             │
           │  ┌─────────────┐  ┌──────────────────────┐ │
           │  │ APX Config  │  │  Standard x86-64     │ │
           │  │ +egpr       │  │  generic CPU         │ │
           │  │ +ndd        │  │  no APX features     │ │
           │  │ +nf         │  │                      │ │
           │  └─────────────┘  └──────────────────────┘ │
           └────────────────────────────────────────────┘
                              │
                              ▼
           ┌────────────────────────────────────────────┐
           │           Native x86-64/APX Code           │
           │                                             │
           │  APX Mode:         Legacy Mode:             │
           │  - Uses R16-R31    - Uses R8-R15 only      │
           │  - NDD forms       - 2-operand forms       │
           │  - NF variants     - Standard flags        │
           └────────────────────────────────────────────┘
```

## Performance Benefits

APX optimizations can provide significant benefits for eBPF JIT:

### Reduced Register Spills
- eBPF has 11 virtual registers (R0-R10)
- Standard x86-64: 16 GPRs → some spills needed for complex code
- APX: 32 GPRs → virtual registers fit comfortably, fewer memory ops

### Shorter Dependency Chains
- 3-operand forms: `ADD R16, R8, R9` instead of `MOV R16, R8; ADD R16, R9`
- Reduced instruction count and shorter critical paths

### Better Branch Prediction
- Flag suppression avoids unnecessary EFLAGS updates
- Reduces false dependencies between instructions

## Dynamic Binary Translation Pipeline

The full DBT pipeline for rewriting legacy x86-64 code to APX:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Legacy x86-64 Code                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Rellume Lifter (x86-64 → LLVM IR)                  │
│                                                                 │
│  • Decode instructions using XED                                │
│  • Build control flow graph                                     │
│  • Generate LLVM IR preserving semantics                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  APX Optimization Passes                        │
│                                                                 │
│  • Register remapping: Move temporaries to R16-R31             │
│  • NDD conversion: 2-op → 3-op forms                           │
│  • NF insertion: Add flag suppression where flags unused       │
│  • Dead flag elimination: Remove redundant EFLAGS updates      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLVM Code Generation (with APX features)           │
│                                                                 │
│  • Target: x86-64 with +egpr,+ndd,+nf,+push2pop2               │
│  • Optimization: O3 with APX-aware register allocation         │
│  • Output: Native machine code using R16-R31                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Selective XSAVE Management                       │
│                                                                 │
│  • Track which R16-R31 registers were modified                 │
│  • Use XSAVEC with minimal mask (only APX component)           │
│  • Reduce XSAVE overhead from 4KB → ~128 bytes                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Code Cache + Execution                       │
│                                                                 │
│  • Cache APX-translated regions                                │
│  • Trampoline from original code to APX version                │
│  • XSAVE before entry, XRSTOR on exit                          │
└─────────────────────────────────────────────────────────────────┘
```

### Using the DBT

```cpp
#include "lifter/rellume_lifter.hpp"

using namespace bpftime::vm::apx;

// Create DBT instance
APXDBT dbt;

// Check if APX is available
if (dbt.is_apx_enabled()) {
    // Translate legacy code
    void* apx_code = dbt.translate(original_addr, code, code_size);

    // Execute with state management
    APXCpuState state = {};
    state.rip = original_addr;
    // ... initialize other registers ...

    uint64_t next_rip = dbt.execute(original_addr, state);
}

// Get statistics
auto stats = dbt.get_stats();
std::cout << "Translations: " << stats.translations << std::endl;
std::cout << "Cache hits: " << stats.cache_hits << std::endl;
std::cout << "APX optimizations: " << stats.apx_optimizations << std::endl;
```

### Selective XSAVE Integration

The module uses patterns from xsave-utils to manage APX register state:

```cpp
// Only save/restore R16-R31 that were modified
SelectiveXSaveManager xsave;

// Initialize per-thread state
void* buffer = xsave.init_thread_state(thread_id);

// Save only modified APX registers (e.g., R16-R19 = 0x000F)
xsave.save_apx_state(thread_id, modified_mask);

// Execute APX-optimized code
execute_apx_code();

// Restore APX state
xsave.restore_apx_state(thread_id, modified_mask);
```

The XSAVE mask is computed to include only the APX state component:
- Legacy area (x87, SSE): Skipped
- AVX/AVX-512: Only if used
- APX (R16-R31): Only if modified

## Files

```
vm/compat/apx-jit/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This documentation
├── apx_cpu_features.hpp        # APX detection header
├── apx_cpu_features.cpp        # APX detection implementation
├── compat_apx_llvm.hpp         # APX LLVM VM header
├── compat_apx_llvm.cpp         # APX LLVM VM implementation
├── apx_jit_context.hpp         # APX JIT context header
├── llvmbpf_apx_support.patch   # Patch for llvmbpf submodule
├── lifter/
│   ├── rellume_lifter.hpp      # Rellume integration header
│   └── rellume_lifter.cpp      # Rellume integration + APX passes
└── test/
    ├── CMakeLists.txt          # Test build configuration
    └── apx_feature_test.cpp    # Unit tests
```

## Requirements

- LLVM 15+ (LLVM 18+ recommended for full APX support)
- CMake 3.16+
- C++20 compiler
- x86-64 processor (APX detection on other architectures returns no support)

## Testing

```bash
# Build with tests
cmake -Bbuild -DBPFTIME_ENABLE_UNIT_TESTING=ON
cmake --build build

# Run APX feature tests
./build/vm/compat/apx-jit/test/apx_feature_test
```

## Integration with xsave-utils

This module leverages patterns from the xsave-utils repository:
- CPUID detection methodology
- XSAVE state analysis
- XCR0 verification

The xsave-utils probe tool can be used to verify APX support:

```bash
cd xsave-utils
xmake build
./build/xsave-area-probe
```

## References

- [Intel APX Architecture Specification](https://www.intel.com/content/www/us/en/developer/articles/technical/advanced-performance-extensions-apx.html)
- [LLVM X86 Backend](https://llvm.org/docs/CodeGenerator.html)
- [bpftime Project](https://github.com/eunomia-bpf/bpftime)
- [xsave-utils](../../../xsave-utils/)

## License

MIT License - See LICENSE file for details.
