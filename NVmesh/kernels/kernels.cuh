#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "common.h"
#include "utils.h"
#include "mpi_helper.h"
#include "error.h"

const unsigned long long DEFAULT_SPIN_KERNEL_TIMEOUT_MS = 10000ULL;
const int numThreadPerBlock = 256;

// Forward declarations for copy kernels
__global__ void copyKernel_fallback(unsigned long long loopCount, uint4 *dst, uint4 *src, size_t chunkSizeInElement, unsigned int totalThreadCount);

// Declare all copy kernel variants using macro
#define FOR_EACH_PIPES(M)  M(1) M(2) M(4) M(8) M(12) M(16) M(32)
#define DECL_COPY(N) \
    __global__ void copyKernel_##N##pipes(unsigned long long loopCount, uint4 *dst, uint4 *src, size_t chunkSizeInElement, unsigned int totalThreadCount);
FOR_EACH_PIPES(DECL_COPY)
#undef DECL_COPY
#undef FOR_EACH_PIPES

// Spin kernel declarations
__global__ void spinKernelDevice(volatile int *latch, const unsigned long long timeoutClocks);
CUresult spinKernel(volatile int *latch, CUstream stream, unsigned long long timeoutMs);

__global__ void spinKernelDeviceMultistage(volatile int *latch1, volatile int *latch2, const unsigned long long timeoutClocks);
CUresult spinKernelMultistage(volatile int *latch1, volatile int *latch2, CUstream stream, unsigned long long timeoutMs);

// Pattern fill kernel
__global__ void patternFillKernel(uint4* dst, int seed, size_t bufferSize, int groupId, int groupSize);

// Pattern check kernel
__global__ void patternCheckKernel(uint* buffer, int seed, size_t bufferSize, unsigned long long *errorCount, int groupSize, int multiplier);

// Host function declarations
void memsetBuffer(void *ptr, int seed, size_t size, CUstream stream, int groupId, int groupSize);
void zeroOutBuffer(void *ptr, size_t size, CUstream stream);
void memsetBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, MemoryPurpose memoryPurpose);

unsigned long long checkBuffer(void *ptr, int seed, size_t size, CUstream stream, int groupSize, int multiplier = 1);
unsigned long long checkBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, int iterations);

void preloadKernels(int localDevice);

// Macro for launching copy kernels
#define LaunchCopyKernel(N, dst, src, stream, loopCount, chunkSizeInElement, totalThreadCount, numSm, blockX)       \
    dim3 block(blockX, 1, 1);                                                                                       \
    dim3 grid(numSm, 1, 1);                                                                                         \
    do {                                                                                                            \
        if (__builtin_constant_p(N)) {                                                                              \
            switch (N) {                                                                                            \
                case 1:  copyKernel_1pipes<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;       \
                case 2:  copyKernel_2pipes<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;       \
                case 4:  copyKernel_4pipes<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;       \
                case 8:  copyKernel_8pipes<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;       \
                case 12: copyKernel_12pipes<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;      \
                case 16: copyKernel_16pipes<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;      \
                case 32: copyKernel_32pipes<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;      \
                default: copyKernel_fallback<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount); break;                                           \
            }                                                                                                       \
        } else {                                                                                                    \
            copyKernel_fallback<<<grid, block, 0, stream>>>(loopCount, (uint4*)dst, (uint4*)src, chunkSizeInElement, totalThreadCount);                   \
        }                                                                                                           \
    } while (0)