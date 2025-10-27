#include "kernels.cuh"
#include "utils.h"
#include "error.h"
#include "common.h"

#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


// template<int N>
// __global__ void copyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src, size_t chunkSizeInElement, unsigned int totalThreadCount)
// {
//     unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
//     dst += from;
//     src += from;
//     unsigned long long chunkSizeModifyInElement = chunkSizeInElement / N;
//     unsigned long long chunkSizeleftInElement = chunkSizeInElement - chunkSizeModifyInElement * N;
//     for (unsigned long long i = 0; i < loopCount; i++) {
//         uint4 * cdst = dst;
//         uint4 * csrc = src;
//         for (unsigned int j = 0; j < chunkSizeModifyInElement; j++) {
//             int4 pipe[N];
//             #pragma unroll
//             for (unsigned int k = 0; k < N; k++) {
//                 pipe[k] = ld_nc_global(csrc);
//                 csrc += totalThreadCount;
//             }
//             #pragma unroll
//             for (unsigned int k = 0; k < N; k++) {
//                 st_nc_global(cdst, pipe[k]);
//                 cdst += totalThreadCount;
//             }
//         }
//         for (unsigned int j = 0; j < chunkSizeleftInElement; j++) {
//             st_nc_global(cdst, ld_nc_global(csrc));
//             cdst += totalThreadCount;
//             csrc += totalThreadCount;
//         }
//     }
// }

__global__ void copyKernel_fallback(unsigned long long loopCount, uint4 *dst, uint4 *src, size_t chunkSizeInElement, unsigned int totalThreadCount) {
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    dst += from;
    src += from;
    unsigned long long chunkSizeModifyInElement = chunkSizeInElement / totalThreadCount;
    for (unsigned int i = 0; i < loopCount; i++) {
        uint4* cdst = dst;
        uint4* csrc = src;
        for (unsigned int j = 0; j < chunkSizeModifyInElement; j++) {
            st_nc_global(cdst, ld_nc_global(csrc));
            cdst += totalThreadCount;
            csrc += totalThreadCount;
        }
    }
}
#define FOR_EACH_PIPES(M)  M(1) M(2) M(4) M(8) M(12) M(16) M(32)

// #    kernel type 1
// #    chunkSizeIntElement = Number of uint4 per totalThreadCount
// #    dst
// #    |---------------------------------------------------------| BufferSize
// #    chunk1        chunk2
// #    |-------------|-------------|......
// #      |             |
// #      cdst          cdst += totalThreadCount   

#define DEF_COPY(N)                                                                 \
__global__ void copyKernel_##N##pipes(unsigned long long loopCount, uint4 *dst, uint4 *src, size_t chunkSizeInElement, unsigned int totalThreadCount) { \
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;                \
    dst += from;                                                                    \
    src += from;                                                                    \
    unsigned long long chunkSizeModifyInElement =  chunkSizeInElement / N;          \
    unsigned long long chunkSizeleftInElement =  chunkSizeInElement - chunkSizeModifyInElement * N; \
    for (unsigned int i = 0; i < loopCount; i++) {                                  \
        uint4* cdst = dst;                                                          \
        uint4* csrc = src;                                                          \
        for (unsigned int j = 0; j < chunkSizeModifyInElement; j++) {               \
            int4 pipe[N];                                                           \
            #pragma unroll                                                          \
            for (int k = 0; k < N; k++) {                                           \
                pipe[k] = ld_nc_global(csrc);                                       \
                csrc += totalThreadCount;                                           \
            }                                                                       \
            #pragma unroll                                                          \
            for (int k = 0; k < N; k++) {                                           \
                st_nc_global(cdst, pipe[k]);                                        \
                cdst += totalThreadCount;                                           \
            }                                                                       \
        }                                                                           \
        for (unsigned int j = 0; j < chunkSizeleftInElement; j++) {                 \
            st_nc_global(cdst, ld_nc_global(csrc));                                 \
            cdst += totalThreadCount;                                               \
            csrc += totalThreadCount;                                               \
        }                                                                           \
    }                                                                               \
}
FOR_EACH_PIPES(DEF_COPY)





__global__ void spinKernelDevice(volatile int *latch, const unsigned long long timeoutClocks) {
    register unsigned long long endTime = _read_globaltimer() + timeoutClocks;
    while (!*latch) {
        if (timeoutClocks != ~0ULL && _read_globaltimer() > endTime) {
            break;
        }
    }
}

CUresult spinKernel(volatile int *latch, CUstream stream, unsigned long long timeoutMs) {
    int clocksPerMs = 0;
    CUcontext ctx;
    CUdevice dev;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    CU_ASSERT(cuDeviceGetAttribute(&clocksPerMs, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));

    unsigned long long timeoutClocks = clocksPerMs * timeoutMs;

    spinKernelDevice<<<1, 1, 0, stream>>>(latch, timeoutClocks);

    return CUDA_SUCCESS;
}

__global__ void spinKernelDeviceMultistage(volatile int *latch1, volatile int *latch2, const unsigned long long timeoutClocks) {
    if (latch1) {
        register unsigned long long endTime = clock64() + timeoutClocks;
        while (!*latch1) {
            if (timeoutClocks != ~0ULL && clock64() > endTime) {
                break;
            }
        }

        *latch2 = 1;
    }
    
    register unsigned long long endTime = clock64() + timeoutClocks;
    while (!*latch2) {
        if (timeoutClocks != ~0ULL && clock64() > endTime) {
            break;
        }
    }
}

CUresult spinKernelMultistage(volatile int *latch1, volatile int *latch2, CUstream stream, unsigned long long timeoutMs) {
    ASSERT(latch2 != nullptr);

    unsigned long long timeoutClocks = rank::getLocalClockRate() * timeoutMs;
    spinKernelDeviceMultistage<<<1, 1, 0, stream>>>(latch1, latch2, timeoutClocks);
    CUDA_ASSERT(cudaPeekAtLastError());

    return CUDA_SUCCESS;
}

__global__ void patternFillKernel(uint4* dst, int seed, size_t bufferSize, int groupId, int groupSize) {
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    size_t totalThreadCount = gridDim.x * blockDim.x;
    char* dstEnd = ((char *) dst) + bufferSize;
    dst += from;

    curandStateXORWOW_t state;
    curand_init(seed, 0, from, &state);

    for (int i = 0; i < groupSize; i++) {
        curand(&state);
    }

    while ((char *) dst < dstEnd) {
        *dst = curand(&state);
        dst += totalThreadCount;

        for (int i = 0; i < groupSize - 1; i++) {
            curand(&state);
        }
    }
}

void memsetBuffer(void *ptr, int seed, size_t size, CUstream stream, int groupId, int groupSize) {
    dim3 gridDim(rank::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    patternFillKernel<<<gridDim, blockDim, 0, stream>>>((uint *)ptr, seed, size, groupId, groupSize);
    CUDA_ASSERT(cudaPeekAtLastError());
}

void zeroOutBuffer(void *ptr, size_t size, CUstream stream) {
    CU_ASSERT(cuMemsetD8Async((CUdeviceptr) ptr, 0, size, stream));
}

void memsetBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, MemoryPurpose memoryPurpose) {
    // if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
    //     memsetBuffer(ptr, seed, size, stream, MPIWrapper::getWorldRank(), MPIWrapper::getWorldSize());
    // } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL) {
    //     if (memoryPurpose == MemoryPurpose::MEMORY_SOURCE) {
    //         memsetBuffer(ptr, seed, size, stream, MPIWrapper::getWorldRank(), MPIWrapper::getWorldSize());
    //     } else {
    //         zeroOutBuffer(ptr, size, stream);
    //     }
    // } else if (copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
    //     if (memoryPurpose == MemoryPurpose::MEMORY_SOURCE) {
    //         memsetBuffer(ptr, seed, size, stream, 0, 1);
    //     } else {
    //         zeroOutBuffer(ptr, size, stream);
    //     }
    // } else {
    //     memsetBuffer(ptr, seed, size, stream, 0, 1);
    // }
    memsetBuffer(ptr, seed, size, stream, 0, 1);
}

// 验证错误
__global__ void patternCheckKernel(uint* buffer, int seed, size_t bufferSize, unsigned long long *errorCount, int groupSize, int multiplier) {
    uint* originalBuffer = buffer;
    unsigned long long threadId = blockDim.x * blockIdx.x + threadIdx.x;
    size_t totalThreadCount = gridDim.x * blockDim.x;
    char* bufferEnd = ((char *) buffer) + bufferSize;
    buffer += threadId;

    curandStateXORWOW_t state;
    curand_init(seed, 0, threadId, &state);

    while ((char *) buffer < bufferEnd) {
        uint expectedValue = 0;

        for (int i = 0; i < groupSize; i++) {
            // overflow for uint is well defined
            expectedValue += curand(&state);
        }

        expectedValue *= multiplier;

        uint actualValue = *buffer;
        if (actualValue != expectedValue) {
            printf("Error found at byte offset %llu: expected %u but got %u\n", (char *) buffer - (char *) originalBuffer, expectedValue, actualValue);
            atomicAdd(errorCount, 1);
            // Only report one error per thread to avoid spamming prints
            break;
        }
        buffer += totalThreadCount;
    }
}

unsigned long long checkBuffer(void *ptr, int seed, size_t size, CUstream stream, int groupSize, int multiplier = 1) {
    unsigned long long *errorCount;
    CU_ASSERT(cuMemAlloc((CUdeviceptr *) &errorCount, sizeof(*errorCount)));
    CU_ASSERT(cuMemsetD8((CUdeviceptr) errorCount, 0, sizeof(*errorCount)));

    dim3 gridDim(NvLoom::getLocalMultiprocessorCount(), 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    patternCheckKernel<<<gridDim, blockDim, 0, stream>>>((uint *)ptr, seed, size, errorCount, groupSize, multiplier);
    CUDA_ASSERT(cudaPeekAtLastError());
    CU_ASSERT(cuStreamSynchronize(stream));

    unsigned long long errorCountCopy;
    CU_ASSERT(cuMemcpy((CUdeviceptr) &errorCountCopy, (CUdeviceptr) errorCount, sizeof(errorCountCopy)));

    CU_ASSERT(cuMemFree((CUdeviceptr) errorCount));

    return errorCountCopy;
}

unsigned long long checkBuffer(void *ptr, int seed, size_t size, CUstream stream, CopyType copyType, int iterations) {
    // if (copyType == COPY_TYPE_MULTICAST_LD_REDUCE) {
    //     return checkBuffer(ptr, seed, size, stream, MPIWrapper::getWorldSize(), 1);
    // } else if (copyType == COPY_TYPE_MULTICAST_RED_ALL) {
    //     return checkBuffer(ptr, seed, size, stream, MPIWrapper::getWorldSize(), iterations);
    // } else if (copyType == COPY_TYPE_MULTICAST_RED_SINGLE) {
    //     return checkBuffer(ptr, seed, size, stream, 1, iterations);
    // } else {
    //     return checkBuffer(ptr, seed, size, stream, 1);
    // }
    return checkBuffer(ptr, seed, size, stream, 1);
}

void preloadKernels(int localDevice) {
    cudaFuncAttributes unused;
    cudaSetDevice(localDevice);
    cudaFuncGetAttributes(&unused, &copyKernel_1pipes);
    cudaFuncGetAttributes(&unused, &copyKernel_2pipes);
    cudaFuncGetAttributes(&unused, &copyKernel_4pipes);
    cudaFuncGetAttributes(&unused, &copyKernel_8pipes);
    cudaFuncGetAttributes(&unused, &copyKernel_12pipes);
    cudaFuncGetAttributes(&unused, &copyKernel_16pipes);
    cudaFuncGetAttributes(&unused, &copyKernel_32pipes);
    cudaFuncGetAttributes(&unused, &copyKernel_fallback);
    cudaFuncGetAttributes(&unused, &spinKernelDevice);
    cudaFuncGetAttributes(&unused, &spinKernelDeviceMultistage);
    cudaFuncGetAttributes(&unused, &patternFillKernel);
    cudaFuncGetAttributes(&unused, &patternCheckKernel);
}