#pragma once


#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                             \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          \
        auto __src = (SRC);                                                                                                           \
        auto __dst = (DST);                                                                                                           \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);  \
        }                                                                                                                             \
        {                                                                                                                             \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                                                           \
                }                                                                                                                     \
            }                                                                                                                         \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                                                            \
                }                                                                                                                     \
            }                                                                                                                         \
        }                                                                                                                             \
    }


__device__ __forceinline__ void memory_fence() {
    asm volatile("fence.acq_rel.sys;":: : "memory");
}

__device__ __forceinline__ void memory_fence_gpu() {
    asm volatile("fence.acq_rel.gpu;":: : "memory");
}

__device__ __forceinline__ void memory_fence_cpu() {
    asm volatile("fence.acq_rel.cpu;":: : "memory");
}

// int4 load/store

__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
    int4 ret;
    asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr) : "memory");
    return ret;
}

__device__ __forceinline__ void st_nc_global(const int4 *ptr, const int4& value) {
    asm volatile("st.volatile.global.v4.s32 [%0], {%1,%2,%3,%4};\n"
    : : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

//OPTIMIZE
/*
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif
*/

__device__ __forceinline__ uint64_t _read_globaltimer() {
    uint64_t timer;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
    return timer;
}

