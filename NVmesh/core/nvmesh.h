#pragma once

#include <string>
#include <vector>
#include <map>
#include <cuda.h>
#include <mpi.h>
#include <memory>
#include "allocators.h"
#include "kernels.cuh"
#include "utils.h"
#include "error.h"
#include "common.h"
#include "mpi_helper.h"

constexpr unsigned long long WARMUP_ITERATIONS = 1;

// struct ICommBackend {
//     virtual ~ICommBackend() = default;
//     virtual int worldSize() const = 0;
//     virtual int worldRank() const = 0;
// };

// //OPTIMIZE EDIT THIS
// class MPIBackend : public ICommBackend {
// public:
//         int worldSize() const override { return MPIWrapper::getWorldSize(); }
//         int worldRank() const override { return MPIWrapper::getWorldRank(); }
// };

// class Comm {
// public:
//     static int getWorldSize() { return instance().impl_->worldSize(); }
//     static int getWorldRank() { return instance().impl_->worldRank(); }

//     static int worldSize() { return getWorldSize(); }
//     static int worldRank() { return getWorldRank(); }

//     static void useMPI() { instance().impl_ = std::make_unique<MPIBackend>(); }
//     // static void useshuttle() { instance().impl_ = std::make_unique<ShuttleBackend>(); }

//     static void install(std::unique_ptr<ICommBackend> backend) { 
//         instance().impl_ = std::move(backend); 
//     }

// private:
//     Comm() {
//         impl_ = std::make_unique<MPIBackend>();
//     }

//     static Comm& instance() {
//         static Comm self;
//         return self;
//     }

//     std::unique_ptr<ICommBackend> impl_;
// };

class Copy {
public:
    std::shared_ptr<MemoryAllocation> src;
    std::shared_ptr<MemoryAllocation> dst;
    CopyType copyType;
    CopyDirection copyDirection;

    int iterations;
    int executingMPIRank;

    Copy(std::shared_ptr<MemoryAllocation> _dst, std::shared_ptr<MemoryAllocation> _src, CopyDirection _copyDirection, CopyType _copyType, int _iterations = DEFAULT_ITERATIONS):
        dst(_dst), 
        src(_src), 
        copyDirection(_copyDirection), 
        copyType(_copyType), 
        iterations(_iterations) {
        if (copyDirection == COPY_DIRECTION_WRITE) {
            executingMPIRank = dst->MPIRank;
        } else {
            executingMPIRank = src->MPIRank;
        }
    }
};

class nvmesh {
private:
    //static struct rank rank;
    static inline int localDevice;
    static inline CUdevice localCuDevice;
    static inline CUcontext localCtx;
    static inline int localMultiprocessorCount;
    static inline int localClockRate;
    static inline int localCpuNumaNode;
    static inline std::string localHostname;
    static inline std::map<std::string, std::vector<int>> rackToProcessMap;

    static void doMemcpy(CopyType copyType, CUdeviceptr src, CUdeviceptr dst, size_t byteCount, CUstream hStream, unsigned long long loopCount);

    static unsigned long long doMemcpyInSpinKernel(CopyType copyType, CUdeviceptr src, CUdeviceptr dst, size_t byteCount, CUstream hStream, unsigned long long loopCount);

    static void doMemcpyBeyondSpinKernel(CopyType copyType, CUdeviceptr src, CUdeviceptr dst, size_t byteCount, CUstream hStream);

    // Launch up to loopCount iterations, using GPU clock as the timer
    static void doMemcpyDeviceClockKernel(CopyType copyType, CUdeviceptr src, CUdeviceptr dst, size_t byteCount, CUstream hStream, unsigned long long loopCount);

public:
    static std::vector<double> doBenchmark(std::vector<Copy> copies);
    static std::vector<double> doSingleText(std::vector<Copy> copies);
    //FIXME struct rank
    static int getLocalDevice() { return localDevice; };
    static CUdevice getLocalCuDevice() { return localCuDevice; };
    static CUcontext getLocalCtx() { return localCtx; };
    static int getLocalMultiprocessorCount() { return localMultiprocessorCount; };
    static int getLocalClockRate() { return localClockRate; };
    static int getLocalCpuNumaNode() { return localCpuNumaNode; };
    static std::string getLocalHostname() { return localHostname; };


    //FIXME
    static std::map<std::string, std::vector<int>> getRackToProcessMap() { return rackToProcessMap; };
    static void initialize(int _localDevice, std::map<std::string, std::vector<int>> _rackToProcessMap = {});
    static void finalize();
};

