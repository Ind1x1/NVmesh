#pragma once

#include <locale>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <memory>
#include <cuda.h>

constexpr unsigned long long DEFAULT_ITERATIONS = 16;

enum CopyDirection {
    COPY_DIRECTION_WRITE = 0,
    COPY_DIRECTION_READ,
};

enum CopyType {
    COPY_TYPE_CE = 0,
    COPY_TYPE_SM,
    COPY_TYPE_TMA,
    
    COPY_TYPE_MULTICAST_LD_REDUCE,
    COPY_TYPE_MULTICAST_RED_ALL,
    COPY_TYPE_MULTICAST_RED_SINGLE,
};

enum MemoryPurpose {
    MEMORY_SOURCE = 0,
    MEMORY_DESTINATION,
};

// struct rank {
// private:
//     static inline int localDevice;
//     static inline CUdevice localCuDevice;
//     static inline CUcontext localCtx;
//     static inline int localMultiprocessorCount;
//     static inline int localClockRate;
//     static inline int localCpuNumaNode;
//     static inline std::string localHostName;
//     // static inline std::map<std::string, std::vector<int> > rackToProcessMap;
// public:
//     static void setLocalDevice(int _localDevice) { localDevice = _localDevice; };
//     static void setLocalCuDevice(CUdevice _localCuDevice) { localCuDevice = _localCuDevice; };
//     static void setLocalCtx(CUcontext _localCtx) { localCtx = _localCtx; };
//     static void setLocalMultiprocessorCount(int _localMultiprocessorCount) { localMultiprocessorCount = _localMultiprocessorCount; };
//     static void setLocalClockRate(int _localClockRate) { localClockRate = _localClockRate; };
//     static void setLocalCpuNumaNode(int _localCpuNumaNode) { localCpuNumaNode = _localCpuNumaNode; };
//     static void setLocalHostName(std::string _localHostName) { localHostName = _localHostName; };

//     static int getLocalDevice() { return localDevice; };
//     static CUdevice getLocalCuDevice() { return localCuDevice; };
//     static CUcontext getLocalCtx() { return localCtx; };
//     static int getLocalMultiprocessorCount() { return localMultiprocessorCount; };
//     static int getLocalClockRate() { return localClockRate; };
//     static int getLocalCpuNumaNode() { return localCpuNumaNode; };
//     static std::string getLocalHostName() { return localHostName; };
//     // static std::map<std::string, std::vector<int> > getRackToProcessMap() { return rackToProcessMap; };
// };