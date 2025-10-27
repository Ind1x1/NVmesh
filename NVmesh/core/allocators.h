#pragma once

#include <map>
#include <cuda.h>
#include "kernel.cuh"
#include "common.h"
#include "error.h"

class MemoryAllocation {
public:
    void *ptr = nullptr;
    size_t allocationSize = 0;
    int CommRank = -1;
    static inline int uniqueIdCounter = 0;
    int uniqueId = 0;

    MemoryAllocation() {
        uniqueId = uniqueIdCounter;
        uniqueIdCounter++;
    }

    ~MemoryAllocation() {};

    static bool filter() {
        return true;
    }

    static std::string getName() {
        return "N/A";
    }

    virtual void memset(int value, CopyType copytype = COPY_TYPE_CE, MemoryPurpose memorypurpose = MEMORY_SOURCE);
    virtual unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MEMORY_SOURCE);
};

class DeviceMemoryAllocation : public MemoryAllocation {
public:
    DeviceMemoryAllocation(size_t _allocationSize, int _CommRank);
    ~DeviceMemoryAllocation();

    static std::string getName() {
        return "N/A";
    }
};

class HostMemoryAllocation : public MemoryAllocation {
public:
    HostMemoryAllocation(size_t _allocationSize, int _CommRank);
    ~HostMemoryAllocation();

    static std::string getName() {
        return "Host";
    }
};

class MultinodeMemoryAllocationBase : public MemoryAllocation {
private:
    CUmemGenericAllocationHandle handle = {};
    CUmemFabricHandle fh = {};
    CUmemAllocationHandleType handleType = {};
    CUmemAllocationProp prop = {};
    CUmemAccessDesc desc = {};
    size_t roundedUpAllocationSize;
    
public:
    MultinodeMemoryAllocationBase(size_t _allocationSize, int _CommRank, CUmemLocationType location);
    virtual ~MultinodeMemoryAllocationBase();
};

class MultinodeMemoryAllocationUnicast : public MultinodeMemoryAllocationBase {
public:
    MultinodeMemoryAllocationUnicast(size_t _allocationSize, int _CommRank) : MultinodeMemoryAllocationBase(_allocationSize, _CommRank, CU_MEM_LOCATION_TYPE_DEVICE) {};
    static bool filter();
    
    static std::string getName() {
        return "device";
    }
};
    
class MultinodeMemoryAllocationEGM : public MultinodeMemoryAllocationBase {
public:
    MultinodeMemoryAllocationEGM(size_t _allocationSize, int _CommRank) : MultinodeMemoryAllocationBase(_allocationSize, _CommRank, CU_MEM_LOCATION_TYPE_HOST_NUMA) {};
    static bool filter();
    
    static std::string getName() {
        return "egm";
    }
};

class MultinodeMemoryPoolAllocationBase : public MemoryAllocation {
private:
    static inline bool devicePoolsInitialized;
    static inline bool egmPoolsInitialized;
    static inline std::vector<CUmemoryPool> device_pools;
    static inline std::vector<CUmemoryPool> egm_pools;
    std::vector<CUmemFabricHandle> fh_vector;
    CUmemLocationType mem_location;
    CUmemPoolProps poolProps = { };
    CUmemAllocationHandleType handleType = {};
    CUmemPoolPtrExportData data;
    CUmemAccessDesc desc = {};
public:
    MultinodeMemoryPoolAllocationBase(size_t _allocationSize, int _CommRank, CUmemLocationType location);
    virtual ~MultinodeMemoryPoolAllocationBase();
    virtual std::vector<CUmemoryPool> initPoolsLazy();
};

class MultinodeMemoryPoolAllocationUnicast : public MultinodeMemoryPoolAllocationBase {
public:
    MultinodeMemoryPoolAllocationUnicast(size_t _allocationSize, int _CommRank) : MultinodeMemoryPoolAllocationBase(_allocationSize, _CommRank, CU_MEM_LOCATION_TYPE_DEVICE) {};
    static bool filter();
    
    static std::string getName() {
        return "device";
    }
};

class MultinodeMemoryPoolAllocationEGM : public MultinodeMemoryPoolAllocationBase {
public:
    MultinodeMemoryPoolAllocationEGM(size_t _allocationSize, int _CommRank) : MultinodeMemoryPoolAllocationBase(_allocationSize, _CommRank, CU_MEM_LOCATION_TYPE_HOST_NUMA) {};
    static bool filter();
    
    static std::string getName() {
        return "egm";
    }
};

template<class T>
class AllocationPool : public MemoryAllocation {
private:
    static inline std::multimap< std::pair<size_t, int>, T* > pool;
    T *current;
public:
    AllocationPool(size_t _allocationSize, int _CommRank);
    ~AllocationPool();

    static void clear();
    static bool filter();

    static std::string getName();

    void memset(int value, CopyType copyType = COPY_TYPE_CE, MemoryPurpose memoryPurpose = MemoryPurpose::MEMORY_SOURCE) {current->memset(value, copyType, memoryPurpose);}
    unsigned long long check(int value, CopyType copyType = COPY_TYPE_CE, int iterations = 0) {return current->check(value, copyType, iterations);}
};

template class AllocationPool<MultinodeMemoryAllocationUnicast>;
template class AllocationPool<MultinodeMemoryAllocationEGM>;
template class AllocationPool<HostMemoryAllocation>;

void clearAllocationPools();

class MemoryAdjustHelper {

};