
#include <cuda.h>
#include <chrono>
#include <thread>

#include "allocators.h"
#include "error.h"
#include "common.h"
#include "nvmesh.h"
#include "kernels.cuh"

inline unsigned int getCuMemFlagByInt(int isGB200) {
    return (isGB200 == 1) ? CU_MEM_HANDLE_TYPE_FABRIC : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

unsigned long long MemoryAllocation::check(int value, CopyType copyType, int iterations) {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        return checkBuffer(ptr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, iterations);
    }
    return 0;
}

void MemoryAllocation::memset(int value, CopyType copyType, MemoryPurpose memoryPurpose) {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        memsetBuffer(ptr, value, allocationSize, CU_STREAM_PER_THREAD, copyType, memoryPurpose);
    }
}

DeviceMemoryAllocation::DeviceMemoryAllocation(size_t _allocationSize, int _MPIrank) {
    MPIrank = _MPIrank;
    if (MPIrank == MPIWrapper::getWorldRank()) {
        allocationSize = _allocationSize;
        CU_ASSERT(cuMemAlloc((CUdeviceptr *) &ptr, _allocationSize));
    }
}

DeviceMemoryAllocation::~DeviceMemoryAllocation() {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFree((CUdeviceptr) ptr));
    }
}

HostMemoryAllocation::HostMemoryAllocation(size_t _allocationSize, int _MPIrank) {
    MPIrank = _MPIrank;
    if (MPIrank == MPIWrapper::getWorldRank()) {
        allocationSize = _allocationSize;
        CU_ASSERT(cuMemAllocHost(&ptr, _allocationSize));
    }
}

HostMemoryAllocation::~HostMemoryAllocation() {
    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFreeHost(ptr));
    }
}

static size_t roundUp(size_t number, size_t multiple) {
    return ((number + multiple - 1) / multiple) * multiple;
}

MultinodeMemoryAllocationBase::MultinodeMemoryAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location) {
    handleType = CU_MEM_HANDLE_TYPE_FABRIC:
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = handleType;

    prop.requestedHandleTypes = handleType;
    prop.location = location;
    if (location == CU_MEM_LOCATION_TYPE_DEVICE) {
        prop.location.id = nvmesh::getLocalDevice();
    } else {
        prop.location.id = nvmesh::getLocalHostname(); 
    }
    
    //Get the recommended allocation granularity
    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));  

    roundedUpAllocationSize = roundUp(_allocationSize, granularity);

    if (_MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemCreate(&handle, roundedUpAllocationSize, &prop, 0 /*flags*/));

        CU_ASSERT(cuMemExportToShareableHandle(&fh, handle, handleType, 0 /*flags*/));
    }

    //FIXME
    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);
    
    if (_MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemImportFromShareableHandle(&handle, fh, handleType, 0 /*flags*/));
    }

    // Map the memory
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &ptr, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));

    CU_ASSERT(cuMemMap((CUdeviceptr) ptr, roundedUpAllocationSize, 0 /*offset*/, handle, 0 /*flags*/));

    // Map EGM memory on host on the exporting node
    if ((_MPIrank == MPIWrapper::getWorldRank()) && (location == CU_MEM_LOCATION_TYPE_HOST_NUMA)) {
        desc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        desc.location.id = 0;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT(cuMemSetAccess((CUdeviceptr) ptr, roundedUpAllocationSize, &desc, 1 /*count*/));
    }

    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = nvmesh::getLocalDevice();
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) ptr, roundedUpAllocationSize, &desc, 1 /*count*/));

    MPIrank = _MPIrank;
    allocationSize = _allocationSize;

    // Make sure that everyone is done with mapping the fabric allocation
    //FIXME MPI
    MPI_Barrier(MPI_COMM_WORLD);
}

MultinodeMemoryAllocationBase::~MultinodeMemoryAllocationBase() {
    // Make sure that everyone is done using the memory
    //FIXME MPI
    MPI_Barrier(MPI_COMM_WORLD);

    CU_ASSERT(cuMemUnmap((CUdeviceptr) ptr, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) ptr, roundedUpAllocationSize));
}

bool MultinodeMemoryAllocationUnicast::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, nvmesh::getLocalCuDevice()));
    return pi != 0;
};

bool MultinodeMemoryAllocationEGM::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, nvmesh::getLocalCuDevice()));
    return pi != 0;
};


MultinodeMemoryPoolAllocationBase::MultinodeMemoryPoolAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;
    mem_location = location;

    if (!devicePoolsInitialized && mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        devicePoolsInitialized = true;
        device_pools.resize((MPIWrapper::getWorldSize()));
        device_pools = initPoolsLazy();
    }

    if (!egmPoolsInitialized && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        egmPoolsInitialized = true;
        egm_pools.resize((MPIWrapper::getWorldSize()));
        egm_pools = initPoolsLazy();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (_MPIrank == MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&ptr, allocationSize, egm_pools[_MPIrank], CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        CU_ASSERT(cuMemPoolExportPointer(&data, (CUdeviceptr) ptr));
    }
    if (_MPIrank == MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&ptr, allocationSize, device_pools[_MPIrank], CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        CU_ASSERT(cuMemPoolExportPointer(&data, (CUdeviceptr) ptr));
    }

    //FIXME MPI
    MPI_Bcast(&data, sizeof(data), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        CU_ASSERT(cuMemPoolImportPointer((CUdeviceptr*) &ptr, egm_pools[_MPIrank], &data));
    }
    if (_MPIrank != MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_DEVICE){
        CU_ASSERT(cuMemPoolImportPointer((CUdeviceptr*) &ptr, device_pools[_MPIrank], &data));
    }

    //FIXME MPI
    MPI_Barrier(MPI_COMM_WORLD);
}

std::vector<CUmemoryPool> MultinodeMemoryPoolAllocationBase::initPoolsLazy() {
    std::vector<CUmemoryPool> pools;
    pools.resize(MPIWrapper::getWorldSize());
    fh_vector.resize(MPIWrapper::getWorldSize());
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    poolProps.handleTypes = handleType;
    poolProps.location.type = mem_location;
    cuuint64_t thresholdSize = 1073741824; // 1 GB in bytes;

    if (mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        poolProps.location.id = nvmesh::getLocalDevice();
    } else {
        poolProps.location.id = nvmesh::getLocalCpuNumaNode();
    }

    CU_ASSERT(cuMemPoolCreate(&pools[MPIWrapper::getWorldRank()], &poolProps));

    // Set pool release threshold to reserve 1GB before it releases memory back to the OS - Pending investigation @ https://gitlab-master.nvidia.com/dcse-appsys/nvloom/-/issues/24
    CU_ASSERT(cuMemPoolSetAttribute(pools[MPIWrapper::getWorldRank()], CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &thresholdSize));
    CU_ASSERT(cuMemPoolExportToShareableHandle(&fh_vector[MPIWrapper::getWorldRank()], pools[MPIWrapper::getWorldRank()], handleType, 0 /*flags*/));
    //FIXME MPI
    MPI_Barrier(MPI_COMM_WORLD);

    // Import handles of other pools to fill vector
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        //FIXME MPI
        MPI_Bcast(&fh_vector[i], sizeof(CUmemFabricHandle), MPI_BYTE, i, MPI_COMM_WORLD);
        if (i != MPIWrapper::getWorldRank()) {
            CU_ASSERT(cuMemPoolImportFromShareableHandle(&pools[i], (void *)&fh_vector[i], handleType, 0));
        }
    }

    // Set access for the pools
    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
            desc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
            desc.location.id = nvmesh::getLocalCpuNumaNode();
            desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_ASSERT(cuMemPoolSetAccess(pools[i], &desc, 1));
        }
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = nvmesh::getLocalCuDevice();
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_ASSERT(cuMemPoolSetAccess(pools[i], &desc, 1));
    }
    return pools;
}

MultinodeMemoryPoolAllocationBase::MultinodeMemoryPoolAllocationBase(size_t _allocationSize, int _MPIrank, CUmemLocationType location) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;
    mem_location = location;

    if (!devicePoolsInitialized && mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        devicePoolsInitialized = true;
        device_pools.resize((MPIWrapper::getWorldSize()));
        device_pools = initPoolsLazy();
    }

    if (!egmPoolsInitialized && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        egmPoolsInitialized = true;
        egm_pools.resize((MPIWrapper::getWorldSize()));
        egm_pools = initPoolsLazy();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (_MPIrank == MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&ptr, allocationSize, egm_pools[_MPIrank], CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        CU_ASSERT(cuMemPoolExportPointer(&data, (CUdeviceptr) ptr));
    }
    if (_MPIrank == MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_DEVICE) {
        CU_ASSERT(cuMemAllocFromPoolAsync((CUdeviceptr *)&ptr, allocationSize, device_pools[_MPIrank], CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
        CU_ASSERT(cuMemPoolExportPointer(&data, (CUdeviceptr) ptr));
    }

    MPI_Bcast(&data, sizeof(data), MPI_BYTE, _MPIrank, MPI_COMM_WORLD);

    if (_MPIrank != MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        CU_ASSERT(cuMemPoolImportPointer((CUdeviceptr*) &ptr, egm_pools[_MPIrank], &data));
    }
    if (_MPIrank != MPIWrapper::getWorldRank() && mem_location == CU_MEM_LOCATION_TYPE_DEVICE){
        CU_ASSERT(cuMemPoolImportPointer((CUdeviceptr*) &ptr, device_pools[_MPIrank], &data));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

MultinodeMemoryPoolAllocationBase::~MultinodeMemoryPoolAllocationBase() {
    CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));

    if (MPIrank != MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFreeAsync((CUdeviceptr) ptr, CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (MPIrank == MPIWrapper::getWorldRank()) {
        CU_ASSERT(cuMemFreeAsync((CUdeviceptr) ptr, CU_STREAM_PER_THREAD));
        CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

bool MultinodeMemoryPoolAllocationUnicast::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, nvmesh::getLocalCuDevice()));
    return pi != 0;
};

bool MultinodeMemoryPoolAllocationEGM::filter() {
    int pi;
    CU_ASSERT(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED, nvmesh::getLocalCuDevice()));
    return pi != 0;
};

template<class T>
AllocationPool<T>::AllocationPool(size_t _allocationSize, int _MPIrank) {
    allocationSize = _allocationSize;
    MPIrank = _MPIrank;

    auto key = std::make_pair(allocationSize, MPIrank);
    auto it = pool.find(key);
    if (it == pool.end()) {
        current = new T(_allocationSize, _MPIrank);
    } else {
        current = it->second;
        pool.erase(it);
    }

    ptr = current->ptr;
}

template<class T>
AllocationPool<T>::~AllocationPool() {
    auto key = std::make_pair(allocationSize, MPIrank);
    pool.insert({key, current});
}

template<class T>
void AllocationPool<T>::clear() {
    for (const auto& elem : pool) {
        delete elem.second;
    }
    pool.clear();
}

template<class T>
bool AllocationPool<T>::filter() {
    return T::filter();
}

template<class T>
std::string AllocationPool<T>::getName() {
    return T::getName();
}

void clearAllocationPools() {
    AllocationPool<MultinodeMemoryAllocationUnicast>::clear();
    AllocationPool<MultinodeMemoryAllocationEGM>::clear();
    AllocationPool<HostMemoryAllocation>::clear();
}