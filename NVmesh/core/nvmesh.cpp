#include <iostream>
#include <iomanip>
#include <string.h>
#include <unistd.h>
#include <set>
#include "error.h"
#include "kernels.cuh"
#include "nvmesh.h"

#include <cuda.h>
#include <nvml.h>

MPIOutput Output;

auto getCopiesWithUniqueSources(std::vector<Copy> copies) {
    auto comp = [](Copy a, Copy b) {return a.src.get() < b.src.get();};
    std::set<Copy, decltype(comp) > uniqueSources(comp);
    for (auto &copy : copies) {
        uniqueSources.insert(copy);
    }
    return uniqueSources;
}

auto getCopiesWithUniqueDestinations(std::vector<Copy> copies) {
    auto comp = [](Copy a, Copy b) {return a.dst.get() < b.dst.get();};
    std::set<Copy, decltype(comp) > uniqueDestinations(comp);
    for (auto &copy : copies) {
        uniqueDestinations.insert(copy);
    }
    return uniqueDestinations;
}

//TODO MAIN FUNCTION
std::vector<double> nvmesh::doBenchmark(std::vector<Copy> copies) {
    if (copies.size() == 0) { return {}; };

    std::vector<double> results;

    std::vector<Copy> filteredCopies;
    for (auto copy : copies) {
        if (MPIWrapper::getWorldRank() == copy.executingRank) {
            filteredCopies.push_back(copy);
        }
    }

    std::vector<CUstream> filteredStreams(filteredCopies.size());
    std::vector<CUevent> filteredStartEvents(filteredCopies.size());
    std::vector<CUevent> filteredEndEvents(filteredCopies.size());
    std::vector<double> filteredBandwidths(filteredCopies.size());
    std::vector<unsigned long long> filterExecutedIterations(filteredCopies.size());

    AllocationPool<HostMemoryAllocation> blockingVarHost(sizeof(int), copies[0].executingRank);
    AllocationPool<MultinodeMemoryAllocationUnicast> blockingVarDevice(sizeof(int), copies[0].executingRank);

    // init 
    if (MPIWrapper::getWorldRank() == copies[0].executingRank) {
        *((int *) (blockingVarHost.ptr)) = 0;
        CU_ASSERT(cuMemsetD32((CUdeviceptr) blockingVarDevice.ptr, 0, 1));
    }

    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(cuStreamCreate(&filteredStreams[i], CU_STREAM_NON_BLOCKING));

        CU_ASSERT(cuEventCreate(&filteredStartEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&filteredEndEvents[i], CU_EVENT_DEFAULT));
    }

    for (auto &copy : getCopiesWithUniqueSources(copies)) {
        copy.src->memset(copy.src->uniqueId, copy.copyType, MemoryPurpose::MEMORY_SOURCE);
    }

    for (auto &copy : getCopiesWithUniqueDestinations(copies)) {
        copy.dst->memset(copy.dst->uniqueId, copy.copyType, MemoryPurpose::MEMORY_DESTINATION);
    }

    CU_ASSERT(cuCtxSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // warmup
    for (int i = 0; i < filteredCopies.size(); i++) {
        Copy &copy = filteredCopies[i];
        doMemcpy(copy.copyType, (CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, filteredStreams[i], WARMUP_ITERATIONS);
    }

    // block streams
    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(spinKernelMultistage((MPIWrapper::getWorldRank() == copies[0].executingRank) ? (volatile int *) blockingVarHost.ptr : nullptr,
                                        (volatile int *) blockingVarDevice.ptr,
                                        filteredStreams[i]));
    }

    // schedule work
    for (int i = 0; i < filteredCopies.size(); i++) {
        Copy &copy = filteredCopies[i];
        CU_ASSERT(cuEventRecord(filteredStartEvents[i], filteredStreams[i]));
        filteredExecutedIterations[i] = doMemcpyInSpinKernel(copy.copyType, (CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, filteredStreams[i], copy.iterations);
        if (filteredExecutedIterations[i] == copy.iterations) {
            CU_ASSERT(cuEventRecord(filteredEndEvents[i], filteredStreams[i]));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // release the barrier
    if (MPIWrapper::getWorldRank() == copies[0].executingRank) {
        *((int *) (blockingVarHost.ptr)) = 1;
    }

    int maxIterations = 0;
    for (auto &copy : filteredCopies) {
        maxIterations = std::max(maxIterations, copy.iterations);
    }

    // keep adding copies to streams, one at a time, in a round robin fashion
    bool pendingWork = true;
    while (pendingWork) {
        pendingWork = false;
        for (int i = 0; i < filteredCopies.size(); i++) {
            Copy &copy = filteredCopies[i];
            if (filteredExecutedIterations[i] < copy.iterations) {
                doMemcpyBeyondSpinKernel(copy.copyType, (CUdeviceptr) copy.dst->ptr, (CUdeviceptr) copy.src->ptr, copy.src->allocationSize, filteredStreams[i]);
                filteredExecutedIterations[i]++;
                pendingWork = true;
            }
            if (filteredExecutedIterations[i] == copy.iterations) {
                CU_ASSERT(cuEventRecord(filteredEndEvents[i], filteredStreams[i]));
            }
        }
    }

    // Make sure all the work
    CU_ASSERT(cuCtxSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // calculate bandwidths
    for (int i = 0; i < filteredCopies.size(); i++) {
        float elapsedMs;
        CU_ASSERT(cuEventElapsedTime(&elapsedMs, filteredStartEvents[i], filteredEndEvents[i]));

        filteredBandwidths[i] = filteredCopies[i].src->allocationSize * filteredCopies[i].iterations / (1000 * 1000 * elapsedMs);
    }

    // exchange bandwidths
    int currentIndex = 0;
    for (auto copy : copies) {
        double exchange;
        if (MPIWrapper::getWorldRank() == copy.executingRank) {
            exchange = filteredBandwidths[currentIndex];
            currentIndex++;
        }

        MPI_Bcast(&exchange, 1, MPI_DOUBLE, copy.executingRank, MPI_COMM_WORLD);
        results.push_back(exchange);
    }

    // verify buffers
    for (auto &copy: getCopiesWithUniqueDestinations(copies)) {
        ASSERT(0 == copy.dst->check(copy.src->uniqueId, copy.copyType, WARMUP_ITERATIONS + copy.iterations));
    }

    for (int i = 0; i < filteredCopies.size(); i++) {
        CU_ASSERT(cuStreamDestroy(filteredStreams[i]));
        CU_ASSERT(cuEventDestroy(filteredStartEvents[i]));
        CU_ASSERT(cuEventDestroy(filteredEndEvents[i]));
    }

    return results;
}

void nvmesh::doMemcpy(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream, unsigned long long loopCount) {
    if (copyType == COPY_TYPE_CE) {
        for (int iter = 0; iter < loopCount; iter++) {
            CU_ASSERT(cuMemcpyAsync(dst, src, byteCount, hStream));
        }
    } else if (copyType == COPY_TYPE_SM) {
        LaunchCopyKernel(16, dst, src, hStream, loopCount, byteCount, 256, 32, 8); //FIXME
    } else {
        ASSERT(0);
    }
}

unsigned long long nvmesh::doMemcpyInSpinKernel(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream, unsigned long long loopCount) {
    if (copyType == COPY_TYPE_CE ) {
        loopCount = std::min(loopCount, (unsigned long long) 128);
        for (int iter = 0; iter < loopCount; iter++) {
            CU_ASSERT(cuMemcpyAsync(dst, src, byteCount, hStream));
        }
    } else if (copyType == COPY_TYPE_SM) {
        LaunchCopyKernel(16, dst, src, hStream, loopCount, byteCount, 256, 32, 8); //FIXME
    } else {
        ASSERT(0);
    }

    return loopCount;
}

void nvmesh::doMemcpyBeyondSpinKernel(CopyType copyType, CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream) {
    if (copyType == COPY_TYPE_CE) {
        CU_ASSERT(cuMemcpyAsync(dst, src, byteCount, hStream));
    } 
}

static std::string getHostName() {
#define HOSTNAME_LENGTH 128
    char _hostname[HOSTNAME_LENGTH] = {};
    gethostname(_hostname, HOSTNAME_LENGTH -1);
#undef HOSTNAME_LENGTH
    return std::string(_hostname);
}

void nvmesh::initialize(int _localDevice, std::map<std::string, std::vector<int>> _rackToProcessMap) {
    localDevice = _localDevice;
    rackToProcessMap = _rackToProcessMap;

    CU_ASSERT(cuInit(0));
    CU_ASSERT(cuDeviceGet(&localCuDevice, localDevice));
    CU_ASSERT(cuDevicePrimaryCtxRetain(&localCtx, localCuDevice));
    CU_ASSERT(cuCtxSetCurrent(localCtx));

    CU_ASSERT(cuDeviceGetAttribute(&localCpuNumaNode, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, localCuDevice));
    CU_ASSERT(cuDeviceGetAttribute(&localMultiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, localCuDevice));
    CU_ASSERT(cuDeviceGetAttribute(&localClockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, localCuDevice));

    localHostName = getHostName();

    preloadKernels(localDevice);
    MPIWrapper::getWorldSize();
    checkCliques();
}

void nvmesh::finalize() {
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Finalize();
    }
}

// Utility function to convert CUDA format to NVML format
static std::string convertUuid(CUuuid& cuUuid)
{
    uint8_t *uuidPtr = (uint8_t *) &cuUuid;
    std::stringstream s;
    s << "GPU-";
    for (int i = 0; i < sizeof(CUuuid); i++) {
        s << std::setfill('0') << std::setw(2) << std::hex << (int) uuidPtr[i];
        if (std::set<int>{3,5,7,9}.count(i)) {
            s << "-";
        }
    }
    return s.str();
}

static nvmlDevice_t getNvmlDevice(int device) {
    NVML_ASSERT(nvmlInit());
    CU_ASSERT(cuInit(0));

    CUuuid uuid;
    CU_ASSERT(cuDeviceGetUuid_v2(&uuid, device));

    std::string nvmlUuid = convertUuid(uuid);

    nvmlDevice_t nvmlDev;
    NVML_ASSERT(nvmlDeviceGetHandleByUUID((const char *) nvmlUuid.c_str(), &nvmlDev));
    return nvmlDev;
}

static int checkCliques() {
    nvmlDevice_t nvmlDev = getNvmlDevice(nvmesh::getLocalDevice());

    nvmlGpuFabricInfoV_t fabricInfo { .version = nvmlGpuFabricInfo_v2 };
    fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    NVML_ASSERT(nvmlDeviceGetGpuFabricInfoV(nvmlDev, &fabricInfo));

    // allowing running without MNNVL for development purposes
    if (fabricInfo.state == NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
        OUTPUT << "WARNING: MNNVL fabric not available. Only single node operation available." << std::endl;
    }

    auto cliqueIdArray = std::vector<unsigned int>(MPIWrapper::getWorldSize());
    MPI_Allgather(&fabricInfo.cliqueId, 1, MPI_UNSIGNED, &cliqueIdArray[0], 1, MPI_UNSIGNED, MPI_COMM_WORLD);

    auto clusterUuidArray = std::string(MPIWrapper::getWorldSize() * NVML_GPU_FABRIC_UUID_LEN, 0);
    MPI_Allgather(&fabricInfo.clusterUuid, NVML_GPU_FABRIC_UUID_LEN, MPI_UNSIGNED_CHAR, &clusterUuidArray[0], NVML_GPU_FABRIC_UUID_LEN, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

    for (int i = 0; i < MPIWrapper::getWorldSize(); i++) {
        if (0 != memcmp(&fabricInfo.clusterUuid, &clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN], NVML_GPU_FABRIC_UUID_LEN)) {
            std::cerr << "Process " << MPIWrapper::getWorldRank() << " clusterUuid=" << ((unsigned long *)&fabricInfo.clusterUuid)[0] << ";" << ((unsigned long *)&fabricInfo.clusterUuid)[1] <<
                " is different than process " << i << " clusterUuid=" << ((unsigned long *)&clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN])[0] << ";" <<  ((unsigned long *)&clusterUuidArray[i * NVML_GPU_FABRIC_UUID_LEN])[1] << std::endl;
            ASSERT(0);
        }

        if (cliqueIdArray[i] != fabricInfo.cliqueId) {
            std::cerr << "Process " << MPIWrapper::getWorldRank() << " cliqueId=" << fabricInfo.cliqueId << " is different than process " << i << " cliqueId=" << cliqueIdArray[i] << std::endl;
            ASSERT(0);
        }
    }

    return 0;
}

