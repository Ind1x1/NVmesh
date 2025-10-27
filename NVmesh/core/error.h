#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cuda.h>

#define BASIC_CU_ASSERT(x) do {                                 \
    CUresult cuResult = (x);                                    \
    if ((cuResult) != CUDA_SUCCESS) {                           \
        const char *errDescStr, *errNameStr;                    \
        cuGetErrorString(cuResult, &errDescStr);                \
        cuGetErrorName(cuResult, &errNameStr);                  \
        std::cerr << "[" << errNameStr << "] " << errDescStr << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1);                                           \
    }                                                           \
} while(0)

static std::vector<std::string> getCudaLogs() {
    std::vector<std::string> messages;
#if CUDA_VERSION >= 12090
    size_t messageSize = 25601;
    std::string messagesRaw;
    messagesRaw.resize(messageSize);

    CUresult (*cuLogsDumpToMemory_ptr)(CUlogIterator*, char*, size_t*, unsigned int);
    CUdriverProcAddressQueryResult result;
    BASIC_CU_ASSERT(cuGetProcAddress("cuLogsDumpToMemory", (void **) &cuLogsDumpToMemory_ptr, 12090, CU_GET_PROC_ADDRESS_DEFAULT, &result));

    if (result == CU_GET_PROC_ADDRESS_SUCCESS) {
        BASIC_CU_ASSERT((*cuLogsDumpToMemory_ptr)(NULL, messagesRaw.data(), &messageSize, 0));
        messagesRaw.resize(messageSize);
        std::istringstream messagesStringstream(messagesRaw);
        std::string message;

        while(std::getline(messagesStringstream, message, '\n')) {
            messages.push_back(message);
        }
    }
#endif
    return messages;
}

#undef BASIC_CU_ASSERT

#define ASSERT(x) do {                                      \
    if (!(x)) {                                             \
        std::cerr << "[" << "pass" << ":" << "pass" << "]: " << "ASSERT in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__  << std::endl; \
        std::exit(1);                                       \
    }                                                       \
} while (0)

#define CU_ASSERT(x) do {                                   \
    CUresult cuResult = (x);                                \
    if ((cuResult) != CUDA_SUCCESS) {                       \
        const char *errDescStr, *errNameStr;                \
        cuGetErrorString(cuResult, &errDescStr);            \
        cuGetErrorName(cuResult, &errNameStr);              \
        for (auto message : getCudaLogs()) {std::cerr << "[" << "pass" << ":" << "pass" << "]: " << message << std::endl;}; \
        std::cerr << "[" << "pass" << ":" << "pass" << "]: " << "[" << errNameStr << "] " << errDescStr << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1);                                       \
    }                                                       \
} while(0)

#define CUDA_ASSERT(x) do {                                 \
    cudaError_t cudaErr = (x);                              \
    if ((cudaErr) != cudaSuccess) {                         \
        for (auto message : getCudaLogs()) {std::cerr << "[" << "pass" << ":" << "pass" << "]: " << message << std::endl;}; \
        std::cerr << "[" << "pass" << ":" << "pass" << "]: " << "[" << cudaGetErrorName(cudaErr) << "] " << cudaGetErrorString(cudaErr) << " in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1);                                       \
    }                                                       \
} while (0)

#define NVML_ASSERT(x) do {                                 \
    nvmlReturn_t nvmlResult = (x);                          \
    if ((nvmlResult) != NVML_SUCCESS) {                     \
        std::cerr << "[" << "pass" << ":" << "pass" << "]: " << "[" << nvmlErrorString(nvmlResult) << "] in expression " << #x << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        std::exit(1);                                       \
    }                                                       \
} while (0)
