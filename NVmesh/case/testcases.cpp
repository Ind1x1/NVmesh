#include <iostream>
#include <memory>
#include <random>
#include <algorithm>
#include <utility>

#include "common.h"
#include "testcases.h"
#include "utils.h"
#include "error.h"
#include "nvmesh.h"
#include "mpi_helper.h"


enum CopyCount {
    COPY_COUNT_UNIDIR = 0,
    COPY_COUNT_BIDIR,
};

std::string getCopyCountName(CopyCount copyCount) {
    if (copyCount == COPY_COUNT_UNIDIR) return "unidir";
    if (copyCount == COPY_COUNT_BIDIR) return "bidir";
    return "";
}

std::string getCopyDirectionName(CopyDirection copyDirection) {
    if (copyDirection == COPY_DIRECTION_WRITE) return "write";
    if (copyDirection == COPY_DIRECTION_READ) return "read";
    return "";
}

std::string getCopyTypeName(CopyType copyType) {
    if (copyType == COPY_TYPE_CE) return "ce";
    if (copyType == COPY_TYPE_SM) return "sm";
    if (copyType == COPY_TYPE_TMA) return "tma";
    //OPTIMIZE
    return "";
}

template<typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
double doUnidir(int i, int j, size_t copySize) {
    std::shared_ptr<srcAllocator> src = std::make_shared<srcAllocator>(copySize, i);
    std::shared_ptr<dstAllocator> dst = std::make_shared<dstAllocator>(copySize, j);
    Copy copy(dst, src, copyDirection, copyType, iterations);

    auto results = nvmesh::doBenchmark({copy});
    return results[0];
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
double doBidir(int i, int j, size_t copySize) {
    std::shared_ptr<srcAllocator> src1 = std::make_shared<srcAllocator>(copySize, i);
    std::shared_ptr<dstAllocator> dst1 = std::make_shared<dstAllocator>(copySize, j);
    Copy copy1(dst1, src1, copyDirection, copyType, iterations);

    std::shared_ptr<srcAllocator> src2 = std::make_shared<srcAllocator>(copySize, j);
    std::shared_ptr<dstAllocator> dst2 = std::make_shared<dstAllocator>(copySize, i);
    Copy copy2(dst2, src2, copyDirection, copyType, iterations);

    auto results = nvmesh::doBenchmark({copy1, copy2});
    return results[0] + results[1];
}

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType, CopyCount copyCount>
double doUnidirBidirHelper(int i, int j, size_t copySize) {
    if (copyCount == COPY_COUNT_UNIDIR) {
        return doUnidir<dstAllocator, srcAllocator, copyDirection, copyType>(i, j, copySize);
    } else {
        return doBidir<dstAllocator, srcAllocator, copyDirection, copyType>(i, j, copySize);
    }
}

static void addTestcase(
    std::map<std::string, std::vector<std::string> > &suites,
    std::string suiteName,
    std::map<std::string, std::unique_ptr<Testcase> > &testcases,
    std::unique_ptr<Testcase> &&testcase) {

    if (suiteName.size() > 0) {
        suites[suiteName].push_back(testcase->getName());
    }
    testcases[testcase->getName()] = std::move(testcase);
}


template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType, CopyCount copyCount>
class N_squared_pattern : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        MatrixOutputer output(getName(), MPIWrapper::getWorldSize(), MPIWrapper::getWorldSize());
        for(int j = 0; j < MPIWrapper::getWorldSize(); j++) {
            for(int i = 0; i < MPIWrapper::getWorldSize(); i++) {
                if (i == j) {
                    output.set(i, j, 0);
                    continue;
                }

                double bandwidth = doUnidirBidirHelper<dstAllocator, srcAllocator, copyDirection, copyType, copyCount>(i, j, copySize);
                output.set(i, j, bandwidth);
            }
        }
    }

    std::string getName() {
        return srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyCountName(copyCount) +
        "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename dstAllocator, typename srcAllocator, CopyDirection copyDirection, CopyType copyType>
class N_squared_pattern_bidir : public TestcaseDstSrc<dstAllocator, srcAllocator> {
public:
    void run(size_t copySize) {
        OutputMatrix output(getName(), MPIWrapper::getWorldSize(), MPIWrapper::getWorldSize());
        for(int j = 0; j < MPIWrapper::getWorldSize(); j++) {
            for(int i = j; i < MPIWrapper::getWorldSize(); i++) {
                if (i == j) {
                    output.set(i, j, 0);
                    continue;
                }

                double bandwidth = doBidir<dstAllocator, srcAllocator, copyDirection, copyType>(i, j, copySize);
                output.set(i, j, bandwidth);
                output.set(j, i, bandwidth);
            }
        }
    }

    std::string getName() {
        return srcAllocator::getName() + "_to_" + dstAllocator::getName() + "_" + getCopyCountName(COPY_COUNT_BIDIR) +
        "_" + getCopyDirectionName(copyDirection) + "_" + getCopyTypeName(copyType);
    }
};

template <typename unicastAllocator, typename egmAllocator, typename multicastAllocator>
std::tuple<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string>>> buildTestcasesLower() {
    std::map<std::string, std::unique_ptr<Testcase>> testcases;
    std::map<std::string, std::vector<std::string>> suites;

    //OPTIMIZE
    return {std::move(testcases), std::move(suites)};
}

std::tuple<std::map<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string> > > buildTestcases(AllocatorStrategy strategy) {
    ;
}