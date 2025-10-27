#pragma once

#include <Memory>
#include <map>
#include "common.h"
#include <iostream>

extern int iterations;

enum AllocatorStrategy {
    ALLOCATOR_STRATEGY_UNIQUE = 0,
    ALLOCATOR_STRATEGY_REUSE,
    ALLOCATOR_STRATEGY_CUDA_POOLS,
};

class Testcase {
public:
    virtual void run(size_t copySize) = 0;
    virtual bool filter() = 0;

    virtual void filterRun(size_t copySize) {
        if (filter()) {
            run(copySize);
        } else {
            std::cout << "testcase WAIVED" << std::endl;
        }
    }

    virtual std::string getName() {
        return "not implemented";
    }
};

template<typename dstAllocator, typename srcAllocator>
class TestcaseDstSrc : public Testcase {
public:
    bool filter() {
        return MPIWrapper::getWorldSize() > 1 && srcAllocator::filter() && dstAllocator::filter();
    }
};

std::tuple<std::map<std::string, std::unique_ptr<Testcase> >, std::map<std::string, std::vector<std::string> > > buildTestcases(AllocatorStrategy strategy);
