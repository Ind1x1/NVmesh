#include <vector>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <map>
#include <mpi.h>
#include <memory>

#include "error.h"
#include "common.h"

class MPIWrapper {
private:
    int worldSize;
    int worldRank;
    
    MPIWrapper() {
        int mpiInitalized;
        MPI_Initialized(&mpiInitalized);
        if (!mpiInitalized) {
            MPI_Init(NULL, NULL);
        }
    
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    }
    
public:
    static MPIWrapper& instance() {
        static MPIWrapper mpiWrapper;
        return mpiWrapper;
    }
    
    static int getWorldSize() {
        return instance().worldSize;
    }
    static int getWorldRank() {
        return instance().worldRank;
    }
};

class MPIOutput {
public:
    template <typename T>
    MPIOutput& operator<<(T input) {
        if (MPIWrapper::getWorldRank() == 0) {
            std::cout << input;
        }
        return *this;
    }
    using StreamType = decltype(std::cout);
    MPIOutput& operator<<(StreamType &(*func)(StreamType &)) {
        if (MPIWrapper::getWorldRank() == 0) {
            std::cout << func;
        }
        return *this;
    }
};
extern MPIOutput Output;