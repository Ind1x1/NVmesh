#include <cstring>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <map>

#include "helper.h"

static std::string getDeviceDisplayInfo(int deviceOrdinal) {
    std::stringstream sstream;
    CUdevice dev;
    char name[STRING_LENGTH];
    int busId, deviceId, domainId;

    CU_ASSERT(cuDeviceGet(&dev, deviceOrdinal));
    CU_ASSERT(cuDeviceGetName(name, STRING_LENGTH, dev));
    CU_ASSERT(cuDeviceGetAttribute(&domainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
    CU_ASSERT(cuDeviceGetAttribute(&busId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
    CU_ASSERT(cuDeviceGetAttribute(&deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));
    sstream << name << " (" <<
        std::hex << std::setw(8) << std::setfill('0') << domainId << ":" <<
        std::hex << std::setw(2) << std::setfill('0') << busId << ":" <<
        std::hex << std::setw(2) << std::setfill('0') << deviceId << ")" <<
        std::dec << std::setfill(' ') << std::setw(0);  // reset formatting

    return sstream.str();
}

int discoverRanks(std::map<std::string, std::vector<int>> &rackToProcessMap) {
    ;
}

void MatrixOutputer::incrementalPrint(int stage) {
    if (stage == 0) {
        OUTPUT << "\t";
        for (int i = 0; i < labelsX.size(); i++) {
            OUTPUT << labelsX[i] << "\t";
            if (std::find(columnSeparators.begin(), columnSeparators.end(), i) != columnSeparators.end()) {
                OUTPUT << "| \t";
            }
        }
        OUTPUT << "\n";
    }

    if (stage % dimX == 0) {
        OUTPUT << labelsY[stage / dimX] << "\t";
    }

    if (data[stage] == 0) {
        OUTPUT << "N/A";
    } else {
        OUTPUT << std::fixed << std::setprecision(2) << data[stage];
    }

    if (notes[stage] != "") {
        OUTPUT << ";(" << notes[stage] << ")";
    }

    OUTPUT << "\t";

    if (std::find(columnSeparators.begin(), columnSeparators.end(), stage % dimX) != columnSeparators.end()) {
        OUTPUT << "|\t";
    }

    if (stage % dimX == (dimX - 1)) {
        OUTPUT << "\n";
    }
}