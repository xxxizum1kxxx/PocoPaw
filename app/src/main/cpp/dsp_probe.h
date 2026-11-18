#pragma once
#include <cstdint>
#include <string>

struct DspProbeInfo {
    bool cdspFastRpcAvailable = false;
    bool domainSupport = false;
    bool unsignedPdSupported = false;
    uint32_t hvx128Count = 0;
    uint32_t arch = 0;
    uint32_t hmxDepth = 0;
    bool skelOpened = false;
    int skelError = 0;
    std::string note;
};

// Structured Hexagon cDSP FastRPC capability probe (dlopen libcdsprpc.so).
DspProbeInfo dsp_probe_info();

// One line per finding about the Hexagon cDSP (see dsp_probe.cpp).
std::string dsp_probe();
