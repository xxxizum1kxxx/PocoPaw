#include "dsp_probe.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>

// Read-only look at the Hexagon cDSP through FastRPC (libcdsprpc.so, a public vendor library): what the
// DSP reports about itself, and whether an app process may open a module in an unsigned protection
// domain, the only way third-party code can run there without Qualcomm's signing. Nothing is executed on
// the DSP: the probe opens the stock calculator skel and closes it again.
//
// Proprietary SDK boundary: compiling custom HVX/cDSP kernels (lib*_skel.so) requires Qualcomm's
// proprietary Hexagon SDK (qaic / hexagon-clang), which is not bundled. Without a signed or unsigned
// custom skel on the device, custom burst stages run on OpenCL GPU or big+prime CPU cores.
namespace {

// remote.h (Hexagon SDK)
constexpr uint32_t kGetDspInfo = 2;            // remote_handle_control: DSPRPC_GET_DSP_INFO
constexpr uint32_t kControlUnsignedModule = 2;  // remote_session_control: DSPRPC_CONTROL_UNSIGNED_MODULE
constexpr int kCdsp = 3;
struct DspCapability {
    uint32_t domain, attribute, capability;
};
struct UnsignedModule {
    int domain, enable;
};
using HandleControl = int (*)(uint32_t, void*, uint32_t);
using SessionControl = int (*)(uint32_t, void*, uint32_t);
using HandleOpen = int (*)(const char*, uint64_t*);
using HandleClose = int (*)(uint64_t);

}  // namespace

DspProbeInfo dsp_probe_info() {
    DspProbeInfo info;
    void* lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        const char* err = dlerror();
        info.note = std::string("libcdsprpc.so not loadable: ") + (err ? err : "unknown");
        return info;
    }
    info.cdspFastRpcAvailable = true;
    auto control = reinterpret_cast<HandleControl>(dlsym(lib, "remote_handle_control"));
    auto session = reinterpret_cast<SessionControl>(dlsym(lib, "remote_session_control"));
    auto openModule = reinterpret_cast<HandleOpen>(dlsym(lib, "remote_handle64_open"));
    auto closeModule = reinterpret_cast<HandleClose>(dlsym(lib, "remote_handle64_close"));

    if (control) {
        auto query = [&](uint32_t attr, uint32_t& dst) {
            DspCapability c{kCdsp, attr, 0};
            if (control(kGetDspInfo, &c, sizeof c) == 0) dst = c.capability;
        };
        uint32_t dom = 0, upd = 0;
        query(0, dom);
        query(1, upd);
        query(3, info.hvx128Count);
        query(6, info.arch);
        query(7, info.hmxDepth);
        info.domainSupport = dom != 0;
        info.unsignedPdSupported = upd != 0;
    }
    if (session && openModule && closeModule) {
        UnsignedModule u{kCdsp, 1};
        int err = session(kControlUnsignedModule, &u, sizeof u);
        uint64_t h = 0;
        int oerr = err ? err : openModule("file:///libcalculator_skel.so?calculator_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h);
        if (!oerr) {
            closeModule(h);
            info.skelOpened = true;
        } else {
            info.skelError = oerr;
        }
    }
    info.note = "FastRPC runtime probed; custom cDSP skel requires proprietary Hexagon SDK (unavailable)";
    dlclose(lib);
    return info;
}

std::string dsp_probe() {
    void* lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        const char* err = dlerror();
        return std::string("libcdsprpc.so not loadable: ") + (err ? err : "unknown");
    }
    auto control = reinterpret_cast<HandleControl>(dlsym(lib, "remote_handle_control"));
    auto session = reinterpret_cast<SessionControl>(dlsym(lib, "remote_session_control"));
    auto openModule = reinterpret_cast<HandleOpen>(dlsym(lib, "remote_handle64_open"));
    auto closeModule = reinterpret_cast<HandleClose>(dlsym(lib, "remote_handle64_close"));
    std::string out;
    char buf[160];
    if (!control) {
        out = "remote_handle_control missing\n";
    } else {
        const struct { uint32_t id; const char* name; } attrs[] = {
            {0, "domain support flag"}, {1, "unsigned PD supported"}, {3, "HVX 128-byte"}, {6, "architecture"}, {7, "HMX depth"}};
        for (const auto& a : attrs) {
            DspCapability c{kCdsp, a.id, 0};
            int err = control(kGetDspInfo, &c, sizeof c);
            if (err) snprintf(buf, sizeof buf, "%s: error 0x%x\n", a.name, err);
            else if (a.id == 6) snprintf(buf, sizeof buf, "%s: 0x%x\n", a.name, c.capability);
            else snprintf(buf, sizeof buf, "%s: %u\n", a.name, c.capability);
            out += buf;
        }
    }
    if (!session || !openModule || !closeModule) {
        out += "unsigned module open: FastRPC session API missing";
    } else {
        UnsignedModule u{kCdsp, 1};
        int err = session(kControlUnsignedModule, &u, sizeof u);
        uint64_t h = 0;
        int oerr = err ? err : openModule("file:///libcalculator_skel.so?calculator_skel_handle_invoke&_modver=1.0&_dom=cdsp", &h);
        if (!oerr) closeModule(h);
        if (err) snprintf(buf, sizeof buf, "unsigned module open: session control error 0x%x", err);
        else if (oerr) snprintf(buf, sizeof buf, "unsigned module open: calculator skel error 0x%x", oerr);
        else snprintf(buf, sizeof buf, "unsigned module open: calculator skel opened");
        out += buf;
    }
    dlclose(lib);
    return out;
}
