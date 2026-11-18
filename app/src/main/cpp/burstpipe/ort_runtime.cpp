#include "ort_runtime.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <mutex>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#if defined(__ANDROID__)
#include "nnapi_provider_factory.h"
#endif

namespace {

struct OrtLibState {
    bool probed = false;
    bool ok = false;
    void* handle = nullptr;
    const OrtApiBase* apiBase = nullptr;
    std::string version;
    std::unique_ptr<Ort::Env> env;
    std::mutex mu;
};

OrtLibState& ort_state() {
    // Process-lifetime heap singleton: avoids atexit destruction ordering issues against dlopen'd libonnxruntime.
    static auto* s = new OrtLibState();
    return *s;
}

bool ensure_ort_loaded() {
    OrtLibState& s = ort_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.probed) return s.ok;
    s.probed = true;

#if defined(__APPLE__)
    const char* candidates[] = {
        "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
        "/usr/local/opt/onnxruntime/lib/libonnxruntime.dylib",
        "libonnxruntime.dylib",
    };
#else
    const char* candidates[] = {
        "libonnxruntime.so",
        "/vendor/lib64/libonnxruntime.so",
        "/data/local/tmp/ort/libonnxruntime.so",
        "/data/local/tmp/libonnxruntime.so",
    };
#endif

    for (const char* path : candidates) {
        s.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (s.handle) break;
    }
    if (!s.handle) return false;

    using GetApiBaseFn = const OrtApiBase* (*)();
    auto getApiBase = reinterpret_cast<GetApiBaseFn>(dlsym(s.handle, "OrtGetApiBase"));
    if (!getApiBase) {
        dlclose(s.handle);
        s.handle = nullptr;
        return false;
    }

    s.apiBase = getApiBase();
    if (!s.apiBase) {
        dlclose(s.handle);
        s.handle = nullptr;
        return false;
    }

    const OrtApi* api = s.apiBase->GetApi(ORT_API_VERSION);
    if (!api) {
        dlclose(s.handle);
        s.handle = nullptr;
        return false;
    }

    Ort::InitApi(api);
    if (const char* ver = s.apiBase->GetVersionString()) s.version = ver;
    try {
        s.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "pocopaw");
        s.ok = true;
    } catch (...) {
        s.ok = false;
    }
    return s.ok;
}

bool has_provider(const std::vector<std::string>& providers, const char* name) {
    return std::find(providers.begin(), providers.end(), name) != providers.end();
}

Ort::SessionOptions make_base_session_options(int nThreads, const std::vector<int>& cpus) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(nThreads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");
    if (nThreads > 1 && !cpus.empty()) {
        const std::string aff = format_ort_intra_op_affinities(cpus, nThreads);
        if (!aff.empty()) so.AddConfigEntry("session.intra_op_thread_affinities", aff.c_str());
    }
    return so;
}

}  // namespace

struct OrtSessionRunner::Impl {
    std::unique_ptr<Ort::Session> session;
    std::string modelPath;
    int nThreads = 4;
    std::string inputName;
    std::vector<std::string> outputNames;
    std::vector<int> bigPrimeCpus;
};

OrtSessionRunner::~OrtSessionRunner() = default;

bool ort_runtime_available() {
    return ensure_ort_loaded();
}

std::string ort_runtime_version() {
    if (!ensure_ort_loaded()) return "";
    return ort_state().version;
}

std::vector<std::string> ort_available_providers() {
    if (!ensure_ort_loaded()) return {};
    try {
        return Ort::GetAvailableProviders();
    } catch (...) {
        return {};
    }
}

std::unique_ptr<OrtSessionRunner> OrtSessionRunner::create(
    const std::string& modelPath,
    AccelBackend backend,
    int threads,
    std::string* error) {
    if (!ensure_ort_loaded()) {
        if (error) *error = "ONNX Runtime shared library unavailable";
        return nullptr;
    }
    if (modelPath.empty()) {
        if (error) *error = "empty ONNX model path";
        return nullptr;
    }

    const int nThreads = std::clamp(threads, 1, 4);
    const std::vector<int> cpus = detect_big_prime_cpus(4);

    try {
        Ort::SessionOptions so = make_base_session_options(nThreads, cpus);

        std::string activeEp = "CPU";
        if (backend == AccelBackend::NPU_DSP) {
            const auto providers = ort_available_providers();
            if (has_provider(providers, "QNNExecutionProvider")) {
                try {
                    so.AppendExecutionProvider("QNN", {});
                    activeEp = "QNN";
                } catch (...) {
                    activeEp = "CPU";
                }
            }
#if defined(__ANDROID__)
            if (activeEp == "CPU" && has_provider(providers, "NnapiExecutionProvider")) {
                using NnapiAppendFn = OrtStatus* (*)(OrtSessionOptions*, uint32_t);
                auto appendNnapi = reinterpret_cast<NnapiAppendFn>(
                    dlsym(ort_state().handle, "OrtSessionOptionsAppendExecutionProvider_Nnapi"));
                if (appendNnapi) {
                    // Use FP32 NNAPI (NNAPI_FLAG_USE_NONE) to preserve K-Sigma raw precision;
                    // unsupported ops automatically fall back to CPU EP.
                    OrtStatus* st = appendNnapi(so, NNAPI_FLAG_USE_NONE);
                    if (st == nullptr) {
                        activeEp = "NNAPI";
                    } else {
                        Ort::GetApi().ReleaseStatus(st);
                    }
                }
            }
#endif
        }

        std::unique_ptr<Ort::Session> session;
        try {
            session = std::make_unique<Ort::Session>(*ort_state().env, modelPath.c_str(), so);
        } catch (...) {
            if (activeEp == "CPU") throw;
            // Retry with CPU-only SessionOptions if NPU/DSP EP partitioning throws on this model.
            Ort::SessionOptions cpuSo = make_base_session_options(nThreads, cpus);
            session = std::make_unique<Ort::Session>(*ort_state().env, modelPath.c_str(), cpuSo);
            activeEp = "CPU";
        }
        Ort::AllocatorWithDefaultOptions alloc;
        auto inName = session->GetInputNameAllocated(0, alloc);
        const size_t nOut = session->GetOutputCount();
        std::vector<std::string> outNames;
        outNames.reserve(nOut);
        for (size_t i = 0; i < nOut; i++) {
            auto outName = session->GetOutputNameAllocated(i, alloc);
            outNames.push_back(outName ? outName.get() : ("out" + std::to_string(i)));
        }
        if (outNames.empty()) outNames.push_back("out");

        std::unique_ptr<OrtSessionRunner> runner(new OrtSessionRunner());
        runner->impl_ = std::make_unique<Impl>();
        runner->impl_->session = std::move(session);
        runner->impl_->modelPath = modelPath;
        runner->impl_->nThreads = nThreads;
        runner->impl_->inputName = inName ? inName.get() : "in";
        runner->impl_->outputNames = std::move(outNames);
        runner->impl_->bigPrimeCpus = cpus;
        runner->activeEp_ = std::move(activeEp);
        runner->threads_ = nThreads;
        return runner;
    } catch (const Ort::Exception& e) {
        if (error) *error = e.what();
        return nullptr;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    } catch (...) {
        if (error) *error = "unknown ORT session creation error";
        return nullptr;
    }
}

bool OrtSessionRunner::run(
    const float* inputData,
    const int64_t* inputShape,
    int inputRank,
    std::vector<float>& outputData,
    std::vector<int64_t>& outputShape,
    std::string* error) {
    std::vector<std::vector<float>> outputsData;
    std::vector<std::vector<int64_t>> outputsShape;
    if (!run_multi(inputData, inputShape, inputRank, outputsData, outputsShape, error)) return false;
    if (outputsData.empty() || outputsShape.empty()) {
        if (error) *error = "missing ORT output tensor";
        return false;
    }
    outputData = std::move(outputsData[0]);
    outputShape = std::move(outputsShape[0]);
    return true;
}

bool OrtSessionRunner::run_multi(
    const float* inputData,
    const int64_t* inputShape,
    int inputRank,
    std::vector<std::vector<float>>& outputsData,
    std::vector<std::vector<int64_t>>& outputsShape,
    std::string* error) {
    if (!impl_ || !impl_->session || impl_->outputNames.empty() || !inputData || !inputShape || inputRank <= 0) {
        if (error) *error = "invalid ORT runner state or input";
        return false;
    }

    size_t elemCount = 1;
    for (int i = 0; i < inputRank; i++) {
        if (inputShape[i] <= 0) {
            if (error) *error = "non-positive input tensor dimension";
            return false;
        }
        elemCount *= static_cast<size_t>(inputShape[i]);
    }

    try {
        ScopedCpuAffinity affinity(impl_->bigPrimeCpus);
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(inputData), elemCount, inputShape, static_cast<size_t>(inputRank));
        const char* inNames[] = {impl_->inputName.c_str()};
        std::vector<const char*> outNames;
        outNames.reserve(impl_->outputNames.size());
        for (const auto& nm : impl_->outputNames) outNames.push_back(nm.c_str());

        std::vector<Ort::Value> outputs;
        try {
            outputs = impl_->session->Run(
                Ort::RunOptions{nullptr}, inNames, &inputTensor, 1, outNames.data(), outNames.size());
        } catch (...) {
            if (activeEp_ == "CPU" || impl_->modelPath.empty()) throw;
            // Fallback to CPU EP if NPU/NNAPI inference throws at runtime (e.g. dynamic shapes).
            Ort::SessionOptions cpuSo = make_base_session_options(impl_->nThreads, impl_->bigPrimeCpus);
            impl_->session = std::make_unique<Ort::Session>(*ort_state().env, impl_->modelPath.c_str(), cpuSo);
            activeEp_ = "CPU";
            outputs = impl_->session->Run(
                Ort::RunOptions{nullptr}, inNames, &inputTensor, 1, outNames.data(), outNames.size());
        }
        if (outputs.size() != outNames.size()) {
            if (error) *error = "unexpected ORT output count";
            return false;
        }

        outputsData.resize(outputs.size());
        outputsShape.resize(outputs.size());
        for (size_t i = 0; i < outputs.size(); i++) {
            if (!outputs[i].IsTensor()) {
                if (error) *error = "missing ORT output tensor";
                return false;
            }
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                if (error) *error = "unexpected ORT output element type (expected float32)";
                return false;
            }
            outputsShape[i] = info.GetShape();
            const size_t outElems = info.GetElementCount();
            const float* outPtr = outputs[i].GetTensorData<float>();
            if (!outPtr || outElems == 0) {
                if (error) *error = "empty ORT output tensor";
                return false;
            }
            outputsData[i].assign(outPtr, outPtr + outElems);
        }
        return true;
    } catch (const Ort::Exception& e) {
        if (error) *error = e.what();
        return false;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    } catch (...) {
        if (error) *error = "unknown ORT inference error";
        return false;
    }
}
