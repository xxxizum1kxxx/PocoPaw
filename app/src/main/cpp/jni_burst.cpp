#include <android/bitmap.h>
#include <jni.h>

#include <cstdlib>
#include <cstring>

#include "burstpipe/burst.h"
#include "dsp_probe.h"

namespace {

BurstSession* session(jlong handle) { return reinterpret_cast<BurstSession*>(handle); }

void throwState(JNIEnv* env, const char* msg) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), msg);
}

// C++ exceptions must not cross into the JVM
template <class F>
void guarded(JNIEnv* env, F f) {
    try {
        f();
    } catch (const std::exception& e) {
        throwState(env, e.what());
    }
}

// options: kernelRegression, robustLo, robustHi, dilate, demosaic (0 MHC, 1 RCD, 2 AMaZE), freqMerge, freqC,
// freqSpatial, lcaR[2], lcaB[2]
BurstOptions burst_options(JNIEnv* env, jfloatArray jopt) {
    BurstOptions o;
    jfloat* v = env->GetFloatArrayElements(jopt, nullptr);
    o.kernelRegression = v[0] != 0;
    o.robustLo = v[1];
    o.robustHi = v[2];
    o.dilate = (int)v[3];
    o.demosaic = v[4] == 0 ? Demosaic::MHC : (v[4] == 2 ? Demosaic::AMAZE : Demosaic::RCD);
    o.freqMerge = v[5] != 0;
    o.freqC = v[6];
    o.freqSpatial = v[7];
    for (int i = 0; i < 2; i++) { o.lcaR[i] = v[8 + i]; o.lcaB[i] = v[10 + i]; }
    env->ReleaseFloatArrayElements(jopt, v, JNI_ABORT);
    return o;
}

std::mutex g_rawMu;
struct RawBlock { void* ptr; size_t bytes; };
std::vector<RawBlock> g_rawPool;
constexpr size_t kMaxRawPoolEntries = 4;
constexpr size_t kMinRawPoolBytes = 1ull << 20;
constexpr size_t kMaxRawPoolBytes = 48ull << 20;

void* raw_pool_alloc(size_t bytes) {
    if (bytes >= kMinRawPoolBytes && bytes <= kMaxRawPoolBytes) {
        std::lock_guard<std::mutex> lk(g_rawMu);
        for (size_t i = g_rawPool.size(); i > 0; i--) {
            if (g_rawPool[i - 1].bytes == bytes) {
                void* p = g_rawPool[i - 1].ptr;
                g_rawPool.erase(g_rawPool.begin() + (ptrdiff_t)(i - 1));
                return p;
            }
        }
    }
    return malloc(bytes);
}

void raw_pool_free(void* p, size_t bytes) {
    if (!p) return;
    if (bytes >= kMinRawPoolBytes && bytes <= kMaxRawPoolBytes) {
        std::lock_guard<std::mutex> lk(g_rawMu);
        if (g_rawPool.size() < kMaxRawPoolEntries) {
            g_rawPool.push_back({p, bytes});
            return;
        }
        auto it = std::find_if(g_rawPool.begin(), g_rawPool.end(), [&](const RawBlock& b) { return b.bytes != bytes; });
        if (it != g_rawPool.end()) {
            free(it->ptr);
            g_rawPool.erase(it);
            g_rawPool.push_back({p, bytes});
            return;
        }
    }
    free(p);
}

}  // namespace

extern "C" {

// meta: width, height, cfa, white, black[4] (R, G_even, G_odd, B), wb[4], ccm[9] (row-major),
// lscRows, lscCols; lsc: rows*cols*4 gains (R, G_even, G_odd, B) or null; cal: optional 42-float DualColorCal;
// exposureNs / iso per frame (of the first frame only for a session with no frames, which merges frames as added).
JNIEXPORT jlong JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeCreate(JNIEnv* env, jclass, jfloatArray jmeta, jfloatArray jlsc,
                                                         jfloatArray jcal, jobjectArray jframes, jint ref,
                                                         jlongArray jexp, jintArray jiso) {
    Meta m;
    jfloat* v = env->GetFloatArrayElements(jmeta, nullptr);
    m.width = (int)v[0];
    m.height = (int)v[1];
    m.rowStride = m.width * 2;
    m.cfa = (int)v[2];
    m.white = v[3];
    for (int i = 0; i < 4; i++) m.black[i] = v[4 + i];
    for (int i = 0; i < 4; i++) m.wb[i] = v[8 + i];
    for (int i = 0; i < 9; i++) m.ccm[i] = v[12 + i];
    m.lscRows = (int)v[21];
    m.lscCols = (int)v[22];
    env->ReleaseFloatArrayElements(jmeta, v, JNI_ABORT);
    if (jcal != nullptr && env->GetArrayLength(jcal) >= 42 && !ccm_usable(m.ccm)) {
        jfloat c[42];
        env->GetFloatArrayRegion(jcal, 0, 42, c);
        DualColorCal cal;
        cal.illuminants = (int)c[0];
        cal.illuminant[0] = (int)c[1];
        cal.illuminant[1] = (int)c[2];
        for (int i = 0; i < 9; i++) {
            cal.colorMatrix[0][i] = c[3 + i];
            cal.colorMatrix[1][i] = c[12 + i];
            cal.forwardMatrix[0][i] = c[21 + i];
            cal.forwardMatrix[1][i] = c[30 + i];
        }
        for (int i = 0; i < 3; i++) cal.neutral[i] = c[39 + i];
        compute_dual_illuminant_ccm(cal, m.wb, m.ccm);
    }
    if (jlsc != nullptr && env->GetArrayLength(jlsc) == m.lscRows * m.lscCols * 4) {
        m.lsc.resize((size_t)m.lscRows * m.lscCols * 4);
        env->GetFloatArrayRegion(jlsc, 0, (jsize)m.lsc.size(), m.lsc.data());
    } else {
        m.lscRows = m.lscCols = 0;
    }

    int n = env->GetArrayLength(jframes), ne = env->GetArrayLength(jexp);
    if (env->GetArrayLength(jiso) == ne) {
        std::vector<jlong> e(ne);
        std::vector<jint> iso(ne);
        env->GetLongArrayRegion(jexp, 0, ne, e.data());
        env->GetIntArrayRegion(jiso, 0, ne, iso.data());
        for (int k = 0; k < ne; k++) { m.exposureNs.push_back((double)e[k]); m.iso.push_back(iso[k]); }
    }
    std::vector<Raw> frames(n);
    for (int k = 0; k < n; k++) {
        jobject buf = env->GetObjectArrayElement(jframes, k);
        auto* p = static_cast<const uint16_t*>(env->GetDirectBufferAddress(buf));
        jlong cap = env->GetDirectBufferCapacity(buf);
        env->DeleteLocalRef(buf);
        if (p == nullptr || cap < (jlong)m.width * m.height * 2) {
            throwState(env, "RAW frame is not a direct buffer of width*height*2 bytes");
            return 0;
        }
        frames[k].w = m.width;
        frames[k].h = m.height;
        frames[k].d = p;
    }
    return reinterpret_cast<jlong>(new BurstSession(m, std::move(frames), ref));
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeCrop(JNIEnv*, jclass, jlong h, jfloat zoom) { session(h)->crop(zoom); }

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeSetAspect(JNIEnv*, jclass, jlong h, jfloat aspect) { session(h)->set_aspect(aspect); }

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeUseGpu(JNIEnv*, jclass, jlong h, jboolean on) { session(h)->use_gpu(on); }

JNIEXPORT jint JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeSharpestFrame(JNIEnv* env, jclass, jobjectArray jframes, jint w, jint h) {
    int n = env->GetArrayLength(jframes);
    std::vector<Raw> frames(n);
    for (int k = 0; k < n; k++) {
        jobject buf = env->GetObjectArrayElement(jframes, k);
        auto* p = static_cast<const uint16_t*>(env->GetDirectBufferAddress(buf));
        jlong cap = env->GetDirectBufferCapacity(buf);
        env->DeleteLocalRef(buf);
        if (p == nullptr || cap < (jlong)w * h * 2) return 0;
        frames[k].w = w;
        frames[k].h = h;
        frames[k].d = p;
    }
    return sharpest_frame(frames, n);
}

// iso / a / b: the camera's noise curve (NoiseCurve in merge.h), equal lengths
JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeSetNoiseCurve(JNIEnv* env, jclass, jlong h, jfloatArray jiso, jfloatArray ja,
                                                                jfloatArray jb) {
    NoiseCurve c;
    int n = env->GetArrayLength(jiso);
    if (env->GetArrayLength(ja) != n || env->GetArrayLength(jb) != n) return;
    c.iso.resize(n);
    c.a.resize(n);
    c.b.resize(n);
    env->GetFloatArrayRegion(jiso, 0, n, c.iso.data());
    env->GetFloatArrayRegion(ja, 0, n, c.a.data());
    env->GetFloatArrayRegion(jb, 0, n, c.b.data());
    session(h)->set_noise_curve(std::move(c));
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeAlign(JNIEnv* env, jclass, jlong h) {
    guarded(env, [&] { session(h)->align(); });
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeMerge(JNIEnv* env, jclass, jlong h, jfloatArray jopt) {
    BurstOptions o = burst_options(env, jopt);
    guarded(env, [&] { session(h)->merge(o); });
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeBeginStream(JNIEnv* env, jclass, jlong h, jfloatArray jopt) {
    BurstOptions o = burst_options(env, jopt);
    guarded(env, [&] { session(h)->begin_stream(o); });
}

// the reference's index once chosen, else -1
JNIEXPORT jint JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeAddFrame(JNIEnv* env, jclass, jlong h, jobject buf, jint w, jint ht) {
    auto* p = static_cast<const uint16_t*>(env->GetDirectBufferAddress(buf));
    if (p == nullptr || env->GetDirectBufferCapacity(buf) < (jlong)w * ht * 2) {
        throwState(env, "RAW frame is not a direct buffer of width*height*2 bytes");
        return -1;
    }
    Raw r;
    r.w = w;
    r.h = ht;
    r.d = p;
    jint ref = -1;
    guarded(env, [&] { ref = session(h)->add_frame(r); });
    return ref;
}

JNIEXPORT jint JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeEndFrames(JNIEnv* env, jclass, jlong h) {
    jint ref = -1;
    guarded(env, [&] { ref = session(h)->end_frames(); });
    return ref;
}

// style: "default" uses look as given, "pixel" / "night" replace it with that preset (finish.cpp);
// look: lscLumaStrength, chromaDenoise, compression, targetKey, contrast, saturation, sharpen, sharpenRadius, localContrast
JNIEXPORT jlong JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeFinish(JNIEnv* env, jclass, jlong h, jstring jstyle, jfloatArray jlook,
                                                         jboolean keepHdr) {
    LookParams lp;
    jfloat* v = env->GetFloatArrayElements(jlook, nullptr);
    lp.lscLumaStrength = v[0];
    lp.chromaDenoise = v[1];
    lp.compression = v[2];
    lp.targetKey = v[3];
    lp.contrast = v[4];
    lp.saturation = v[5];
    lp.sharpen = v[6];
    lp.sharpenRadius = v[7];
    lp.localContrast = v[8];
    env->ReleaseFloatArrayElements(jlook, v, JNI_ABORT);
    const char* style = env->GetStringUTFChars(jstyle, nullptr);
    if (std::string(style) != "default") apply_look_preset(style, lp);
    env->ReleaseStringUTFChars(jstyle, style);
    BurstSession* s = session(h);
    guarded(env, [&] { s->finish(lp, keepHdr); });
    return ((jlong)s->width() << 32) | (jlong)(uint32_t)s->height();
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeRenderThumb(JNIEnv* env, jclass, jlong h, jobject bitmap) {
    BurstSession* s = session(h);
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width == 0 || info.height == 0) {
        throwState(env, "thumbnail bitmap must be non-empty ARGB_8888");
        return;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throwState(env, "cannot lock thumbnail bitmap");
        return;
    }
    guarded(env, [&] { s->render_thumb(static_cast<uint8_t*>(pixels), info.stride, (int)info.width, (int)info.height); });
    AndroidBitmap_unlockPixels(env, bitmap);
}

JNIEXPORT jbyteArray JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeEncodePhoto(JNIEnv* env, jclass, jlong h, jint quality,
                                                               jboolean wantUltraHdr, jfloat headroom,
                                                               jbooleanArray jisUltraHdr) {
    std::vector<uint8_t> file;
    bool uhdr = false;
    guarded(env, [&] { file = session(h)->encode_photo(quality, wantUltraHdr, headroom, &uhdr); });
    if (file.empty()) return nullptr;
    if (jisUltraHdr != nullptr && env->GetArrayLength(jisUltraHdr) >= 1) {
        jboolean flag = uhdr ? JNI_TRUE : JNI_FALSE;
        env->SetBooleanArrayRegion(jisUltraHdr, 0, 1, &flag);
    }
    jbyteArray out = env->NewByteArray((jsize)file.size());
    if (out != nullptr) env->SetByteArrayRegion(out, 0, (jsize)file.size(), reinterpret_cast<const jbyte*>(file.data()));
    return out;
}

// The merged mosaic as a 16-bit DNG in a malloc'd direct buffer (NativeMemory.free), or null. color: per illuminant
// colorMatrix[9], forwardMatrix[9] (row-major); illuminants: EXIF codes, 0-2 of them; neutral[3]; thumb: RGB8 or null.
JNIEXPORT jobject JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeEncodeDng(JNIEnv* env, jclass, jlong h, jfloatArray jcolor, jintArray jill,
                                                            jfloatArray jneutral, jint orientation, jstring jmake, jstring jmodel,
                                                            jstring jsoftware, jbyteArray jthumb, jint tw, jint th) {
    DngColor c;
    c.illuminants = std::min(2, (int)env->GetArrayLength(jill));
    if (env->GetArrayLength(jcolor) < c.illuminants * 18 || env->GetArrayLength(jneutral) < 3) return nullptr;
    std::vector<float> col(c.illuminants * 18);
    env->GetFloatArrayRegion(jcolor, 0, (jsize)col.size(), col.data());
    std::vector<jint> ill(c.illuminants);
    env->GetIntArrayRegion(jill, 0, c.illuminants, ill.data());
    for (int i = 0; i < c.illuminants; i++) {
        c.illuminant[i] = ill[i];
        std::copy(col.begin() + i * 18, col.begin() + i * 18 + 9, c.colorMatrix[i]);
        std::copy(col.begin() + i * 18 + 9, col.begin() + i * 18 + 18, c.forwardMatrix[i]);
    }
    env->GetFloatArrayRegion(jneutral, 0, 3, c.neutral);
    DngExtras x;
    x.orientation = orientation;
    auto str = [&](jstring js) {
        const char* p = env->GetStringUTFChars(js, nullptr);
        std::string v = p;
        env->ReleaseStringUTFChars(js, p);
        return v;
    };
    x.make = str(jmake);
    x.model = str(jmodel);
    x.software = str(jsoftware);
    if (jthumb != nullptr && env->GetArrayLength(jthumb) >= tw * th * 3) {
        x.thumb.resize((size_t)tw * th * 3);
        env->GetByteArrayRegion(jthumb, 0, (jsize)x.thumb.size(), reinterpret_cast<jbyte*>(x.thumb.data()));
        x.thumbW = tw;
        x.thumbH = th;
    }
    std::vector<uint8_t> file;
    guarded(env, [&] { file = session(h)->encode_dng(c, x); });
    if (file.empty()) return nullptr;
    void* p = malloc(file.size());
    if (p == nullptr) return nullptr;
    memcpy(p, file.data(), file.size());
    return env->NewDirectByteBuffer(p, (jlong)file.size());
}

JNIEXPORT jstring JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeSummary(JNIEnv* env, jclass, jlong h) {
    return env->NewStringUTF(session(h)->summary().c_str());
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeRelease(JNIEnv*, jclass, jlong h) {
    delete session(h);
    pool_trim_excess(96ull << 20);
    Gpu::trim_excess(256ull << 20);
}

// Sets where compiled OpenCL programs are cached and creates the device (compiling the kernels on the
// first run, ~0.5 s): one line per device property, or why the GPU is unavailable.
JNIEXPORT jstring JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeGpuInit(JNIEnv* env, jclass, jstring jdir) {
    const char* dir = env->GetStringUTFChars(jdir, nullptr);
    Gpu::set_cache_dir(dir);
    env->ReleaseStringUTFChars(jdir, dir);
    return env->NewStringUTF(Gpu::describe().c_str());
}

JNIEXPORT jstring JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeDspProbe(JNIEnv* env, jclass) {
    return env->NewStringUTF(dsp_probe().c_str());
}

JNIEXPORT jobject JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeMemory_nativeAllocate(JNIEnv* env, jclass, jlong bytes) {
    if (bytes <= 0) return nullptr;
    void* p = raw_pool_alloc((size_t)bytes);
    if (p == nullptr) return nullptr;
    return env->NewDirectByteBuffer(p, bytes);
}

JNIEXPORT jobject JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeMemory_nativeCopyRows(JNIEnv* env, jclass, jobject srcBuf, jint srcStride,
                                                             jint rowBytes, jint height) {
    if (rowBytes <= 0 || height <= 0 || srcStride < rowBytes) return nullptr;
    const auto* src = static_cast<const uint8_t*>(env->GetDirectBufferAddress(srcBuf));
    jlong srcCap = env->GetDirectBufferCapacity(srcBuf);
    size_t total = (size_t)rowBytes * (size_t)height;
    size_t required = (size_t)(height - 1) * (size_t)srcStride + (size_t)rowBytes;
    if (src == nullptr || srcCap < (jlong)required) return nullptr;
    auto* dst = static_cast<uint8_t*>(raw_pool_alloc(total));
    if (dst == nullptr) return nullptr;
    if (srcStride == rowBytes) {
        memcpy(dst, src, total);
    } else {
        for (int y = 0; y < height; y++) {
            memcpy(dst + (size_t)y * rowBytes, src + (size_t)y * srcStride, (size_t)rowBytes);
        }
    }
    return env->NewDirectByteBuffer(dst, (jlong)total);
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeMemory_nativeFree(JNIEnv* env, jclass, jobject buffer) {
    if (!buffer) return;
    void* p = env->GetDirectBufferAddress(buffer);
    jlong cap = env->GetDirectBufferCapacity(buffer);
    raw_pool_free(p, cap > 0 ? (size_t)cap : 0);
}

}  // extern "C"
