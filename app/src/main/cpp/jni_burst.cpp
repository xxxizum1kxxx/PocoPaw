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
// freqSpatial, lcaR[2], lcaB[2], krScale, krAuto2x, krSnrThresholdDb
BurstOptions burst_options(JNIEnv* env, jfloatArray jopt) {
    BurstOptions o;
    jsize len = env->GetArrayLength(jopt);
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
    if (len >= 15) {
        o.krScale = (int)v[12];
        o.krAuto2x = v[13] != 0;
        o.krSnrThresholdDb = v[14];
    }
    if (len >= 17) {
        o.earlyStopEnabled = v[15] != 0;
        o.earlyStopTargetSnrDb = v[16];
    }
    env->ReleaseFloatArrayElements(jopt, v, JNI_ABORT);
    return o;
}

std::string jstring_to_std(JNIEnv* env, jstring js) {
    if (js == nullptr) return {};
    const char* p = env->GetStringUTFChars(js, nullptr);
    std::string out = p ? p : "";
    if (p != nullptr) env->ReleaseStringUTFChars(js, p);
    return out;
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

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeSetAccelBackend(JNIEnv*, jclass, jlong h, jint backend) {
    session(h)->set_accel_backend(parse_accel_backend(backend));
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeSetFlowOptions(JNIEnv* env, jclass, jlong h, jint flowMethod,
                                                                 jstring jflowModelPath) {
    session(h)->set_flow_options(parse_flow_method(flowMethod), jstring_to_std(env, jflowModelPath));
}

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
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeMerge(JNIEnv* env, jclass, jlong h, jfloatArray jopt,
                                                         jboolean pmridEnabled, jfloat pmridLowNoiseCutoff,
                                                         jstring jpmridModelPath, jint accelBackend,
                                                         jint flowMethod, jstring jflowModelPath) {
    BurstOptions o = burst_options(env, jopt);
    o.pmridEnabled = pmridEnabled != JNI_FALSE;
    o.pmridLowNoiseCutoff = pmridLowNoiseCutoff;
    o.pmridModelPath = jstring_to_std(env, jpmridModelPath);
    o.accelBackend = parse_accel_backend(accelBackend);
    o.flowMethod = parse_flow_method(flowMethod);
    o.flowModelPath = jstring_to_std(env, jflowModelPath);
    guarded(env, [&] { session(h)->merge(o); });
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeBeginStream(JNIEnv* env, jclass, jlong h, jfloatArray jopt,
                                                               jboolean pmridEnabled, jfloat pmridLowNoiseCutoff,
                                                               jstring jpmridModelPath, jint accelBackend,
                                                               jint flowMethod, jstring jflowModelPath,
                                                               jboolean earlyStopEnabled, jfloat earlyStopTargetSnrDb) {
    BurstOptions o = burst_options(env, jopt);
    o.pmridEnabled = pmridEnabled != JNI_FALSE;
    o.pmridLowNoiseCutoff = pmridLowNoiseCutoff;
    o.pmridModelPath = jstring_to_std(env, jpmridModelPath);
    o.accelBackend = parse_accel_backend(accelBackend);
    o.flowMethod = parse_flow_method(flowMethod);
    o.flowModelPath = jstring_to_std(env, jflowModelPath);
    o.earlyStopEnabled = earlyStopEnabled != JNI_FALSE;
    o.earlyStopTargetSnrDb = earlyStopTargetSnrDb;
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
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeAddFrameProgress(JNIEnv* env, jclass, jlong h, jobject buf, jint w, jint ht,
                                                                      jfloatArray jout) {
    auto* p = static_cast<const uint16_t*>(env->GetDirectBufferAddress(buf));
    if (p == nullptr || env->GetDirectBufferCapacity(buf) < (jlong)w * ht * 2) {
        throwState(env, "RAW frame is not a direct buffer of width*height*2 bytes");
        return -1;
    }
    Raw r;
    r.w = w;
    r.h = ht;
    r.d = p;
    StreamProgress sp;
    guarded(env, [&] { sp = session(h)->add_frame_progress(r); });
    if (jout != nullptr && env->GetArrayLength(jout) >= 6) {
        jfloat out[6] = {
            static_cast<jfloat>(sp.refIdx),
            static_cast<jfloat>(sp.acceptedCount),
            sp.neff,
            sp.phaseCoverage,
            sp.estimatedSnrGainDb,
            sp.shouldStop ? 1.0f : 0.0f,
        };
        env->SetFloatArrayRegion(jout, 0, 6, out);
    }
    return sp.refIdx;
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
                                                         jstring jlutPath, jfloat lutStrength, jboolean keepHdr,
                                                         jint tonemapMode, jstring jhdrnetModelPath,
                                                         jboolean semanticSegEnabled, jstring jsegModelPath,
                                                         jfloatArray jfaceBoxes, jint orientationDegrees) {
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
    std::string lutPath = jstring_to_std(env, jlutPath);
    const char* style = env->GetStringUTFChars(jstyle, nullptr);
    if (std::string(style) != "default") apply_look_preset(style, lp);
    env->ReleaseStringUTFChars(jstyle, style);
    // Apply non-preset finishing options after apply_look_preset so preset selection does not overwrite them.
    lp.lutPath = std::move(lutPath);
    lp.lutStrength = lutStrength;
    lp.tonemapMode = parse_tonemap_mode(tonemapMode);
    lp.hdrnetModelPath = jstring_to_std(env, jhdrnetModelPath);
    lp.semanticSegEnabled = semanticSegEnabled != JNI_FALSE;
    lp.segModelPath = jstring_to_std(env, jsegModelPath);
    lp.orientationDegrees = orientationDegrees;
    if (jfaceBoxes != nullptr) {
        const jsize nBoxes = env->GetArrayLength(jfaceBoxes);
        if (nBoxes >= 4) {
            lp.faceBoxes.resize((size_t)(nBoxes / 4) * 4);
            env->GetFloatArrayRegion(jfaceBoxes, 0, (jsize)lp.faceBoxes.size(), lp.faceBoxes.data());
        }
    }
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
    pool_trim();
    Gpu::release_memory();
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

JNIEXPORT jstring JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeAccelProbe(JNIEnv* env, jclass) {
    return env->NewStringUTF(probe_accel_capabilities().summary().c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeUnpackRaw10(JNIEnv* env, jclass, jobject jsrc, jint width, jint height,
                                                                jint rowStride, jobject jdst) {
    if (jsrc == nullptr || jdst == nullptr || width <= 0 || height <= 0) return JNI_FALSE;
    const auto* src = static_cast<const uint8_t*>(env->GetDirectBufferAddress(jsrc));
    const jlong srcCap = env->GetDirectBufferCapacity(jsrc);
    auto* dst = static_cast<uint16_t*>(env->GetDirectBufferAddress(jdst));
    const jlong dstCap = env->GetDirectBufferCapacity(jdst);
    if (src == nullptr || dst == nullptr || srcCap <= 0 || dstCap < (jlong)width * height * 2) return JNI_FALSE;
    return unpack_raw10(src, static_cast<size_t>(srcCap), width, height, rowStride, dst) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobject JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeMemory_nativeAllocate(JNIEnv* env, jclass, jlong bytes) {
    void* p = malloc((size_t)bytes);
    if (p == nullptr) return nullptr;
    return env->NewDirectByteBuffer(p, bytes);
}

JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeMemory_nativeFree(JNIEnv* env, jclass, jobject buffer) {
    free(env->GetDirectBufferAddress(buffer));
}

}  // extern "C"
