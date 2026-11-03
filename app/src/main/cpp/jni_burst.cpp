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

}  // namespace

extern "C" {

// meta: width, height, cfa, white, black[4] (R, G_even, G_odd, B), wb[4], ccm[9] (row-major),
// lscRows, lscCols; lsc: rows*cols*4 gains (R, G_even, G_odd, B) or null; exposureNs / iso per frame (of the
// first frame only for a session with no frames, which merges frames as they are added).
JNIEXPORT jlong JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeCreate(JNIEnv* env, jclass, jfloatArray jmeta, jfloatArray jlsc,
                                                         jobjectArray jframes, jint ref, jlongArray jexp, jintArray jiso) {
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
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeRender(JNIEnv* env, jclass, jlong h, jobject bitmap) {
    BurstSession* s = session(h);
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || (int)info.width != s->width() || (int)info.height != s->height()) {
        throwState(env, "bitmap must be ARGB_8888 of the rendered size");
        return;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throwState(env, "cannot lock bitmap");
        return;
    }
    guarded(env, [&] { s->render(static_cast<uint8_t*>(pixels), info.stride); });
    AndroidBitmap_unlockPixels(env, bitmap);
}

// An ARGB_8888 bitmap as a 4:2:0 JPEG (libjpeg-turbo, strip-parallel), or null.
JNIEXPORT jbyteArray JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeEncodeJpeg(JNIEnv* env, jclass, jobject bitmap, jint quality) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS)
        return nullptr;
    const int w = (int)info.width, h = (int)info.height;
    std::vector<uint8_t> file;
    guarded(env, [&] {
        const uint8_t* px = static_cast<const uint8_t*>(pixels);
        std::vector<uint8_t> packed;
        if (info.stride != (uint32_t)w * 4) {
            packed.resize((size_t)w * h * 4);
            for (int y = 0; y < h; y++) memcpy(&packed[(size_t)y * w * 4], px + (size_t)y * info.stride, (size_t)w * 4);
            px = packed.data();
        }
        file = encode_jpeg(px, w, h, 4, quality, true);
    });
    AndroidBitmap_unlockPixels(env, bitmap);
    if (file.empty()) return nullptr;
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

// The finished image as an Ultra HDR JPEG (gain map at 1/4 resolution), or null when finish() did not keep
// the HDR luminance or the encoder failed.
JNIEXPORT jbyteArray JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeEncodeUltraHdr(JNIEnv* env, jclass, jlong h, jint quality, jfloat headroom) {
    std::vector<uint8_t> file;
    guarded(env, [&] { file = session(h)->encode_ultrahdr(quality, headroom); });
    if (file.empty()) return nullptr;
    jbyteArray out = env->NewByteArray((jsize)file.size());
    if (out != nullptr) env->SetByteArrayRegion(out, 0, (jsize)file.size(), reinterpret_cast<const jbyte*>(file.data()));
    return out;
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
