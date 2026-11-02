#include <android/bitmap.h>
#include <jni.h>

#include <cstdlib>

#include "burstpipe/burst.h"

namespace {

BurstSession* session(jlong handle) { return reinterpret_cast<BurstSession*>(handle); }

void throwState(JNIEnv* env, const char* msg) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), msg);
}

}  // namespace

extern "C" {

// meta: width, height, cfa, white, black[4] (R, G_even, G_odd, B), wb[4], ccm[9] (row-major),
// lscRows, lscCols; lsc: rows*cols*4 gains (R, G_even, G_odd, B) or null; exposureNs / iso per frame.
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

    int n = env->GetArrayLength(jframes);
    if (env->GetArrayLength(jexp) == n && env->GetArrayLength(jiso) == n) {
        std::vector<jlong> e(n);
        std::vector<jint> iso(n);
        env->GetLongArrayRegion(jexp, 0, n, e.data());
        env->GetIntArrayRegion(jiso, 0, n, iso.data());
        for (int k = 0; k < n; k++) { m.exposureNs.push_back((double)e[k]); m.iso.push_back(iso[k]); }
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
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeAlign(JNIEnv*, jclass, jlong h) { session(h)->align(); }

// options: kernelRegression, robustLo, robustHi, dilate, demosaic (0 MHC, 1 RCD, 2 AMaZE), freqMerge, freqC,
// freqSpatial, lcaR[2], lcaB[2]
JNIEXPORT void JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeMerge(JNIEnv* env, jclass, jlong h, jfloatArray jopt) {
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
    session(h)->merge(o);
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
    s->finish(lp, keepHdr);
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
    s->render(static_cast<uint8_t*>(pixels), info.stride);
    AndroidBitmap_unlockPixels(env, bitmap);
}

// The finished image as an Ultra HDR JPEG (gain map at 1/4 resolution), or null when finish() did not keep
// the HDR luminance or the encoder failed.
JNIEXPORT jbyteArray JNICALL
Java_com_izum1k_pocopaw_pipeline_NativeBurst_nativeEncodeUltraHdr(JNIEnv* env, jclass, jlong h, jint quality, jfloat headroom) {
    std::vector<uint8_t> file = session(h)->encode_ultrahdr(quality, headroom);
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
