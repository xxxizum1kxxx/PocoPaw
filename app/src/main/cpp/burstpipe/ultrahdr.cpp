#include "ultrahdr.h"

#include <fstream>

#include "finish.h"
#include "third_party/libultrahdr/include/ultrahdr_api.h"

HdrLuma hdr_luma(const ImageF& rgb, int ds) {
    HdrLuma o;
    const int b = o.block * std::max(1, ds);
    const int w = (rgb.w / std::max(1, ds) + o.block - 1) / o.block, h = (rgb.h / std::max(1, ds) + o.block - 1) / o.block;
    o.y = ImageF(w, h);
    parallel_for(h, [&](int by) {
        for (int bx = 0; bx < w; bx++) {
            double s = 0;
            int n = 0;
            for (int y = by * b; y < std::min(rgb.h, (by + 1) * b); y++) {
                const float* q = rgb.row(y);
                for (int x = bx * b; x < std::min(rgb.w, (bx + 1) * b); x++, n++)
                    s += 0.2126f * q[3 * x] + 0.7152f * q[3 * x + 1] + 0.0722f * q[3 * x + 2];
            }
            o.y.at(bx, by) = n ? (float)(s / n) : 0.f;
        }
    });
    return o;
}

namespace {
float srgb_to_linear(float v) { return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f); }

// identity up to SDR white, then exponential shoulder that approaches `peak`
float rolloff(float y, float peak) {
    constexpr float knee = 1.f;
    if (y <= knee) return y;
    float r = peak - knee;
    return knee + r * (1.f - std::exp(-(y - knee) / r));
}

float smoothstep(float a, float b, float x) {
    float t = clampf((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3 - 2 * t);
}
}  // namespace

std::vector<uint8_t> encode_ultrahdr(const ImageF& sdr, int ds, const HdrLuma& hdr, float exposure, float headroom,
                                     int quality, UltraHdrStats* st) {
    int w, h;
    std::vector<uint8_t> px = to_rgb8(sdr, ds, w, h);
    std::vector<uint8_t> base = encode_jpeg(px.data(), w, h, 3, quality);

    float lut[256];
    for (int i = 0; i < 256; i++) lut[i] = srgb_to_linear(i / 255.f);
    const int B = hdr.block, gw = hdr.y.w, gh = hdr.y.h;
    constexpr float kOffset = 1.f / 64;  // offset_sdr = offset_hdr, keeps the ratio finite in black areas
    ImageF gain(gw, gh);
    parallel_for(gh, [&](int by) {
        for (int bx = 0; bx < gw; bx++) {
            double ylin = 0, ydisp = 0;
            int n = 0;
            for (int y = by * B; y < std::min(h, (by + 1) * B); y++)
                for (int x = bx * B; x < std::min(w, (bx + 1) * B); x++, n++) {
                    const uint8_t* p = &px[((size_t)y * w + x) * 3];
                    ylin += 0.2126f * lut[p[0]] + 0.7152f * lut[p[1]] + 0.0722f * lut[p[2]];
                    ydisp += (0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) / 255.f;
                }
            ylin /= std::max(n, 1); ydisp /= std::max(n, 1);
            float yh = rolloff(hdr.y.at(bx, by) * exposure, headroom);
            float g = (yh + kOffset) / ((float)ylin + kOffset);
            // only brighten highlights the SDR rendition compressed; shadows and midtones keep the SDR look
            float wgt = smoothstep(0.35f, 0.75f, (float)ydisp);
            gain.at(bx, by) = clampf(1.f + wgt * (std::max(g, 1.f) - 1.f), 1.f, headroom);
        }
    });
    // 3x3 box in the log domain: the map is upsampled bilinearly by the decoder, this just removes noise
    ImageF lg(gw, gh);
    parallel_for(gh, [&](int y) {
        for (int x = 0; x < gw; x++) {
            float s = 0;
            int n = 0;
            for (int v = std::max(0, y - 1); v <= std::min(gh - 1, y + 1); v++)
                for (int u = std::max(0, x - 1); u <= std::min(gw - 1, x + 1); u++, n++) s += std::log2(gain.at(u, v));
            lg.at(x, y) = s / n;
        }
    });
    float lmax = 0;
    double lsum = 0;
    size_t boosted = 0;
    for (float v : lg.d) { lmax = std::max(lmax, v); lsum += v; boosted += v > 0.05f; }
    if (st) {
        st->maxBoost = std::exp2(lmax);
        st->meanBoost = (float)std::exp2(lsum / lg.d.size());
        st->boostedFrac = (float)boosted / lg.d.size();
    }
    // nothing worth lifting: the quantised map would only carry noise
    if (lmax < std::log2(kMinBoost)) return {};
    std::vector<uint8_t> gm((size_t)gw * gh);
    for (size_t i = 0; i < gm.size(); i++) gm[i] = (uint8_t)clampi((int)std::lround(lg.d[i] / lmax * 255.f), 0, 255);
    std::vector<uint8_t> gmJpeg = encode_jpeg(gm.data(), gw, gh, 1, 90);

    uhdr_codec_private_t* enc = uhdr_create_encoder();
    uhdr_compressed_image_t bi{base.data(), base.size(), base.size(), UHDR_CG_BT_709, UHDR_CT_SRGB, UHDR_CR_FULL_RANGE};
    uhdr_compressed_image_t gi{gmJpeg.data(), gmJpeg.size(), gmJpeg.size(), UHDR_CG_UNSPECIFIED, UHDR_CT_UNSPECIFIED,
                               UHDR_CR_FULL_RANGE};
    uhdr_gainmap_metadata_t md{};
    for (int c = 0; c < 3; c++) {
        md.max_content_boost[c] = std::exp2(lmax);
        md.min_content_boost[c] = 1.f;
        md.gamma[c] = 1.f;
        md.offset_sdr[c] = md.offset_hdr[c] = kOffset;
    }
    md.hdr_capacity_min = 1.f;
    md.hdr_capacity_max = std::exp2(lmax);
    md.use_base_cg = 1;
    std::vector<uint8_t> file;
    uhdr_error_info_t e = uhdr_enc_set_compressed_image(enc, &bi, UHDR_BASE_IMG);
    if (e.error_code == UHDR_CODEC_OK) e = uhdr_enc_set_gainmap_image(enc, &gi, &md);
    if (e.error_code == UHDR_CODEC_OK) e = uhdr_encode(enc);
    if (e.error_code == UHDR_CODEC_OK) {
        uhdr_compressed_image_t* out = uhdr_get_encoded_stream(enc);
        const uint8_t* d = (const uint8_t*)out->data;
        file.assign(d, d + out->data_sz);
        if (st) st->bytes = out->data_sz;
    } else {
        fprintf(stderr, "ultrahdr: %s\n", e.has_detail ? e.detail : "encode failed");
    }
    uhdr_release_encoder(enc);
    if (st) st->gainMapBytes = gmJpeg.size();
    return file;
}

bool write_ultrahdr(const std::string& path, const ImageF& sdr, int ds, const HdrLuma& hdr, float exposure,
                    float headroom, int quality, UltraHdrStats* st) {
    std::vector<uint8_t> file = encode_ultrahdr(sdr, ds, hdr, exposure, headroom, quality, st);
    if (file.empty()) return false;
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)file.data(), (std::streamsize)file.size());
    return (bool)f;
}
