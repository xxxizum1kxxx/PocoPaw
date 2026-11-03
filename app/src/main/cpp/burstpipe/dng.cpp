#include "dng.h"

#include <algorithm>
#include <cstring>

namespace {

enum Type : uint16_t { BYTE = 1, ASCII = 2, SHORT = 3, LONG = 4, RATIONAL = 5, UNDEFINED = 7, SRATIONAL = 10 };

struct Entry {
    uint16_t tag, type;
    uint32_t count;
    std::vector<uint8_t> data;
};

void put16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xff); b.push_back(v >> 8); }
void put32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((v >> (8 * i)) & 0xff); }
void put32be(std::vector<uint8_t>& b, uint32_t v) { for (int i = 3; i >= 0; i--) b.push_back((v >> (8 * i)) & 0xff); }
void put64be(std::vector<uint8_t>& b, double d) {
    uint64_t v;
    memcpy(&v, &d, 8);
    for (int i = 7; i >= 0; i--) b.push_back((v >> (8 * i)) & 0xff);
}
void putf32be(std::vector<uint8_t>& b, float f) {
    uint32_t v;
    memcpy(&v, &f, 4);
    put32be(b, v);
}

class Ifd {
public:
    void shorts(uint16_t tag, std::initializer_list<uint16_t> v) {
        Entry e{tag, SHORT, (uint32_t)v.size(), {}};
        for (uint16_t s : v) put16(e.data, s);
        add(std::move(e));
    }
    void longs(uint16_t tag, std::initializer_list<uint32_t> v) {
        Entry e{tag, LONG, (uint32_t)v.size(), {}};
        for (uint32_t s : v) put32(e.data, s);
        add(std::move(e));
    }
    void bytes(uint16_t tag, Type type, std::vector<uint8_t> v) {
        Entry e{tag, type, (uint32_t)v.size(), std::move(v)};
        add(std::move(e));
    }
    void ascii(uint16_t tag, const std::string& s) {
        std::vector<uint8_t> v(s.begin(), s.end());
        v.push_back(0);
        bytes(tag, ASCII, std::move(v));
    }
    void rationals(uint16_t tag, bool sign, const float* v, int n) {
        Entry e{tag, sign ? SRATIONAL : RATIONAL, (uint32_t)n, {}};
        for (int i = 0; i < n; i++) {
            put32(e.data, sign ? (uint32_t)(int32_t)std::lround(v[i] * 10000.0) : (uint32_t)std::lround(std::max(0.f, v[i]) * 10000.0));
            put32(e.data, 10000);
        }
        add(std::move(e));
    }
    // appends the IFD at the end of `out`, values over 4 bytes after it, all 4-byte aligned; returns its offset
    uint32_t write(std::vector<uint8_t>& out, uint32_t next = 0) {
        while (out.size() & 3) out.push_back(0);
        std::sort(e_.begin(), e_.end(), [](const Entry& a, const Entry& b) { return a.tag < b.tag; });
        const uint32_t at = (uint32_t)out.size();
        uint32_t extra = at + 2 + 12 * (uint32_t)e_.size() + 4;
        std::vector<uint8_t> tail;
        put16(out, (uint16_t)e_.size());
        for (const Entry& e : e_) {
            put16(out, e.tag);
            put16(out, e.type);
            put32(out, e.count);
            if (e.data.size() <= 4) {
                std::vector<uint8_t> v = e.data;
                v.resize(4, 0);
                out.insert(out.end(), v.begin(), v.end());
            } else {
                put32(out, extra + (uint32_t)tail.size());
                tail.insert(tail.end(), e.data.begin(), e.data.end());
                while (tail.size() & 3) tail.push_back(0);
            }
        }
        put32(out, next);
        out.insert(out.end(), tail.begin(), tail.end());
        return at;
    }

private:
    void add(Entry e) { e_.push_back(std::move(e)); }
    std::vector<Entry> e_;
};

// CFAPattern (0 R, 1 G, 2 B) of a Camera2 CFA enum: RGGB, GRBG, GBRG, BGGR
std::vector<uint8_t> cfa_pattern(int cfa) {
    std::vector<uint8_t> p(4);
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++) p[y * 2 + x] = (uint8_t)cfa_color(cfa, x, y);
    return p;
}

// OpcodeList2 with one GainMap (opcode 9) per CFA position; the map spans the full sensor, placed on the crop
std::vector<uint8_t> gain_maps(const Meta& m) {
    std::vector<uint8_t> b;
    if (m.lscRows < 2 || m.lscCols < 2 || m.lsc.size() < (size_t)m.lscRows * m.lscCols * 4) return b;
    put32be(b, 4);
    const double fw = m.full_w(), fh = m.full_h();
    for (int y0 = 0; y0 < 2; y0++)
        for (int x0 = 0; x0 < 2; x0++) {
            std::vector<uint8_t> p;
            put32be(p, y0); put32be(p, x0); put32be(p, m.height); put32be(p, m.width);  // top, left, bottom, right
            put32be(p, 0); put32be(p, 1); put32be(p, 2); put32be(p, 2);              // plane, planes, row / col pitch
            put32be(p, m.lscRows); put32be(p, m.lscCols);
            put64be(p, fh / ((m.lscRows - 1) * (double)m.height));
            put64be(p, fw / ((m.lscCols - 1) * (double)m.width));
            put64be(p, -m.cropY / (double)m.height);
            put64be(p, -m.cropX / (double)m.width);
            put32be(p, 1);
            const int ch = cfa_channel(m.cfa, x0, y0);
            for (int r = 0; r < m.lscRows; r++)
                for (int c = 0; c < m.lscCols; c++) putf32be(p, m.lsc[((size_t)r * m.lscCols + c) * 4 + ch]);
            put32be(b, 9);
            put32be(b, 0x01030000);
            put32be(b, 1);  // optional: macOS rejects a file with a required opcode it does not apply
            put32be(b, (uint32_t)p.size());
            b.insert(b.end(), p.begin(), p.end());
        }
    return b;
}

}  // namespace

int dng_scale(float white) { return std::max(1, std::min(kDngScale, (int)(65535.f / std::max(1.f, white)))); }

std::vector<uint8_t> encode_dng(const ImageF& mosaic, const Meta& m, const DngColor& c, const DngExtras& x) {
    const int W = mosaic.w, H = mosaic.h, s = dng_scale(m.white);
    const bool thumb = x.thumbW > 0 && x.thumbH > 0 && x.thumb.size() >= (size_t)x.thumbW * x.thumbH * 3;
    const uint32_t thumbBytes = thumb ? (uint32_t)x.thumbW * x.thumbH * 3 : 0, rawBytes = (uint32_t)W * H * 2;
    float black[4], white = m.white * s;
    for (int y = 0; y < 2; y++)
        for (int xx = 0; xx < 2; xx++) black[y * 2 + xx] = m.black[cfa_channel(m.cfa, xx, y)] * s;
    const std::vector<uint8_t> ops = gain_maps(m);

    auto rawIfd = [&](uint32_t rawOff, uint32_t sub) {
        Ifd r;
        r.longs(254, {0});
        r.longs(256, {(uint32_t)W});
        r.longs(257, {(uint32_t)H});
        r.shorts(258, {16});
        r.shorts(259, {1});
        r.shorts(262, {32803});
        r.longs(273, {rawOff});
        r.shorts(277, {1});
        r.longs(278, {(uint32_t)H});
        r.longs(279, {rawBytes});
        r.shorts(284, {1});
        if (thumb) r.longs(330, {sub});
        r.shorts(33421, {2, 2});
        r.bytes(33422, BYTE, cfa_pattern(m.cfa));
        r.bytes(50710, BYTE, {0, 1, 2});
        r.shorts(50711, {1});
        r.shorts(50713, {2, 2});
        r.rationals(50714, false, black, 4);
        r.longs(50717, {(uint32_t)std::lround(white)});
        if (!ops.empty()) r.bytes(51009, UNDEFINED, ops);
        if (!x.make.empty()) r.ascii(271, x.make);
        if (!x.model.empty()) r.ascii(272, x.model);
        r.shorts(274, {(uint16_t)x.orientation});
        if (!x.software.empty()) r.ascii(305, x.software);
        r.bytes(50706, BYTE, {1, 4, 0, 0});
        r.bytes(50707, BYTE, {1, 3, 0, 0});
        r.ascii(50708, x.make + " " + x.model);
        for (int i = 0; i < c.illuminants; i++) {
            r.rationals(i ? 50722 : 50721, true, c.colorMatrix[i], 9);
            r.shorts(i ? 50779 : 50778, {(uint16_t)c.illuminant[i]});
            r.rationals(i ? 50965 : 50964, true, c.forwardMatrix[i], 9);
        }
        r.rationals(50728, false, c.neutral, 3);
        if (x.baselineExposure != 0) r.rationals(50730, true, &x.baselineExposure, 1);
        return r;
    };
    auto thumbIfd = [&](uint32_t thumbOff) {
        Ifd t;
        t.longs(254, {1});
        t.longs(256, {(uint32_t)x.thumbW});
        t.longs(257, {(uint32_t)x.thumbH});
        t.shorts(258, {8, 8, 8});
        t.shorts(259, {1});
        t.shorts(262, {2});
        t.longs(273, {thumbOff});
        t.shorts(277, {3});
        t.longs(278, {(uint32_t)x.thumbH});
        t.longs(279, {thumbBytes});
        t.shorts(284, {1});
        return t;
    };
    // the IFDs come first (macOS reads opcode lists only from a file laid out so), their sizes do not depend
    // on the offsets they hold: one dry run places the pixels
    std::vector<uint8_t> out = {'I', 'I', 42, 0, 8, 0, 0, 0};
    std::vector<uint8_t> dry = out;
    rawIfd(0, 0).write(dry);
    const uint32_t sub = (uint32_t)((dry.size() + 3) & ~(size_t)3);
    if (thumb) thumbIfd(0).write(dry);
    const uint32_t thumbOff = (uint32_t)((dry.size() + 3) & ~(size_t)3);
    const uint32_t rawOff = (thumbOff + thumbBytes + 3) & ~3u;

    out.reserve((size_t)rawOff + rawBytes);
    rawIfd(rawOff, sub).write(out);
    if (thumb) thumbIfd(thumbOff).write(out);
    out.resize(thumbOff, 0);
    out.insert(out.end(), x.thumb.begin(), x.thumb.begin() + thumbBytes);
    out.resize((size_t)rawOff + rawBytes, 0);
    parallel_for(H, [&](int y) {
        uint8_t* d = out.data() + rawOff + (size_t)y * W * 2;
        const float* r = mosaic.row(y);
        for (int i = 0; i < W; i++) {
            int v = clampi((int)std::lround(r[i] * s), 0, 65535);
            d[2 * i] = v & 0xff;
            d[2 * i + 1] = v >> 8;
        }
    });
    return out;
}
