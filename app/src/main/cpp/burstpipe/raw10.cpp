#include "raw10.h"

#include <algorithm>
#include <limits>

size_t raw10_packed_row_bytes(int width) {
    if (width <= 0) return 0;
    return (static_cast<size_t>(width) * 10u + 7u) / 8u;
}

size_t raw10_packed_size(int width, int height) {
    if (width <= 0 || height <= 0) return 0;
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / h / 10u) return 0;
    return (w * h * 10u + 7u) / 8u;
}

bool unpack_raw10(const uint8_t* src, size_t srcBytes, int width, int height, int rowStride, uint16_t* dst) {
    if (!src || !dst || width <= 0 || height <= 0 || (width % 4) != 0 || rowStride < 0) return false;
    const size_t rowBytes = (static_cast<size_t>(width) / 4u) * 5u;
    const size_t stride = rowStride > 0 ? static_cast<size_t>(rowStride) : rowBytes;
    if (stride < rowBytes) return false;
    const size_t rowsMinusOne = static_cast<size_t>(height - 1);
    if (rowsMinusOne > (std::numeric_limits<size_t>::max() - rowBytes) / stride) return false;
    const size_t minRequired = rowsMinusOne * stride + rowBytes;
    if (srcBytes < minRequired) return false;

    const int groups = width / 4;
    for (int y = 0; y < height; y++) {
        const uint8_t* s = src + static_cast<size_t>(y) * stride;
        uint16_t* d = dst + static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int g = 0; g < groups; g++, s += 5, d += 4) {
            const uint8_t lsb = s[4];
            d[0] = static_cast<uint16_t>((static_cast<uint16_t>(s[0]) << 2) | (lsb & 0x03u));
            d[1] = static_cast<uint16_t>((static_cast<uint16_t>(s[1]) << 2) | ((lsb >> 2) & 0x03u));
            d[2] = static_cast<uint16_t>((static_cast<uint16_t>(s[2]) << 2) | ((lsb >> 4) & 0x03u));
            d[3] = static_cast<uint16_t>((static_cast<uint16_t>(s[3]) << 2) | ((lsb >> 6) & 0x03u));
        }
    }
    return true;
}

bool pack_raw10(const uint16_t* src, int width, int height, uint8_t* dst, size_t dstBytes) {
    if (!src || !dst || width <= 0 || height <= 0 || (width % 4) != 0) return false;
    const size_t rowBytes = (static_cast<size_t>(width) / 4u) * 5u;
    const size_t total = rowBytes * static_cast<size_t>(height);
    if (dstBytes < total) return false;

    const int groups = width / 4;
    for (int y = 0; y < height; y++) {
        const uint16_t* s = src + static_cast<size_t>(y) * static_cast<size_t>(width);
        uint8_t* d = dst + static_cast<size_t>(y) * rowBytes;
        for (int g = 0; g < groups; g++, s += 4, d += 5) {
            const uint16_t p0 = std::min<uint16_t>(s[0], 1023u);
            const uint16_t p1 = std::min<uint16_t>(s[1], 1023u);
            const uint16_t p2 = std::min<uint16_t>(s[2], 1023u);
            const uint16_t p3 = std::min<uint16_t>(s[3], 1023u);
            d[0] = static_cast<uint8_t>(p0 >> 2);
            d[1] = static_cast<uint8_t>(p1 >> 2);
            d[2] = static_cast<uint8_t>(p2 >> 2);
            d[3] = static_cast<uint8_t>(p3 >> 2);
            d[4] = static_cast<uint8_t>((p0 & 0x03u) | ((p1 & 0x03u) << 2) | ((p2 & 0x03u) << 4) | ((p3 & 0x03u) << 6));
        }
    }
    return true;
}
