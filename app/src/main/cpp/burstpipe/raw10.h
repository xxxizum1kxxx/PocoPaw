#pragma once
#include <cstddef>
#include <cstdint>

size_t raw10_packed_row_bytes(int width);
size_t raw10_packed_size(int width, int height);
bool unpack_raw10(const uint8_t* src, size_t srcBytes, int width, int height, int rowStride, uint16_t* dst);
bool pack_raw10(const uint16_t* src, int width, int height, uint8_t* dst, size_t dstBytes);
