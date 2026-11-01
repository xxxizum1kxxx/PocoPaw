#pragma once
#include "common.h"

enum class Demosaic { MHC, RCD, AMAZE };

bool parse_demosaic(const std::string& s, Demosaic& d);
const char* demosaic_name(Demosaic d);

// Normalised Bayer float mosaic ([0,1], WB applied) -> interleaved linear camera RGB.
// RCD and AMaZE come from librtprocess (GPL-3); they run single-threaded per horizontal strip.
ImageF demosaic(const ImageF& cfa, int cfaType, Demosaic method);
