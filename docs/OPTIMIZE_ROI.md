# Poco F5 (`marble`) Pipeline ROI Audit & Hardware Acceleration Report

## Overview

- **Device**: Poco F5 / Redmi Note 12 Turbo (`marble`, device `848893fa`)
- **SoC & GPU**: Qualcomm Snapdragon 7+ Gen 2 (`SM7475`: `4× Cortex-A510` @ 1.8 GHz + `3× Cortex-A710` @ 2.5 GHz + `1× Cortex-X2` @ 2.91 GHz) + `Adreno 725` (`OpenCL 3.0`, Wave128, unified LPDDR5)
- **Toolchain**: Android NDK `29.0.14206865`, CMake `3.22.1`, `-O3 -ffast-math -march=armv8.2-a+fp16+dotprod -flto=thin -fvisibility=hidden -Wl,--gc-sections`
- **Target Package**: `com.izum1k.pocopaw.opt` (`PocoPaw Opt`, side-by-side installable alongside `com.izum1k.pocopaw`)
- **Rendering Mandate**: Single unadjustable multi-frame RAW pipeline (`IndigoHdrPlusPipeline`) locked to the **Project Indigo + Original Pixel (Pixel 1 HDR+)** aesthetic:
  1. **SLR-like highlight & shadow rolloff**: `-1.0 EV` to `-2.0 EV` highlight-clipping exposure guard + `1/4`-resolution Mertens synthetic exposure fusion (`targetKey = 0.20`, `compression = 5.0`) + perceptual Oklab constant-hue gamut compression and specular bleach.
  2. **Zero ringing halos**: `localContrast = 0.0` and noise-cored luminance USM (`sharpen = 0.35`, `sharpenRadius = 0.85`), cutting optical step-edge overshoot from `7.33%` (`dev-exp` Pixel look) to **`0.64%`** (`+0.10%` above the `0.54%` unsharpened optical floor).
  3. **Natural organic fine grain**: 8-frame spatial robust merge with calibrated sensor noise and symmetric shot-noise variance plus luminance-ratio chroma denoise; all texture-wiping CNN denoisers (`PMRID`) and frequency blurs (`merge_freq`) removed.
  4. **Faithful physical color**: Camera2 `HAL AWB` + scene-tuned `HAL CCM` with automatic DNG Section 6 dual-illuminant CCM fallback; no selective sky/skin hue twists, 3D film LUTs, or bilateral-grid color shifts.

---

## 1. Pre-Optimization Baseline & Aesthetic Comparison (`4000×3000` 12MP, 8 Frames)

All baseline and candidate configurations across `main` (`e772292`) and `dev-exp` (`6ade25d` M1–M5) were benchmarked headlessly on Poco F5 (`848893fa`) using real 8-frame bursts (`hh_main8_12mp`, `br_main_backlit`, and native 16MP `4624×3472` `main8`) plus a controlled 12MP step-edge / highlight-disc / log-wedge target.

| Configuration (`4000×3000` 12MP, 8f) | Cold Init + Run (ms) | Warm Median (ms) | Peak RSS (MB) | Flat SNR (dB) / $\sigma_Y$ | Tex HF $p_{80}$ | Hi Clip (%) | Chroma Halo | Step Edge Overshoot / Undershoot | Audit Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`main` CPU 8T** (`RCD`, Default) | 1043.2 | 989.6 | 871.5 | 46.20 / 0.701 | 1.708 | 0.002% (`s5` 2.86%) | 2.557 | **5.92% / 6.36%** | CPU fallback ref (`75.08 dB` vs GPU) |
| **`main` CPU 4T Big** (`CPUs 4..7`) | 1402.2 | 1313.9 | 863.7 | 46.20 / 0.701 | 1.708 | 0.002% | 2.557 | 5.92% / 6.36% | **-324 ms slower** than 8T dynamic steal |
| **`main` OpenCL Batch** (`RCD`, Default) | 1951.9 (`749.9` CL) | 936.2 | 598.0 | 43.32 / 0.701 | 1.708 | 0.002% (`s5` 2.86%) | 2.557 | 5.92% / 6.36% | Fast, but USM/LC rings & RGB clips |
| **`main` OpenCL Stream** (`BurstStream`) | 1011.2 (`.clbin` 14 ms) | **919.6** | **382.0** | 43.32 / 0.701 | 1.708 | 0.002% | 2.557 | 5.92% / 6.36% | **-216 MB RSS**, ~350 ms hidden in capture |
| **`dev-exp` OpenCL Pixel+UHDR+DNG** | 2526.9 (`1080.8` CL) | 1281.3 | 709.7 | 43.12 / **0.829** | 2.202 | 0.002% (`s5` 3.29%) | 3.650 | **7.33% / 7.29%** | Halos (`7.3%`), `+53%` harsher grain |
| **`dev-exp` OpenCL + HDRnet** (`hdrnet.onnx`) | 1354.6 | 1107.9 | 772.0 | 47.83 / 0.557 | 2.817 | **10.446%** | **8.527** | 2.44% / 5.80% | **24× worse highlight clip**, `114°` hue twist |
| **`dev-exp` OpenCL + PMRID** (`pmrid.onnx`, 4f) | 3591.6 | **3345.2** | **2502.1** | 55.67 / 0.266 | **0.313 (-88%)** | 0.002% | 9.437 | 0.58% / 0.00% | **2.5 GB OOM risk + plastic texture wipeout** |
| **`dev-exp` OpenCL + SEA-RAFT** (`4f`) | 4675.4 | **3957.5** | 671.7 | 43.30 / 0.973 | 2.582 | 0.002% | 9.130 | 5.87% / 3.26% | **3.11 s** flow stall, `-0.77 dB` vs block-match |
| **Unsharpened Optical Baseline** (`sh=0, lc=0`) | 1156.9 | 1005.0 | 599.4 | 52.04 / 0.429 | 1.216 | 0.425% (synth) | 0.615 | **0.54%** / 1.76% | Pure RCD optical floor (0% USM halo) |
| **Locked Indigo-Pixel1** (`sh=0.35, r=0.85, lc=0`) | 1220.3 (`.clbin` 14 ms) | **1062.7** (pre-opt) | 599.4 (`384` stream) | **46.49 / 0.543** (`49.9` syn) | **1.383** (`+13.7%`) | **0.001%** (`0.437%` syn) | **0.606** | **0.64% / 2.68%** | **Zero halo (`+0.10%`), `+3.4 dB` cleaner grain** |

*(Native 16.05 MP `4624×3472` `main8` 8-frame GPU stream scaling check prior to kernel optimization: `main` = `1183.2 ms` / `498.6 MB` RSS; unoptimized `dev-exp` Indigo = `1405.8 ms` / `560.7 MB` RSS = `1.32×` of 12MP for `1.338×` pixel count.)*

---

## 2. Per-Stage Timing, Quality Gain & Cost-Effectiveness (ROI) Table

**ROI Metric**:

$$\text{ROI} = \frac{\Delta U}{\max(\Delta t_{\text{GPU}}, 0.1\text{ ms})}$$

where $\Delta U \in [-10, +10]$ measures combined objective quality and Indigo-Pixel1 aesthetic utility (`+1.0` $\approx$ `+1 dB` artifact-free SNR/PSNR; `+5..+10` = eliminates a severe RAW/ISP defect; `< 0` = degrades natural grain, highlight rolloff, halo control, color fidelity, or HAL stability).

| # | Stage / Candidate Module (`main` & `dev-exp`) | GPU Warm (ms) | CPU 8T / 4T (ms) | Cold Extra (ms) | Measured Objective & Visual Gain ($\Delta Q$) | Utility $\Delta U$ | $\text{ROI}$ ($\Delta U/\text{ms}$) | Decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| 1 | **Highlight-Clipping Exposure Guard** (`-1..-2 EV` smoothstep on 2–18% preview clip + low-ISO OIS 66 ms) | **0.00** | 0.00 | 0.0 | Up to **+1.0 EV** extra highlight headroom (`-2.0 EV` total) on bright/backlit scenes with zero bracket ghosting | +8.5 | **$\infty$** | **KEEP / UPGRADE** |
| 2 | **Contrast AF Sweep + 4 Hz `SceneMeter`** (`FocusSweeper`) | **0.00** (preview) | — | — | Cuts `marble` HAL 1.0 D AF scatter (`0.20 -> 0.05 D`), soft shots `15–29% -> 0%` | +10.0 | **$\infty$** | **KEEP** |
| 3 | **`RAW16_LE` `NativeMemory` Copy + Pool** (vs `RAW10` unpack) | — | **1.37**/f (NEON) | 0.0 | Zero-Java-heap `malloc` + 4-block size-evicting pool preserves native `RAW16_LE` buffers required by `DebugCapturePipeline` | +2.5 | **+0.23** | **KEEP (`RAW16_LE` + Pool)** |
| 4 | **`ZSL Ring Buffer` + 58-Req Reopen** (`dev-exp` M4) | 0.00 | — | **580 ms** / 12 s | Polls RAW @ 5 fps -> hits `marble` 65-request leak every **11.6 s**, forcing **500–614 ms preview freeze** + breaks per-shot highlight guard | -6.0 | **-0.010** | **DROP** |
| 5 | **Per-K Frame Early Stop** (`StreamProgress`, `dev-exp` M4) | -131 ms (at 5f) | — | 0.0 | Stopping at `5/8` frames loses **-2.11 dB** raw SNR ($\sigma\ 0.00116 \to 0.00148$); `BurstStream` hides frame compute inside 66 ms readout | -2.1 | **Negative** | **DROP** |
| 6 | **Manual `BRACKET` & `NIGHT` Scene Modes** | +135 to +552 | — | 0.0 | `BRACKET` (`1198 ms`, `+269 MB` RSS) & `NIGHT` pickers violate single unadjustable pipeline; replaced by auto `-2 EV` Highlight Guard + 8f merge | -3.0 | **Negative** | **DROP** |
| 7 | **Reference Frame Selection** (`sharpest_frame`, top of 3) | **3.13** | 2.95 / 3.70 | 0.6 | **+3.2%** handheld gradient sharpness (`hh_main8` selects sharp frame 1/2 over soft frame 0) | +4.5 | **+1.44** | **KEEP** |
| 8 | **Streaming Overlap** (`BurstStream` `add_frame`) | **-350** (hidden) | — | 0.0 | Overlaps 6/8 frames with 528 ms capture; cuts peak RSS **`598 -> 382 MB` (-216 MB)**; batch path kept when `stream == null` for Debug Capture | +9.0 | **$\infty$** | **KEEP** |
| 9 | **4-Level Pyramid Block-Match Align** (integer L2/L1) | **214.7** (`30.7`/pair) | 225.6 / 296.6 | 77.5 | Sub-tile motion alignment across 8 handheld frames (`weight = 0.878`, `ghost = 0.374`) | +10.0 | **+0.047** | **KEEP** |
| 10 | **Subpixel 3×3 Parabolic Align** (`align_subpixel_cost/fit`) | **+53.8** (`+88` @16MP) | +75.7 / +106.9 | +15.0 | Computes `(fx, fy)` **unused by `SpatialMerge`** (`0.00 dB` change on spatial path; only used by dropped `freq`/`kr`) | **0.0** | **0.000** | **DROP (Saves 54–88 ms)** |
| 11 | **`SEA-RAFT` Optical Flow** (`sea_raft.onnx` 35.7 MB) | **+3105.0** (4f CPU) | 3105.0 (4T) | +762.0 | `-0.77 dB` flat SNR vs block-match (`43.30 vs 44.07 dB`), `+69.9 MB` APK (`ORT + model`), 3.1–3.9 s stall | -4.0 | **-0.0013** | **DROP** |
| 12 | **Calibrated Noise Curve** (`cam0` LUT + UW `estimate_noise`) | **0.01** (saves **24 ms**) | 0.01 (vs 18–24) | 0.0 | Saves `24 ms` residual fit on `cam0`, lifts handheld merge weight `.911 -> .934`, `-32%` night visible noise | +3.5 | **$\infty$** | **KEEP / UPGRADE** |
| 13 | **Spatial Robust Merge + Symmetric Variance** (`(mu0+mu)/8 - black`, `dilate=1`) | **165.5** (`0.0` delta) | 143.6 / 198.8 | 10.0 | **+8.68 dB** flat SNR (`37.35 -> 46.03 dB`, raw $\sigma$ `3.10×` lower) w/ 100% organic grain; symmetric variance adds `0.0 ms` & `+0.03 dB` PSNR | +9.5 | **+0.057** | **KEEP / UPGRADE** |
| 14 | **Frequency Wiener Merge** (`merge_freq` / `freq.cl.inc`) | **693.9** (`1188` stream) | 533.5 (4T) | +38.0 | `+528 ms` slower than spatial; wipes `-65%` daylight micro-texture (`tex_hf 1.71 -> 0.60`) and rings edges (`overshoot 8.03%`) | -3.5 | **-0.007** | **DROP** |
| 15 | **Green-Guided Kernel Regression** (`kr` 1×/2×/Auto-2×) | **756.5** (1×) / **548.7** (2×) | 3383.9 (4T) | +25.0 | 1× is `+474 ms` slower than `Spatial+RCD`, breaks `BurstStream`, `5.92%` edge overshoot, worm artifacts at `n=1`; 2× is `+408 ms` slower than Lanczos | -2.5 | **-0.005** | **DROP** |
| 16 | **`PMRID` Bayer CNN Denoiser** (`pmrid.onnx` 4.2 MB) | **+2428.0** (12MP) | 2428.0 (4T) | +229.0 | **Peak RSS `2502.1 MB` (OOM risk)**; obliterates **`-87.6%` real texture** (`tex_hf 2.526 -> 0.313`) -> plastic oil-painting look | **-9.0** | **-0.0037** | **DROP** |
| 17 | **Black-Level, `-1/16` Pedestal, 3×3 Hot-Pixel, 85% LSC & WB** (`normalize_mosaic`) | **22.0** (`main`) | 92.0 / 140.0 | 14.0 | Eliminates `+26.6 DN` shadow pink cast (`shadow_RG = +0.03`), clamps 1691 hot pixels, prevents 5× corner noise blowup | +9.5 | **+0.432** | **KEEP** |
| 18 | **Learned AWB (`FFCC` / `C5` `c5_m7.onnx`)** vs **`HAL AWB`** | **0.00** (HAL) | +0.18 / +6.04 | +142.0 (C5) | `C5` has `10.10°` high-ISO error + 36 MB ORT; `FFCC` has `5.18°` Canon domain bias. `HAL AWB` is `0 ms`, `<1°` locus stability | -2.0 | **Negative** | **DROP `FFCC`/`C5`, KEEP `HAL AWB`** |
| 19 | **Demosaic: `RCD`** (vs `MHC` / `AMaZE`) | **116.0** (`RCD`) | 99.0 / 112.0 | 114.0 | `RCD` beats `MHC` by `+3.09 dB` Kodak & `+5.64 dB` flat SNR (`46.03 vs 40.39 dB`), `-14%` chroma halo; `AMaZE` stalls `660 ms` CPU | +8.5 | **+0.073** | **KEEP `RCD`, DROP `MHC`/`AMaZE`** |
| 20 | **Lateral CA Correction** (`correct_lca`, UW `cam2` only) | **23.0** (12MP) / ~14 (8MP) | 36.0 | 0.0 | Cuts IMX355 ultrawide radial R/B fringing `0.35 -> 0.03 px`; `0.00 ms` no-op on main `cam0` | +4.0 | **+0.174** (UW) | **KEEP** |
| 21 | **Scene-Tuned HAL CCM + Dual-Illuminant Fallback** (`compute_dual_illuminant_ccm`) | **13.0** (pre-fusion) | 9.0 + **0.0014** | 0.0 | Faithful linear sRGB color; `1.4 µs` DNG Sec. 6 fallback guards against singular/identity HAL CCM (`99.00 dB` match) | +8.5 | **+0.654** | **KEEP / UPGRADE** |
| 22 | **Luma-Ratio Adaptive Chroma Denoise** (`(R-L)/(L+4e-3)` 1/4-res) | **42.0** | 40.0 / 57.0 | 2.0 | Suppresses color blotches ($\sigma_C = 0.41$) & highlight chroma bleed (`chroma_halo = 0.606`) while keeping 100% luma grain | +8.0 | **+0.190** | **KEEP** |
| 23 | **2× Center-Crop + Separable Lanczos-3 Upscale** (`crop` + `upscale`) | **-343.1 net** (`88` upscale) | — | 0.0 | Even Bayer crop before align+merge saves `343 ms` net (`413 MB` RSS) and cuts 2× luma noise `4.2% -> 1.0%` vs single/ISP | +7.0 | **$\infty$** | **KEEP** |
| 24 | **1/4-Res Mertens Exposure Fusion + Guided Filter + `clipFade` + Whitepoint** | **63.5** (vs `107` full) | 69.0 / 88.0 | 8.0 | Authentic Pixel 1 HDR+ rolloff (`62.98 dB` vs full-res, saves `43.5 ms`), tuned to `compression = 5.0, targetKey = 0.20` | +9.0 | **+0.142** | **KEEP (`1/4`-res)** |
| 25 | **`HDRnet` Bilateral Grid Tonemap** (`hdrnet.onnx` 1.9 MB) | **+13.5** warm | — | **+124.5** | **Catastrophic `10.45%` highlight clip** (24× worse than Mertens `0.44%`), **`114.2°` hue twist**, `+5.15` red shadow cast | **-8.5** | **-0.630** | **DROP** |
| 26 | **Oklab Gamut Compression + Highlight Shoulder/Bleach + S-Curve** (`pixel_look`) | **119.1** (pre-fusion) | 535 / 860 | 0.0 | Cuts saturated highlight clipping **-49% to -63%** (`s5` `2.86% -> 1.47%`), zero cyan twist on cool highlights, smooth specular bleach | +7.5 | **+0.063** (+0.17 fused) | **KEEP / UPGRADE** |
| 27 | **Sky/Skin Selective `hue_tune`** (`skySat 1.18, skinSat 0.92`) | **+58.6 to +89.0** | +325.0 | 0.0 | Second full-frame Oklab round-trip (`+59..89 ms` GPU) shifting sky `-4°` away from faithful SLR rendering | -2.0 | **-0.034** | **DROP (Saves 59–89 ms)** |
| 28 | **Guided-Filter `local_contrast`** (`0.30..0.60` vs **`0.0`**) | **+32.0** | +65.0 | 0.0 | At `0.60`, amplifies flat wall noise **+53% (`0.543 -> 0.829`, -3.4 dB SNR)** & chroma halo (`0.61 -> 3.65`), creating crunchy HDR | -3.5 | **-0.109** | **DROP (`0.0`, Saves 32 ms)** |
| 29 | **Coring Luma USM `sharpen_luma`** (`0.80..0.90` -> **`0.35, r=0.85`**) | **70.0** (pre-fusion) | 61.0 / 83.0 | 0.0 | Lowering to `0.35 (r=0.85)` cuts step-edge overshoot **`7.33% -> 0.64%` (-91% ringing)** while raising `tex_grain_ratio` `+13.4%` | +6.5 | **+0.093** (+0.33 fused) | **KEEP / UPGRADE** |
| 30 | **3D `.cube` Film LUTs** (`film_warm` / `classic_chrome` / `bw_silver`) | **+21.0** | +109.0 | 0.0 | `+2.07 MB` assets, raises `hi_clip` (`0.46% -> 0.83%`), adds shadow cast (`RG -1.43, BG +2.04`) & `71.8°` hue shift | -4.0 | **-0.190** | **DROP** |
| 31 | **Semantic Segmentation** (`TopFormer-T` `semantic_seg.onnx` 5.6 MB) | **+96.2** warm | — | **+225.0** | `+39.8 MB` APK (`ORT + model`), region-boundary modulation violates uniform physical SLR rendering | -3.0 | **-0.031** | **DROP** |
| 32 | **512px Thumbnail + Strip-Parallel `libjpeg-turbo` q95 4:2:0** | **30.5 + 75.5** (pre-opt) | 4.3 + **64.4** (8T) | 10.0 | Direct native RGB8 -> strip-parallel JPEG (avoids `217 ms` `Bitmap.compress` & 64 MB Java heap) | +9.5 | **+0.126** | **KEEP / UPGRADE** |
| 33 | **Production `Merged DNG` & `Ultra HDR` Gain Map** | **+39.7** (DNG) / **+22.0** (UHDR) | — | 0.0 | Production locks SDR JPEG (`writeDng = false`, `ultraHdr = false`), saving `40–120 ms` per shot; `DebugCapturePipeline` archives raw frames directly | 0.0 | **0.000** | **DROP in Prod (`false`)** |

---

## 3. Locked Single-Pipeline Keep / Drop Architecture (`IndigoHdrPlusPipeline`)

### 3.1 Deleted Modules & Binary Footprint Reduction
All negative- and zero-ROI candidates were removed from the native build, Kotlin pipeline, and UI:
- **Removed ML / LUT / Multi-Mode Bloat** (avoids `85.8 MB` of `dev-exp` ONNX Runtime + `.onnx` + `.cube` assets and keeps stripped `libburstpipe.so` at **`2.4 MB`**):
  - All ONNX models and wrappers (`PMRID`, `SEA-RAFT`, `HDRnet`, `TopFormer-T`, `C5` / `FFCC` AWB), 3D `.cube` LUTs, `ZslRingBuffer`, `StreamProgress` early-stop, `BRACKET` / `NIGHT` / `PRO` UI pickers, `CapabilitiesScreen`, and `feature/p1/*` probe screens.
  - Native dead code deleted: `kr.{h,cpp}`, `kr.cl.inc`, `merge_freq.cpp`, `third_party/rtprocess/amaze.cc`, `demosaic_mhc`, subpixel parabolic alignment (`align_subpixel_cost` / `align_subpixel_fit`), `local_contrast` (`lc_*`), `hue_tune`, `apply_lut3d`, and `stb_image_write`.

### 3.2 Locked Production Pipeline Specification
- **Capture & Exposure (`BurstExposure.kt`, `SceneMeter.kt`, `CameraController.kt`)**:
  - `8` RAW frames per press (`1` frame at AE when flash fires), Camera2 `HAL AWB`, and `FocusSweeper` contrast AF.
  - `-1.0 EV` base burst underexposure + **Highlight-Clipping Exposure Guard**: `SceneMeter` tracks preview `clippedFraction` (`>= 0.90f` on any channel) and `BurstExposure` smoothly pulls back up to `-1.0 EV` extra between `2%` and `18%` clipping (**`-2.0 EV` total** at `evIndex == 0`), prioritizing lowest ISO up to `66 ms` (OIS) / `33 ms` (non-OIS).
- **Front-End (`ReferenceStage`, `BurstStream`, `align`, `merge`)**:
  - Top-3 gradient `sharpest_frame` reference selection.
  - `BurstStream` overlaps frame alignment and spatial merge on Adreno 725 OpenCL during sensor readout (while preserving the non-streamed batch path whenever `openStream == null` so `DebugCapturePipeline` can stream untouched `RAW16_LE` frames `frame_00.raw`..`frame_NN.raw` + `capture_debug.json` from `NativeMemory`).
  - Integer 4-level pyramid block-match alignment -> calibrated `cam0` noise LUT / `cam2` residual noise -> spatial robust merge with **symmetric shot-noise variance** `(mu0 + mu) / 8 - black` and motion dilation `R = 1` (with even Bayer center-crop before alignment for `1 < zoom <= 2`).
- **Back-End (`finish.cpp` / `gpu_burst.cpp`, locked `indigo` preset across `1×`, `2×`, and `0.6×` UW)**:
  - Signed `-1/16` black-level pedestal, 3×3 hot-pixel clamp, `lscLumaStrength = 0.85` (100% chroma LSC), white balance.
  - `RCD` demosaic -> ultrawide radial `correct_lca` (`cam2` only) -> scene-tuned `HAL CCM` (with automatic `compute_dual_illuminant_ccm` DNG fallback when `!ccm_usable`) -> luminance-ratio adaptive `chroma_denoise` -> separable Lanczos-3 upscale (`2×` zoom) -> `16:9` aspect crop (when enabled).
  - `1/4`-res Mertens exposure fusion + guided filter + `clipFade` + folded whitepoint (`targetKey = 0.20`, `compression = 5.0`).
  - Single-pass perceptual **Oklab constant-hue gamut compression** + `0.70` highlight shoulder / `0.90–3.2` specular bleach + S-curve (`contrast = 0.20`, `saturation = 1.08`, `localContrast = 0.0`).
  - Noise-adaptive coring luminance USM (`sharpen = 0.35`, `sharpenRadius = 0.85`, `core = max(0.004, 0.6 * visibleSigma)`) -> native 512px thumbnail + 8-thread strip-parallel `libjpeg-turbo` q95 4:2:0 into `DCIM/PocoPawOpt/PocoPawOpt_*`.

---

## 4. Poco F5 (`SM7475` + `Adreno 725`) Hardware Accelerations Implemented

1. **OpenCL Program Trim, Qualcomm Hints, `.clbin` Pre-Warm & Safe Fallback (`cl_runtime.{h,cpp}`, `gpu_burst.{h,cpp}`)**:
   - Compiling dead `freq`/`kr`/experimental kernels into a single monolithic `cl_program` caused Qualcomm's `E031.38.11.14` compiler to spill registers and regress `normalize_mosaic` from `20.7 ms` to `79.2 ms`. Trimming the program to active Indigo kernels only restored full occupancy.
   - Enabled `cl_qcom_perf_hint` (`CL_PERF_HINT_HIGH_QCOM`) and `cl_qcom_priority_hint` (`CL_PRIORITY_HINT_HIGH_QCOM`), explicit Adreno Wave128 (`16×8` / `128`) work-group sizes, and a `256 MB` retained buffer pool (`trim_excess`) across shots.
   - Cached compiled `.clbin` binaries with eager `clCreateKernelsInProgram` pre-warmed asynchronously in `PocoPawApp` (`12.2 ms` warm `gpu_init`, `0 ms` first-shot kernel compile stall).
   - Preserved automatic per-stage OpenCL-to-CPU fallback, keeping `merged_` resident on GPU until final `rgb8_` readback completes so any mid-`finish` `GpuError` safely downloads `merged_` via `download_merged()` and completes on `finish_cpu`.
2. **Zero-Spill Kernels, Streaming Ping-Pong, FP16 RCD & Hardware `image2d_t` (`kernels/*.cl.inc`)**:
   - Eliminated all runtime-indexed private arrays (`align.cl.inc` scalar `int2`, `merge.cl.inc` taps, `demosaic.cl.inc` borders) and serial global thumbnail loops.
   - Fused `gray_down2` + `__local` `box3_3x3`; dropped unused subpixel parabolic align; kept streaming alignment offsets (`odx_`/`ody_`) on GPU with ping-pong raw upload slots and fused `min3_merge_acc` (`0` per-frame host/device stalls during `BurstStream`).
   - Fused `normalize_mosaic` (`__local` despike + `__constant float4` LSC).
   - Stored bounded `demosaic_rcd` directional intermediates in FP16 (`half4` via `cl_khr_fp16`, cutting DRAM traffic in half) and fused `rcd_finish_rgb` (`float2` R/B-at-G + CCM).
   - Replaced manual bilinear taps with Adreno texture-pipe `image2d_t` (`CL_RGBA` `CL_FLOAT` `CLK_FILTER_LINEAR`) in `chroma_apply_img + tm_prepare` and `tonemap_and_look`.
   - Vectorized `box4`/`box2` guided-filter passes, folded whitepoint scaling onto `1/4`-res luma, mapped tiny reduction buffers (`keySub`, `wpSub`) via LPDDR5 zero-copy `CL_MEM_ALLOC_HOST_PTR`, and removed mid-finish `clFinish()` bubbles.
   - Fused Oklab constant-hue gamut compression (with a 4-comparison deeply-in-gamut fast exit + SFU OETF) into a single `tonemap_and_look` pass, and fused `sharpen_luma` + dithering + `to_rgb8` into one `__local`-tiled 7×7 Gaussian USM kernel (`sharpen_to_rgb8`) with 32-bit coalesced writes.
3. **SM7475 Big+Prime Host Affinity + 8-Core Dynamic `ThreadPool` & `NativeMemory` Pool (`common.h`, `align.cpp`, `jni_burst.cpp`)**:
   - Empirical Poco F5 profiling proved a critical scheduling split: pinning the **OpenCL host dispatch thread** (`ScopedBigCoreAffinity`) to big+prime `CPUs 4..7` (`3× Cortex-A710 + 1× Cortex-X2`) saved **`89.9 ms`** by keeping driver dispatch and `nth_element` off in-order `Cortex-A510` cores, whereas restricting `ThreadPool` from `8 -> 4` threads *slowed* CPU fallback by `+324 ms` and `libjpeg-turbo` by `+25 ms` because `ThreadPool` uses lock-free per-row/per-strip `fetch_add(1)` stealing. Production pins only the calling host thread to `CPUs 4..7` while keeping `g_threads = 8` for `ThreadPool` and strip-parallel JPEG.
   - Added NEON `gray_from_raw` / `box3`, single-JNI `NativeMemory.copyRows`, and a size-evicting 4-block `NativeMemory` (`malloc`) recycle pool (evicting mismatched blocks on lens switch, with zero `ByteBuffer.allocateDirect` heap pressure).

---

## 5. Final Before / After Headless Benchmarks & PSNR Verification (`848893fa`)

### 5.1 End-to-End & Per-Stage Latency (`Poco F5`, Zero Screen Wakes)

| Workload & Mode (`848893fa`) | Pre-Opt Locked Baseline (ms) | Optimized Median (ms) | Optimized Best (ms) | Speedup / Saved | Per-Stage Breakdown (ms) |
|---|---:|---:|---:|---|---|
| **12MP (`4000×3000×8` `--synth`) GPU Batch** | 1062.7 (`dev-exp` Pixel: 1281.3) | **592.1** | **585.7–586.0** | **`1.80×` (`-470.6 ms`)** | `align 201.0`, `merge 121.5`, `finish 223.0`, `thumb 3.7`, `jpeg 26.4` |
| **12MP (`4000×3000×8` `--synth`) GPU Stream** | 1014.1 (`1149.1` cold) | **628.9** | **602.7** | **`1.61×` (`-385.2 ms`)** | `< 260 ms` post-shutter latency after burst readout |
| **16MP Real Burst (`main8`, `4624×3472×8`) GPU Batch** | ~1405.8 (CPU: 1744.1) | **769.8** | **765.5** | **`1.83×` (`-636.0 ms`)** | `align 263.5`, `merge 152.2`, `finish 300.1`, `thumb 3.3`, `jpeg 34.9` |
| **16MP Real Burst (`main8`, `4624×3472×8`) GPU Stream** | 1373.8 (`1465.0` cold) | **795.1** | **771.5–793.2** | **`1.73×` (`-578.7 ms`)** | `finish 299.7` (with mid-finish fallback guard active) |
| **Ultrawide + Radial LCA (`--gpu`)** | — | **648.1** | — | — | Full 8-frame UW path with radial R/B `correct_lca` |
| **`2×` Crop Zoom + `16:9` Aspect (`--gpu`)** | 719.6 (4:3 pre-opt) | **342.0** | — | **`2.10×` (`-377.6 ms`)** | Bayer center-crop + align/merge + Lanczos-3 + 16:9 crop |
| **Single-Frame Flash (`-n 1 --gpu`)** | — | **314.8** | — | — | 1-frame bypass + full Indigo `finish` + strip JPEG |
| **Cached OpenCL Init (`gpu_init` `.clbin`)** | 749.9–1080.8 (cold) | **12.2** | — | **`0 ms` on 1st shot** | Pre-warmed asynchronously in `PocoPawApp` |

### 5.2 GPU vs CPU Numerical Equivalence & Aesthetic Verification

Every GPU path stays **> 12 dB above** the project's `50.0 dB` GPU-vs-CPU PSNR requirement, with zero 8-bit outlier pixels (`over8 = 0.0000%`):

| Verification Path (`848893fa`) | GPU vs CPU PSNR (dB) | Max 8-Bit Diff (`/ 255`) | Pixels `> 8 / 255` (`over8`) | Quality & Style Check |
|---|---:|---:|---:|---|
| **12MP (`4000×3000×8` `--synth`)** | **62.87 dB** | `4` | `0.0000%` | Optical step overshoot **`0.64%`** (0-px synthetic step `28.72% -> 9.94%`); shadow `R-G = -0.01`, `B-G = +0.04`, `grain_std = 1.42` |
| **16MP Real Burst (`main8`, `4624×3472×8`)** | **62.44 dB** | `4` | `0.0000%` | Clean highlight rolloff, natural fine grain, neutral shadows |
| **Ultrawide + Radial LCA** | **62.95 dB** | `3` | `0.0000%` | Radial R/B fringing `0.35 -> 0.03 px` |
| **`2×` Zoom + `16:9` Aspect** | **62.68 dB** | `2` | `0.0000%` | Even Bayer crop + Lanczos-3 upscale + 16:9 crop |
| **Hybrid Stage Fallback (`--gpu-off finish`)** | **87.56 dB** | `1` | `0.0000%` | Verified GPU align/merge -> `download_merged()` -> `finish_cpu` |
