# PocoPaw

PocoPaw is a RAW camera for the Poco F5 / Redmi Note 12 Turbo (marble), in the Yoin design language: Material 3 Expressive, wallpaper-seeded colour, Google Sans Flex, continuous corners, spring motion, Yoin Symbols. Its screen layout follows [LineageOS Aperture](https://github.com/LineageOS/android_packages_apps_Aperture).

A shutter press captures a burst of RAW frames (4, 8 or 16) at an under-exposed, low-ISO manual exposure and runs a CPU-only HDR+ style pipeline on it: tile alignment, robust merge (spatial-domain tiles with a dilated motion mask, frequency-domain Wiener tiles for night, or kernel regression at 1x with lateral chromatic aberration correction on the ultrawide), lens shading and white balance, RCD demosaic, colour, noise-adaptive chroma denoise, synthetic-exposure tone mapping, a standard or Pixel-style look, sharpening, JPEG or Ultra HDR JPEG (gain map). Scene modes: auto (one exposure), HDR bracket (two short frames per long one, merged on the short frame's scale) and night (shutter from gyroscope motion metering). The reference frame can also be saved as DNG, and DNG-only output keeps the single-frame pass-through. On the Poco F5 a 16 MP × 8 burst takes about 2.8 s end to end (Pixel look and Ultra HDR included; about 6.4 s for 16 frames with kernel regression). The research behind it, with what is implemented marked in place, is in `docs/research/`.

PocoPaw 以 GPL-3.0 发布，见 [LICENSE](LICENSE)。

## Build

```sh
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:assembleDebug        # debug
./gradlew :app:assembleRelease      # release, signed with the debug key
```

Requires the Android SDK (compileSdk 36), NDK 29.0.14206865 and CMake 3.22.1. minSdk is 31; the APK is arm64-v8a only. Icons come from [Yoin Symbols](https://github.com/p2o51/yoin-symbols), built from source from `~/code/yoin-symbols` (or `-PyoinSymbolsDir=<path>`); the build stops without that checkout.

## Layout

- `app/src/main/java/com/izum1k/pocopaw/camera`: camera catalogue, the Camera2 controller, the RAW request budget
- `pipeline`: `RawFrame`, `RawBurst`, `RawMetadata`, `PipelineStage`, `ImagePipeline`, the DNG pass-through and HDR+ pipelines, their stages, `BurstExposure`, the JNI wrapper `NativeBurst`, the MediaStore sink
- `app/src/main/cpp/burstpipe`: the native burst pipeline (align, spatial / frequency / kernel-regression merge, bracket scaling, demosaic, finishing, Ultra HDR); `third_party/` holds librtprocess (RCD, AMaZE), libultrahdr (prebuilt arm64) and stb_image_write
- `feature/camera`: the viewfinder screen; `feature/settings`: settings and the device capability page
- `data`: DataStore settings
- `ui`: theme, shared components, navigation
- `docs/design.md`: design rules; `docs/research/`: the camera HAL report, the pipeline report and the algorithm studies; `AGENTS.md`: conventions for contributors and coding agents

## Device notes

The camera layer is built around the limits measured in `docs/research/REPORT.md`: at most 65 RAW requests per opened `CameraDevice`, logical cameras 4, 5 and 6 hang or crash the HAL, no `noiseProfile`, and auto exposure holds the shutter at 1/100 s under 50 Hz anti-banding.

## Credits

- [Aperture](https://github.com/LineageOS/android_packages_apps_Aperture) by The LineageOS Project (Apache-2.0): the screen layout and control set. No code is copied.
- [Yoin](https://github.com/p2o51/Yoin) by Chen Wuyi (Apache-2.0): the theme (`ui/theme`) and the design language. See [NOTICE](NOTICE).
- [ScalePaw](https://github.com/xxxizum1kxxx/ScalePaw) and [PiliPaw](https://github.com/p2o51/PiliPaw) (GPL-3.0): theme port, system bars, wallpaper seeds, shared components.
- [material-color-utilities](https://github.com/material-foundation/material-color-utilities) by Google (Apache-2.0): the colour code under `app/src/main/java/{contrast,dislike,dynamiccolor,hct,palettes,scheme,temperature,utils}`.
- [Google Sans Flex](https://github.com/googlefonts/googlesans-flex) (SIL OFL 1.1, see `licenses/GoogleSansFlex-OFL.txt`).
- [Yoin Symbols](https://github.com/p2o51/yoin-symbols) (MIT): every icon.
- [Handheld Multi-Frame Super-Resolution](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution) by Lafenêtre, Facciolo and Eboli (IPOL 2023, MIT): the kernel-regression merge in `cpp/burstpipe/kr.cpp` is a C++ port of its algorithm.
- HDR+ (Hasinoff et al., SIGGRAPH Asia 2016, including the frequency-domain merge of its Section 5), HDR+ with bracketing, Night Sight motion metering, Malvar-He-Cutler demosaicing and Mertens exposure fusion: the published algorithms behind `align.cpp`, `merge.cpp`, `merge_freq.cpp`, `finish.cpp` and `BurstExposure`, implemented from the papers. [hdr-plus-swift](https://github.com/martin-marek/hdr-plus-swift) (GPL-3.0) was a reference for the frequency merge; no code is copied.
- [librtprocess](https://github.com/CarVac/librtprocess) (GPL-3.0): RCD and AMaZE demosaicing, vendored in `cpp/burstpipe/third_party/rtprocess`.
- [libultrahdr](https://github.com/google/libultrahdr) by Google (Apache-2.0) with [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo): the Ultra HDR container. See [NOTICE](NOTICE).
