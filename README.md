# PocoPaw

PocoPaw is a RAW camera for the Poco F5 / Redmi Note 12 Turbo (marble), in the Yoin design language: Material 3 Expressive, wallpaper-seeded colour, Google Sans Flex, continuous corners, spring motion, Yoin Symbols. Its screen layout follows [LineageOS Aperture](https://github.com/LineageOS/android_packages_apps_Aperture).

Today it captures a single RAW frame per press and saves it as DNG. The capture hands a `RawBurst` to a pluggable `ImagePipeline`, so a multi-frame pipeline (align, merge, demosaic, tone map; see `../PIPELINE_REPORT.md`) can be added as pipeline stages without touching the camera layer or the UI.

PocoPaw 以 GPL-3.0 发布，见 [LICENSE](LICENSE)。

## Build

```sh
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:assembleDebug        # debug
./gradlew :app:assembleRelease      # release, signed with the debug key
```

Requires the Android SDK (compileSdk 36). minSdk is 31. Icons come from [Yoin Symbols](https://github.com/p2o51/yoin-symbols), built from source from `~/code/yoin-symbols` (or `-PyoinSymbolsDir=<path>`); the build stops without that checkout.

## Layout

- `app/src/main/java/com/izum1k/pocopaw/camera`: camera catalogue, the Camera2 controller, the RAW request budget
- `pipeline`: `RawFrame`, `RawBurst`, `RawMetadata`, `PipelineStage`, `ImagePipeline`, the DNG pass-through pipeline, the quick-look stage, the MediaStore sink
- `feature/camera`: the viewfinder screen; `feature/settings`: settings and the device capability page
- `data`: DataStore settings
- `ui`: theme, shared components, navigation
- `docs/design.md`: design rules; `AGENTS.md`: conventions for contributors and coding agents

## Device notes

The camera layer is built around the limits measured in `../REPORT.md`: at most 65 RAW requests per opened `CameraDevice`, logical cameras 4, 5 and 6 hang or crash the HAL, no `noiseProfile`, and auto exposure holds the shutter at 1/100 s under 50 Hz anti-banding.

## Credits

- [Aperture](https://github.com/LineageOS/android_packages_apps_Aperture) by The LineageOS Project (Apache-2.0): the screen layout and control set. No code is copied.
- [Yoin](https://github.com/p2o51/Yoin) by Chen Wuyi (Apache-2.0): the theme (`ui/theme`) and the design language. See [NOTICE](NOTICE).
- [ScalePaw](https://github.com/xxxizum1kxxx/ScalePaw) and [PiliPaw](https://github.com/p2o51/PiliPaw) (GPL-3.0): theme port, system bars, wallpaper seeds, shared components.
- [material-color-utilities](https://github.com/material-foundation/material-color-utilities) by Google (Apache-2.0): the colour code under `app/src/main/java/{contrast,dislike,dynamiccolor,hct,palettes,scheme,temperature,utils}`.
- [Google Sans Flex](https://github.com/googlefonts/googlesans-flex) (SIL OFL 1.1, see `licenses/GoogleSansFlex-OFL.txt`).
- [Yoin Symbols](https://github.com/p2o51/yoin-symbols) (MIT): every icon.
