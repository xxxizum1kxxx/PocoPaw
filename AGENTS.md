# AGENTS.md

## Scope

This file applies to the entire repository.

## Project

PocoPaw is a Compose RAW camera for the Poco F5 / Redmi Note 12 Turbo (marble), in the Yoin design language shared by PiliPaw and ScalePaw. Its layout follows LineageOS Aperture. `docs/design.md` is the source of truth for UI decisions; `docs/research/REPORT.md` records the camera HAL limits the camera layer is built around, and `docs/research/PIPELINE_REPORT.md` / `ALGORITHM_UPGRADE.md` the processing pipeline (each marks what PocoPaw implements).

## Build & Run

- `export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`
- `./gradlew :app:assembleDebug` builds the debug APK. The native pipeline needs NDK 29.0.14206865 and CMake 3.22.1 from the SDK (`sdkmanager "cmake;3.22.1"`).
- `adb logcat -G 16M && adb logcat -s PocoPawPipeline` shows per-stage timings; the default 256 KB buffer is flushed by the camera HAL within seconds.

## Tech Stack

- Kotlin, Jetpack Compose, Material 3 Expressive, Navigation 3 (single activity, one `NavDisplay`)
- Camera2 directly (no CameraX): the app has to count RAW requests per opened device
- DataStore for settings; manual dependency injection through `AppContainer` (`context.container`)

## Architecture

- `camera/` owns Camera2. `CameraCatalog` lists the usable physical cameras, `CameraController` is the only owner of a `CameraDevice`, `RawRequestBudget` counts RAW requests against the HAL's 65-per-open limit.
- Preview requests output YUV only. Only capture requests add the RAW surface.
- Never open logical multi-cameras or the IDs in `CameraCatalog.DENYLIST`: on marble they hang or crash the camera provider.
- `ERROR_CAMERA_DEVICE` is recovered by closing and reopening the device, never by quitting.
- `pipeline/` is pure processing: a `RawBurst` goes in, an `ImagePipeline` runs its `PipelineStage`s and writes through an `OutputSink`. New processing is a new stage registered in `PipelineRegistry`; the camera layer and UI do not change.
- The multi-frame pipeline (`hdrplus`, `hdrplus_kr`) is C++ in `app/src/main/cpp/burstpipe/`, ported unchanged from the research prototype in `../pipeline/src/`, and reached through `NativeBurst` (JNI). It is CPU only and arm64 only. Keep the ported files close to the prototype so results stay comparable; `../pipeline/tools/session_check.cpp` runs the app's native code on a probe burst on the Mac or the phone. The algorithm files (`align`, `merge`, `merge_freq`, `kr`, `demosaic`, `finish`, `ultrahdr`) are byte-identical copies of the prototype's; only `common.h` (frames are read-only `Raw` views, `make_writable()` copies before bracket scaling), `pool.cpp`, `burst.*` and the JNI differ. Change them in `../pipeline/src/` and copy over. `third_party/libultrahdr/lib/android` is rebuilt with `../pipeline/tools/build_ultrahdr.sh`.
- Scene modes live in `BurstExposure` (exposure plans) and `HdrPlusPipeline` (merge / look per scene); per-lens constants (lateral CA) in `LensCalibration`. Bracketed bursts use `ExposurePlan.PerFrame`; the native session picks the shortest exposure as reference by itself.
- RAW frame copies live in `NativeMemory` (malloc), not `ByteBuffer.allocateDirect`, which ART counts against the 256 MB Java heap. Whoever owns a `RawBurst` closes it.
- `feature/*` holds screens and their ViewModels.

## UI rules

- Colour blocks, not dividers. Use `Section`, `SettingRow`, `SwitchRow`, `ChoiceRow`, `PawPage` instead of raw Material lists.
- Icons come from `YoinSymbols` only; never add material-icons, never draw icons in the app. Camera controls without a symbol yet use text labels.
- Every interaction animates with spring specs from `YoinMotion` and gives haptic feedback.
- All user-facing text goes in `res/values/strings.xml` with a Simplified Chinese translation in `res/values-zh-rCN/strings.xml`.

## Code style

- Kotlin naming conventions; feature-based packages; `StateFlow` over `LiveData`.
- New files carry no licence header. Ported files keep theirs.
- Comments only for constraints the code cannot show.

## Git

- One fix or feature per commit, committed and pushed to `origin/main` (GitHub) as soon as it builds and is verified, so GitHub always tracks the latest state.
- Author email: `244161265+xxxizum1kxxx@users.noreply.github.com`.
