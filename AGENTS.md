# AGENTS.md

## Scope

This file applies to the entire repository.

## Project

PocoPaw is a Compose RAW camera for the Poco F5 / Redmi Note 12 Turbo (marble), in the Yoin design language shared by PiliPaw and ScalePaw. Its layout follows LineageOS Aperture. `docs/design.md` is the source of truth for UI decisions; `../REPORT.md` records the camera HAL limits the camera layer is built around.

## Build & Run

- `export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`
- `./gradlew :app:assembleDebug` builds the debug APK.

## Tech Stack

- Kotlin, Jetpack Compose, Material 3 Expressive, Navigation 3 (single activity, one `NavDisplay`)
- Camera2 directly (no CameraX): the app has to count RAW requests per opened device
- DataStore for settings; manual dependency injection through `AppContainer` (`context.container`)

## Architecture

- `camera/` owns Camera2. `CameraCatalog` lists the usable physical cameras, `CameraController` is the only owner of a `CameraDevice`, `RawRequestBudget` counts RAW requests against the HAL's 65-per-open limit.
- Preview requests output YUV only. Only capture requests add the RAW surface.
- Never open logical multi-cameras or the IDs in `CameraCatalog.DENYLIST`: on marble they hang or crash the camera provider.
- `ERROR_CAMERA_DEVICE` is recovered by closing and reopening the device, never by quitting.
- `pipeline/` is pure processing: a `RawBurst` goes in, an `ImagePipeline` runs its `PipelineStage`s and writes through an `OutputSink`. New processing (align, merge, demosaic, tone map) is a new stage registered in `PipelineRegistry`; the camera layer and UI do not change.
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
