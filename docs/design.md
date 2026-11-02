# PocoPaw design

PocoPaw follows the Yoin design language (see Yoin's `docs/design.md`) and the conventions of PiliPaw and ScalePaw. Its screen layout follows LineageOS Aperture. This file records what applies to a camera.

## Principles

1. **M3 Expressive first.** `ToggleButton` and connected button groups for quick settings, `LoadingIndicator` while a capture is processed, `MaterialShapes` for the shutter and the focus ring, `LargeFlexibleTopAppBar` on pushed pages.
2. **Motion is springs.** `YoinMotion.defaultSpatialSpec()` for movement, size and rotation, `defaultEffectsSpec()` for colour and alpha. Controls rotate with the device on a spring; the activity itself stays portrait.
3. **Colour comes from M3 tokens.** The viewfinder screen is always dark (the wallpaper still seeds the scheme); controls sit on translucent `surfaceContainer` blocks. Black and white appear only as scrims over the picture and as the shutter flash.
4. **Colour blocks, not strokes.** No dividers or outlined cards. Settings group rows into `surfaceContainer` blocks with `YoinContainerShapes.Card`.
5. **Typography** is always `MaterialTheme.typography.*` in Google Sans Flex. Exposure values (ISO, shutter, EV, zoom) use tabular figures.
6. **Copy is labels.** Short nouns, no narration.
7. **Haptics** accompany every toggle, mode change, lens change, focus tap and shutter press.

## Camera screen

- **Top bar**: flash, timer, aspect ratio and grid as text toggles in one translucent block; settings on the right.
- **Viewfinder**: 4:3 (16:9 crops the preview only, the RAW frame is always full sensor). Thirds grid. Tap to focus and meter: a cookie-shaped ring that morphs to a circle when focus locks; drag vertically beside it to set exposure compensation. Timer countdown and shutter flash draw over the picture.
- **Lens row**: one pill per back camera (0.6x, 1x); pinch zooms within a lens. Tapping the selected 1x pill springs the zoom between 1x and 2x.
- **Mode bar**: Photo (auto exposure) and Pro (manual ISO, shutter, EV within the camera's ranges).
- **Shutter row**: last capture thumbnail (opens the JPEG, or the DNG for DNG-only output, in the system viewer), the shutter (morphs on press, shows a loading indicator while the pipeline runs), front/back switch.
- **Developer chip** (setting): the RAW size, the remaining RAW requests before the device is reopened, and the frame count and time of the last pipeline run.

## Capture

- Preview is YUV only. A press sends a burst of capture requests with the RAW surface; each frame is copied out, the image released, and the burst handed to the active `ImagePipeline`.
- Output (setting): DNG + JPEG (default), JPEG, or DNG. JPEG output runs the burst pipeline chosen under Processing (HDR+ merge, or kernel regression merge) on 4, 8 or 16 frames; DNG + JPEG also writes the reference frame as DNG before the merge. DNG alone keeps the single-frame pass-through. Files go to `DCIM/PocoPaw`.
- In Photo mode a burst is shot manually: the preview AE's exposure × ISO, one stop under, at the lowest ISO with the shutter capped at 66 ms (OIS) or 33 ms. Pro mode keeps the user's values. With the flash on, the burst is a single frame at auto exposure.
- When the remaining RAW requests cannot cover a capture, the device is closed and reopened first.
- RAW always keeps the full sensor. Pinch zoom and the 16:9 ratio shape the viewfinder only; the zoom ratio travels with the burst (`RawMetadata.zoomRatio`) for a later stage to crop. The 2x step (`RawMetadata.cropZoom`) crops the JPEG: the burst is cropped to the centre half before alignment and the merged image is Lanczos-upscaled back to sensor size before tone mapping. The DNG stays full sensor.
