# VR Menu Rendering Analysis (Cinema-Screen Stereoscopic UI)

## Current menu render path when VR is enabled

1. `DoUIFrame()` and `DoUIFrameWithoutInput()` detect `vr_menu_mode` as `VR_IsEnabled() && GetFunctionMode() == MENU_MODE`.
2. In VR menu mode, the UI is rendered into a VR menu framebuffer via:
   - `VR_BeginMenuFramebufferRender()` before `ui_DoFrame()`
   - `VR_EndMenuFramebufferRender()` after `ui_DoFrame()`
3. The menu framebuffer texture is then submitted through `VR_RenderMenuFrame()`.

This means menus are already **not drawn directly into left/right eye game render buffers**; they are rendered into an offscreen texture first.

## Current curved-screen implementation details

`VR_RenderMenuFrame()` currently:

- Ensures a submit texture for each eye (`Vr_submit_left`, `Vr_submit_right`)
- Attempts to render a curved textured surface into each eye submit texture via `VR_RenderCurvedMenuToSurface(...)`
- Uses per-eye offsets (`±0.5 * Vr_eye_separation`)
- Submits the resulting OpenGL textures to OpenVR

`VR_RenderCurvedMenuToSurface(...)` draws a curved quad strip with:

- Camera translated by `-eye_offset` and pushed forward by `-menu_distance`
- Curvature from sin/cos arc sampling (`segments = 64`)
- Geometry in front of camera using negative Z (`z` is <= 0 after transform), which matches OpenGL camera-forward conventions in this engine path.

## What already matches your target behavior

- ✅ Menus are rendered to a texture/FBO first, not directly to headset eye buffers
- ✅ Menu texture is mapped to a curved polygon strip (“cinema” surface)
- ✅ Separate left/right eye renders are generated and sent to headset
- ✅ IPD is queried from OpenVR eye-to-head transforms and used in per-eye offsets

## Gaps vs. your requested behavior

1. **Head pose is not currently applied during menu surface rendering**
   - `VR_SubmitOpenVrFrame(...)` calls `WaitGetPoses(...)`, but that pose data is not used to build menu view transforms.
   - Result: screen is eye-offset stereo, but not fully head-pose-coupled for orientation/position.

2. **Potential residual desktop-window presentation path remains active**
   - UI loop still calls `rend_Flip()` each frame. The VR submit path is correct, but desktop mirror behavior may still show conventional output depending on renderer/window setup.

3. **No explicit “mono-at-depth comfort” tuning for menu layer**
   - A curved stereo surface can still produce excessive disparity if distance/curvature/FOV are not tuned to comfortable values.

## Recommended implementation changes

1. **Use OpenVR tracked HMD pose for menu eye views**
   - Fetch poses once per menu frame (`WaitGetPoses` or `GetDeviceToAbsoluteTrackingPose`) before curved rendering.
   - Build per-eye camera transforms as:
     - `View_eye = inverse(HmdPose * EyeToHead(eye))`
   - Load this as model-view matrix before drawing curved geometry.

2. **Keep menu as a world-locked cinema surface in front of HMD**
   - Define anchor in seated/standing space (e.g., 1.2–1.8m forward).
   - Compose model transform for the curved surface:
     - position in front of viewer
     - optional yaw-only follow for comfort

3. **Continue using separate per-eye submit textures**
   - Current submit surface architecture is appropriate.
   - Ensure no direct menu draw into world/game eye render path.

4. **Add comfort controls (configurable CVars/flags)**
   - distance
   - radius / arc width
   - height
   - yaw-follow mode

5. **Optional: submit as compositor overlay (future enhancement)**
   - If you want absolutely no stereo doubling artifacts and cleaner composition, OpenVR overlays can be considered.
   - Current scene-submit path is still valid and simpler to integrate.

## Notes on negative Z in this engine

The current menu mesh path places geometry at negative Z in eye space after model transform, which is correct for a traditional OpenGL camera-facing-forward setup used here. No inversion is needed as long as the generated projection/model-view pair is consistent.
