/*
* Descent 3
* Copyright (C) 2024 Parallax Software
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "vr.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "3d.h"
#include "args.h"
#include "bitmap.h"
#include "descent.h"
#include "game.h"
#include "gamesequence.h"
#include "log.h"
#include "NewBitmap.h"
#include "../renderer/dyna_gl.h"
#include "openvr.h"
#include "renderer.h"
#include "vecmat.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

namespace {
bool Vr_enabled = false;
bool Vr_openvr_ready = false;
VrRenderMode Vr_render_mode = VrRenderMode::Cinema;
vr::IVRSystem *Vr_system = nullptr;
float Vr_eye_separation = 0.064f;
uint32_t Vr_submit_width = 0;
uint32_t Vr_submit_height = 0;
struct VrSubmitSurface {
  GLuint texture = 0;
  std::vector<uint32_t> buffer;
};
VrSubmitSurface Vr_submit_cinema;
VrSubmitSurface Vr_submit_left;
VrSubmitSurface Vr_submit_right;
int Vr_menu_bitmap = -1;
int Vr_menu_width = 0;
int Vr_menu_height = 0;
int Vr_menu_texture_size = 0;

void VR_DeleteSubmitSurface(VrSubmitSurface &surface);

constexpr float kPi = 3.14159265358979323846f;

struct VrGlFns {
  bool loaded = false;
  using GenTexturesFn = decltype(&glGenTextures);
  using DeleteTexturesFn = decltype(&glDeleteTextures);
  using BindTextureFn = decltype(&glBindTexture);
  using TexParameteriFn = decltype(&glTexParameteri);
  using TexImage2DFn = decltype(&glTexImage2D);
  using TexSubImage2DFn = decltype(&glTexSubImage2D);

  GenTexturesFn gen_textures = nullptr;
  DeleteTexturesFn delete_textures = nullptr;
  BindTextureFn bind_texture = nullptr;
  TexParameteriFn tex_parameteri = nullptr;
  TexImage2DFn tex_image_2d = nullptr;
  TexSubImage2DFn tex_sub_image_2d = nullptr;
};

VrGlFns &VR_GetGlFns() {
  static VrGlFns fns;
  if (fns.loaded) {
    return fns;
  }

  fns.gen_textures = reinterpret_cast<VrGlFns::GenTexturesFn>(SDL_GL_GetProcAddress("glGenTextures"));
  fns.delete_textures = reinterpret_cast<VrGlFns::DeleteTexturesFn>(SDL_GL_GetProcAddress("glDeleteTextures"));
  fns.bind_texture = reinterpret_cast<VrGlFns::BindTextureFn>(SDL_GL_GetProcAddress("glBindTexture"));
  fns.tex_parameteri = reinterpret_cast<VrGlFns::TexParameteriFn>(SDL_GL_GetProcAddress("glTexParameteri"));
  fns.tex_image_2d = reinterpret_cast<VrGlFns::TexImage2DFn>(SDL_GL_GetProcAddress("glTexImage2D"));
  fns.tex_sub_image_2d = reinterpret_cast<VrGlFns::TexSubImage2DFn>(SDL_GL_GetProcAddress("glTexSubImage2D"));
  fns.loaded = fns.gen_textures && fns.delete_textures && fns.bind_texture && fns.tex_parameteri && fns.tex_image_2d &&
               fns.tex_sub_image_2d;
  return fns;
}

int VR_NextPowerOfTwo(int value) {
  int size = 1;
  while (size < value) {
    size <<= 1;
  }
  return size;
}

void VR_EnsureMenuBitmap() {
  const int desired_width = Max_window_w;
  const int desired_height = Max_window_h;
  const int desired_texture_size = VR_NextPowerOfTwo(std::max(desired_width, desired_height));

  if (Vr_menu_width == desired_width && Vr_menu_height == desired_height && Vr_menu_texture_size == desired_texture_size &&
      Vr_menu_bitmap >= 0) {
    return;
  }

  if (Vr_menu_bitmap >= 0) {
    bm_FreeBitmap(Vr_menu_bitmap);
    Vr_menu_bitmap = -1;
  }

  Vr_menu_width = desired_width;
  Vr_menu_height = desired_height;
  Vr_menu_texture_size = desired_texture_size;
  Vr_menu_bitmap = bm_AllocBitmap(Vr_menu_texture_size, Vr_menu_texture_size, 0);
  if (Vr_menu_bitmap < 0) {
    LOG_WARNING << "VR: Unable to allocate menu bitmap for cinema screen.";
  }
}

void VR_UpdateMenuTexture(const NewBitmap &screenshot) {
  uint32_t w, h;
  screenshot.getSize(w, h);
  auto *src_data = reinterpret_cast<uint32_t *>(screenshot.getData());
  if (!src_data) {
    return;
  }
  const bool stereo_side_by_side = (h > 0) && (w >= h * 2);
  const bool stereo_top_bottom = (w > 0) && (h >= w * 2);
  const uint32_t src_w = stereo_side_by_side ? (w / 2) : w;
  const uint32_t src_h = stereo_top_bottom ? (h / 2) : h;
  uint16_t *dest_data = bm_data(Vr_menu_bitmap, 0);
  if (!dest_data) {
    return;
  }

  const int dest_size = Vr_menu_texture_size;
  for (int y = 0; y < dest_size; ++y) {
    for (int x = 0; x < dest_size; ++x) {
      uint16_t pixel = GR_RGB16(0, 0, 0);
      if (x < static_cast<int>(src_w) && y < static_cast<int>(src_h)) {
        const uint32_t spix = src_data[y * w + x];
        const int r = spix & 0xff;
        const int g = (spix >> 8) & 0xff;
        const int b = (spix >> 16) & 0xff;
        pixel = GR_RGB16(r, g, b);
      }
      dest_data[((dest_size - 1) - y) * dest_size + x] = pixel;
    }
  }
}

void VR_EnsureSubmitTexture() {
  if (!Vr_openvr_ready || !Vr_system || Renderer_type != RENDERER_OPENGL) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.gen_textures || !gl.bind_texture || !gl.tex_parameteri || !gl.tex_image_2d || !gl.delete_textures) {
    static bool missing_gl_warned = false;
    if (!missing_gl_warned) {
      LOG_DEBUG << "OpenVR: Missing GL entry points for texture submission.";
      missing_gl_warned = true;
    }
    return;
  }

  uint32_t target_w = 0;
  uint32_t target_h = 0;
  Vr_system->GetRecommendedRenderTargetSize(&target_w, &target_h);
  if (target_w == 0 || target_h == 0) {
    return;
  }

  if (Vr_submit_width == target_w && Vr_submit_height == target_h) {
    return;
  }

  if (Vr_submit_width != 0 || Vr_submit_height != 0) {
    VR_DeleteSubmitSurface(Vr_submit_cinema);
    VR_DeleteSubmitSurface(Vr_submit_left);
    VR_DeleteSubmitSurface(Vr_submit_right);
  }

  Vr_submit_width = target_w;
  Vr_submit_height = target_h;
}

void VR_DeleteSubmitSurface(VrSubmitSurface &surface) {
  auto &gl = VR_GetGlFns();
  if (surface.texture != 0) {
    if (gl.delete_textures) {
      gl.delete_textures(1, &surface.texture);
    }
    surface.texture = 0;
  }
  surface.buffer.clear();
}

bool VR_EnsureSubmitSurface(VrSubmitSurface &surface) {
  if (Vr_submit_width == 0 || Vr_submit_height == 0) {
    return false;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.gen_textures || !gl.bind_texture || !gl.tex_parameteri || !gl.tex_image_2d || !gl.delete_textures) {
    return false;
  }

  if (surface.texture == 0) {
    gl.gen_textures(1, &surface.texture);
    gl.bind_texture(GL_TEXTURE_2D, surface.texture);
    gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl.tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA8, Vr_submit_width, Vr_submit_height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                    nullptr);
  }

  const size_t desired_size = static_cast<size_t>(Vr_submit_width) * static_cast<size_t>(Vr_submit_height);
  if (surface.buffer.size() != desired_size) {
    surface.buffer.assign(desired_size, 0);
  }

  return surface.texture != 0;
}

void VR_UpdateSubmitSurface(const NewBitmap &screenshot, VrSubmitSurface &surface, bool allow_stereo_crop,
                            int eye_index) {
  if (!Vr_openvr_ready || surface.texture == 0 || Vr_submit_width == 0 || Vr_submit_height == 0) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.bind_texture || !gl.tex_sub_image_2d) {
    return;
  }

  uint32_t src_w = 0;
  uint32_t src_h = 0;
  screenshot.getSize(src_w, src_h);
  auto *src_data = reinterpret_cast<uint32_t *>(screenshot.getData());
  if (!src_data) {
    return;
  }
  const bool stereo_side_by_side = allow_stereo_crop && (src_h > 0) && (src_w >= src_h * 2);
  const bool stereo_top_bottom = allow_stereo_crop && (src_w > 0) && (src_h >= src_w * 2);
  const uint32_t sample_width = stereo_side_by_side ? (src_w / 2) : src_w;
  const uint32_t sample_height = stereo_top_bottom ? (src_h / 2) : src_h;
  const uint32_t offset_x = (stereo_side_by_side && eye_index == 1) ? sample_width : 0;
  const uint32_t offset_y = (stereo_top_bottom && eye_index == 1) ? sample_height : 0;

  for (uint32_t y = 0; y < Vr_submit_height; ++y) {
    const uint32_t src_y = (y * sample_height) / Vr_submit_height + offset_y;
    for (uint32_t x = 0; x < Vr_submit_width; ++x) {
      const uint32_t src_x = (x * sample_width) / Vr_submit_width + offset_x;
      const uint32_t spix = src_data[src_y * src_w + src_x];
      surface.buffer[y * Vr_submit_width + x] = spix;
    }
  }

  gl.bind_texture(GL_TEXTURE_2D, surface.texture);
  gl.tex_sub_image_2d(GL_TEXTURE_2D, 0, 0, 0, Vr_submit_width, Vr_submit_height, GL_RGBA, GL_UNSIGNED_BYTE,
                      surface.buffer.data());
}

void VR_UpdateOpenVRPoses() {
  if (!Vr_openvr_ready || !vr::VRCompositor()) {
    return;
  }

  vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
  vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
}

void VR_DrawCinemaScreen(int /*texture_handle*/, float /*u_max*/, float /*v_max*/) {
  constexpr int kSegments = 32;
  constexpr float kArcDegrees = 100.0f;
  constexpr float kRadius = 6.0f;
  constexpr float kHeight = 3.0f;

  const float arc_radians = kArcDegrees * (kPi / 180.0f);
  const float start_angle = -arc_radians * 0.5f;
  const float delta = arc_radians / static_cast<float>(kSegments);
  const float half_height = kHeight * 0.5f;

  rend_SetZBufferState(0);
  rend_SetTextureType(TT_FLAT);
  rend_SetLighting(LS_NONE);
  rend_SetColorModel(CM_RGB);
  rend_SetFlatColor(GR_BLACK);
  rend_SetAlphaType(AT_ALWAYS);

  for (int i = 0; i < kSegments; ++i) {
    const float a0 = start_angle + (delta * i);
    const float a1 = a0 + delta;

    vector p0{std::sin(a0) * kRadius, -half_height, std::cos(a0) * kRadius};
    vector p1{std::sin(a1) * kRadius, -half_height, std::cos(a1) * kRadius};
    vector p2{std::sin(a1) * kRadius, half_height, std::cos(a1) * kRadius};
    vector p3{std::sin(a0) * kRadius, half_height, std::cos(a0) * kRadius};

    g3Point points[4];
    g3Point *point_list[4] = {&points[0], &points[1], &points[2], &points[3]};

    g3_RotatePoint(&points[0], &p0);
    g3_RotatePoint(&points[1], &p1);
    g3_RotatePoint(&points[2], &p2);
    g3_RotatePoint(&points[3], &p3);

    g3_DrawPoly(4, point_list, 0);
  }
}
} // namespace

void VR_InitFromCommandLine() {
  Vr_enabled = FindArg("-vr") != 0 || FindArg("-vrstereo") != 0;
  Vr_render_mode = FindArg("-vrstereo") ? VrRenderMode::Stereo : VrRenderMode::Cinema;
  if (!Vr_enabled) {
    return;
  }

  vr::EVRInitError error = vr::VRInitError_None;
  Vr_system = vr::VR_Init(&error, vr::VRApplication_Scene);
  if (error != vr::VRInitError_None) {
    LOG_WARNING.printf("OpenVR init failed: %s", vr::VR_GetVRInitErrorAsEnglishDescription(error));
    Vr_system = nullptr;
    Vr_enabled = false;
    return;
  }

  Vr_openvr_ready = true;
  if (!vr::VRCompositor()) {
    LOG_WARNING << "OpenVR compositor unavailable.";
    Vr_openvr_ready = false;
  }

  if (Vr_system) {
    auto left = Vr_system->GetEyeToHeadTransform(vr::Eye_Left);
    auto right = Vr_system->GetEyeToHeadTransform(vr::Eye_Right);
    const float separation = std::fabs(right.m[0][3] - left.m[0][3]);
    if (separation > 0.0f) {
      Vr_eye_separation = separation;
    }
  }

  if (Vr_openvr_ready) {
    uint32_t target_w = 0;
    uint32_t target_h = 0;
    Vr_system->GetRecommendedRenderTargetSize(&target_w, &target_h);
    const char *mode_label = (Vr_render_mode == VrRenderMode::Stereo) ? "stereo" : "cinema";
    LOG_INFO.printf("OpenVR enabled via -vr (%s). Recommended render target %ux%u.", mode_label, target_w, target_h);
  }
}

bool VR_IsEnabled() {
  return Vr_enabled;
}

VrRenderMode VR_GetRenderMode() {
  return Vr_render_mode;
}

bool VR_IsStereoRendering() {
  return Vr_enabled && (Vr_render_mode == VrRenderMode::Stereo || GetGameState() == GAMESTATE_LVLPLAYING);
}

float VR_GetStereoEyeSeparation() {
  return Vr_enabled ? Vr_eye_separation : 0.0f;
}

void VR_RenderMenuFrame() {
  if (!Vr_enabled) {
    return;
  }

  if (!Vr_openvr_ready || !Vr_system || Renderer_type != RENDERER_OPENGL) {
    return;
  }

  VR_UpdateOpenVRPoses();
  VR_EnsureSubmitTexture();
  if (Vr_submit_width == 0 || Vr_submit_height == 0) {
    return;
  }

  VR_EnsureMenuBitmap();
  if (Vr_menu_bitmap < 0) {
    return;
  }

  auto screenshot = rend_Screenshot();
  if (screenshot && screenshot->getData()) {
    VR_UpdateMenuTexture(*screenshot);
    if (VR_IsStereoRendering()) {
      if (!VR_EnsureSubmitSurface(Vr_submit_left) || !VR_EnsureSubmitSurface(Vr_submit_right)) {
        return;
      }
      VR_UpdateSubmitSurface(*screenshot, Vr_submit_left, true, 0);
      VR_UpdateSubmitSurface(*screenshot, Vr_submit_right, true, 1);
    } else {
      if (!VR_EnsureSubmitSurface(Vr_submit_cinema)) {
        return;
      }
      VR_UpdateSubmitSurface(*screenshot, Vr_submit_cinema, true, 0);
    }
  }

  StartFrame(0, 0, Max_window_w, Max_window_h);
  rend_ClearScreen(GR_BLACK);

  EndFrame();

  if (Vr_openvr_ready && vr::VRCompositor()) {
    if (VR_IsStereoRendering()) {
      if (Vr_submit_left.texture != 0 && Vr_submit_right.texture != 0) {
        vr::Texture_t left_texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_left.texture)),
                                      vr::TextureType_OpenGL, vr::ColorSpace_Auto};
        vr::Texture_t right_texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_right.texture)),
                                       vr::TextureType_OpenGL, vr::ColorSpace_Auto};
        vr::VRCompositor()->Submit(vr::Eye_Left, &left_texture);
        vr::VRCompositor()->Submit(vr::Eye_Right, &right_texture);
      }
    } else if (Vr_submit_cinema.texture != 0) {
      vr::Texture_t texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_cinema.texture)),
                               vr::TextureType_OpenGL, vr::ColorSpace_Auto};
      vr::VRCompositor()->Submit(vr::Eye_Left, &texture);
      vr::VRCompositor()->Submit(vr::Eye_Right, &texture);
    }
  }
}

void VR_SubmitStereoFrame(const NewBitmap &left, const NewBitmap &right) {
  if (!VR_IsStereoRendering()) {
    return;
  }

  if (!Vr_openvr_ready || !Vr_system || Renderer_type != RENDERER_OPENGL) {
    return;
  }

  VR_UpdateOpenVRPoses();
  VR_EnsureSubmitTexture();
  if (Vr_submit_width == 0 || Vr_submit_height == 0) {
    return;
  }

  if (!VR_EnsureSubmitSurface(Vr_submit_left) || !VR_EnsureSubmitSurface(Vr_submit_right)) {
    return;
  }

  VR_UpdateSubmitSurface(left, Vr_submit_left, false, 0);
  VR_UpdateSubmitSurface(right, Vr_submit_right, false, 1);

  if (Vr_openvr_ready && vr::VRCompositor()) {
    vr::Texture_t left_texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_left.texture)),
                                  vr::TextureType_OpenGL, vr::ColorSpace_Auto};
    vr::Texture_t right_texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_right.texture)),
                                   vr::TextureType_OpenGL, vr::ColorSpace_Auto};
    vr::VRCompositor()->Submit(vr::Eye_Left, &left_texture);
    vr::VRCompositor()->Submit(vr::Eye_Right, &right_texture);
  }
}

void VR_ResetGraphicsResources() {
  if (!Vr_enabled) {
    return;
  }

  VR_DeleteSubmitSurface(Vr_submit_cinema);
  VR_DeleteSubmitSurface(Vr_submit_left);
  VR_DeleteSubmitSurface(Vr_submit_right);
  Vr_submit_width = 0;
  Vr_submit_height = 0;

  if (Vr_menu_bitmap >= 0) {
    bm_FreeBitmap(Vr_menu_bitmap);
    Vr_menu_bitmap = -1;
  }
  Vr_menu_width = 0;
  Vr_menu_height = 0;
  Vr_menu_texture_size = 0;
}

void VR_Shutdown() {
  VR_DeleteSubmitSurface(Vr_submit_cinema);
  VR_DeleteSubmitSurface(Vr_submit_left);
  VR_DeleteSubmitSurface(Vr_submit_right);
  Vr_submit_width = 0;
  Vr_submit_height = 0;
  if (Vr_system) {
    vr::VR_Shutdown();
    Vr_system = nullptr;
  }
  Vr_openvr_ready = false;
  Vr_enabled = false;
}
