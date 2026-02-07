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
vr::IVRSystem *Vr_system = nullptr;
GLuint Vr_submit_texture = 0;
uint32_t Vr_submit_width = 0;
uint32_t Vr_submit_height = 0;
std::vector<uint32_t> Vr_submit_buffer;
int Vr_menu_bitmap = -1;
int Vr_menu_width = 0;
int Vr_menu_height = 0;
int Vr_menu_texture_size = 0;

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
  const uint32_t src_w = stereo_side_by_side ? (w / 2) : w;
  uint16_t *dest_data = bm_data(Vr_menu_bitmap, 0);
  if (!dest_data) {
    return;
  }

  const int dest_size = Vr_menu_texture_size;
  for (int y = 0; y < dest_size; ++y) {
    for (int x = 0; x < dest_size; ++x) {
      uint16_t pixel = GR_RGB16(0, 0, 0);
      if (x < static_cast<int>(src_w) && y < static_cast<int>(h)) {
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

  if (Vr_submit_texture != 0 && Vr_submit_width == target_w && Vr_submit_height == target_h) {
    return;
  }

  if (Vr_submit_texture != 0) {
    gl.delete_textures(1, &Vr_submit_texture);
    Vr_submit_texture = 0;
  }

  gl.gen_textures(1, &Vr_submit_texture);
  gl.bind_texture(GL_TEXTURE_2D, Vr_submit_texture);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gl.tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA8, target_w, target_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  Vr_submit_width = target_w;
  Vr_submit_height = target_h;
  Vr_submit_buffer.resize(static_cast<size_t>(target_w) * static_cast<size_t>(target_h));
}

void VR_UpdateSubmitTexture(const NewBitmap &screenshot) {
  if (!Vr_openvr_ready || Vr_submit_texture == 0 || Vr_submit_width == 0 || Vr_submit_height == 0) {
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
  const bool stereo_side_by_side = (src_h > 0) && (src_w >= src_h * 2);
  const uint32_t sample_width = stereo_side_by_side ? (src_w / 2) : src_w;

  for (uint32_t y = 0; y < Vr_submit_height; ++y) {
    const uint32_t src_y = (y * src_h) / Vr_submit_height;
    for (uint32_t x = 0; x < Vr_submit_width; ++x) {
      const uint32_t src_x = (x * sample_width) / Vr_submit_width;
      const uint32_t spix = src_data[src_y * src_w + src_x];
      Vr_submit_buffer[((Vr_submit_height - 1) - y) * Vr_submit_width + x] = spix;
    }
  }

  gl.bind_texture(GL_TEXTURE_2D, Vr_submit_texture);
  gl.tex_sub_image_2d(GL_TEXTURE_2D, 0, 0, 0, Vr_submit_width, Vr_submit_height, GL_RGBA, GL_UNSIGNED_BYTE,
                      Vr_submit_buffer.data());
}

void VR_UpdateOpenVRPoses() {
  if (!Vr_openvr_ready || !vr::VRCompositor()) {
    return;
  }

  vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
  vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
}

void VR_DrawCinemaScreen(int texture_handle, float u_max, float v_max) {
  constexpr int kSegments = 32;
  constexpr float kArcDegrees = 100.0f;
  constexpr float kRadius = 6.0f;
  constexpr float kHeight = 3.0f;

  const float arc_radians = kArcDegrees * (kPi / 180.0f);
  const float start_angle = -arc_radians * 0.5f;
  const float delta = arc_radians / static_cast<float>(kSegments);
  const float half_height = kHeight * 0.5f;

  rend_SetZBufferState(0);
  rend_SetTextureType(TT_LINEAR);
  rend_SetLighting(LS_NONE);
  rend_SetAlphaType(AT_CONSTANT_TEXTURE);
  rend_SetAlphaValue(255);

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

    const float u0 = static_cast<float>(i) / static_cast<float>(kSegments);
    const float u1 = static_cast<float>(i + 1) / static_cast<float>(kSegments);

    points[0].p3_flags |= PF_UV;
    points[1].p3_flags |= PF_UV;
    points[2].p3_flags |= PF_UV;
    points[3].p3_flags |= PF_UV;

    points[0].p3_u = u0 * u_max;
    points[0].p3_v = 0.0f;
    points[1].p3_u = u1 * u_max;
    points[1].p3_v = 0.0f;
    points[2].p3_u = u1 * u_max;
    points[2].p3_v = v_max;
    points[3].p3_u = u0 * u_max;
    points[3].p3_v = v_max;

    g3_DrawPoly(4, point_list, texture_handle);
  }
}
} // namespace

void VR_InitFromCommandLine() {
  Vr_enabled = FindArg("-vr") != 0;
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

  if (Vr_openvr_ready) {
    uint32_t target_w = 0;
    uint32_t target_h = 0;
    Vr_system->GetRecommendedRenderTargetSize(&target_w, &target_h);
    LOG_INFO.printf("OpenVR enabled via -vr. Recommended render target %ux%u.", target_w, target_h);
  }
}

bool VR_IsEnabled() {
  return Vr_enabled;
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
  if (Vr_submit_texture == 0) {
    return;
  }

  VR_EnsureMenuBitmap();
  if (Vr_menu_bitmap < 0) {
    return;
  }

  auto screenshot = rend_Screenshot();
  if (screenshot && screenshot->getData()) {
    VR_UpdateMenuTexture(*screenshot);
    VR_UpdateSubmitTexture(*screenshot);
  }

  StartFrame(0, 0, Max_window_w, Max_window_h);
  rend_ClearScreen(GR_BLACK);

  vector view_pos{0.0f, 0.0f, 0.0f};
  matrix view_orient = Identity_matrix;
  g3_StartFrame(&view_pos, &view_orient, D3_DEFAULT_ZOOM);

  const float u_max = static_cast<float>(Vr_menu_width) / static_cast<float>(Vr_menu_texture_size);
  const float v_max = static_cast<float>(Vr_menu_height) / static_cast<float>(Vr_menu_texture_size);
  VR_DrawCinemaScreen(Vr_menu_bitmap, u_max, v_max);

  g3_EndFrame();
  EndFrame();

  if (Vr_openvr_ready && Vr_submit_texture != 0 && vr::VRCompositor()) {
    vr::Texture_t texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_texture)),
                             vr::TextureType_OpenGL, vr::ColorSpace_Auto};
    vr::VRCompositor()->Submit(vr::Eye_Left, &texture);
    vr::VRCompositor()->Submit(vr::Eye_Right, &texture);
  }
}

void VR_ResetGraphicsResources() {
  if (!Vr_enabled) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (Vr_submit_texture != 0) {
    if (gl.delete_textures) {
      gl.delete_textures(1, &Vr_submit_texture);
    }
    Vr_submit_texture = 0;
  }
  Vr_submit_width = 0;
  Vr_submit_height = 0;
  Vr_submit_buffer.clear();

  if (Vr_menu_bitmap >= 0) {
    bm_FreeBitmap(Vr_menu_bitmap);
    Vr_menu_bitmap = -1;
  }
  Vr_menu_width = 0;
  Vr_menu_height = 0;
  Vr_menu_texture_size = 0;
}

void VR_Shutdown() {
  auto &gl = VR_GetGlFns();
  if (Vr_submit_texture != 0) {
    if (gl.delete_textures) {
      gl.delete_textures(1, &Vr_submit_texture);
    }
    Vr_submit_texture = 0;
  }
  Vr_submit_width = 0;
  Vr_submit_height = 0;
  Vr_submit_buffer.clear();
  if (Vr_system) {
    vr::VR_Shutdown();
    Vr_system = nullptr;
  }
  Vr_openvr_ready = false;
  Vr_enabled = false;
}
