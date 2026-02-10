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
#include "3d.h"

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
VrRenderMode Vr_render_mode = VrRenderMode::Cinema;
vr::IVRSystem *Vr_system = nullptr;
float Vr_eye_separation = 0.064f;
uint32_t Vr_submit_width = 0;
uint32_t Vr_submit_height = 0;
struct VrSubmitSurface {
  GLuint texture = 0;
  std::vector<uint32_t> buffer;
};
VrSubmitSurface Vr_submit_left;
VrSubmitSurface Vr_submit_right;
GLuint Vr_menu_fbo = 0;
GLuint Vr_menu_fbo_texture = 0;
int Vr_menu_bitmap = -1;
int Vr_menu_width = 0;
int Vr_menu_height = 0;
int Vr_menu_texture_size = 0;
bool Vr_menu_texture_registered = false;

void VR_DeleteSubmitSurface(VrSubmitSurface &surface);

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kVrTargetWidth = 2000;
constexpr uint32_t kVrTargetHeight = 2040;

struct VrGlFns {
  bool loaded = false;
  using GenTexturesFn = decltype(&glGenTextures);
  using DeleteTexturesFn = decltype(&glDeleteTextures);
  using BindTextureFn = decltype(&glBindTexture);
  using TexParameteriFn = decltype(&glTexParameteri);
  using TexImage2DFn = decltype(&glTexImage2D);
  using TexSubImage2DFn = decltype(&glTexSubImage2D);
  using GenFramebuffersFn = void (*)(GLsizei, GLuint*);
  using DeleteFramebuffersFn = void (*)(GLsizei, const GLuint*);
  using BindFramebufferFn = void (*)(GLenum, GLuint);
  using FramebufferTexture2DFn = void (*)(GLenum, GLenum, GLenum, GLuint, GLint);
  using CheckFramebufferStatusFn = GLenum (*)(GLenum);

  GenTexturesFn gen_textures = nullptr;
  DeleteTexturesFn delete_textures = nullptr;
  BindTextureFn bind_texture = nullptr;
  TexParameteriFn tex_parameteri = nullptr;
  TexImage2DFn tex_image_2d = nullptr;
  TexSubImage2DFn tex_sub_image_2d = nullptr;
  GenFramebuffersFn gen_framebuffers = nullptr;
  DeleteFramebuffersFn delete_framebuffers = nullptr;
  BindFramebufferFn bind_framebuffer = nullptr;
  FramebufferTexture2DFn framebuffer_texture_2d = nullptr;
  CheckFramebufferStatusFn check_framebuffer_status = nullptr;
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
  fns.gen_framebuffers = reinterpret_cast<VrGlFns::GenFramebuffersFn>(SDL_GL_GetProcAddress("glGenFramebuffers"));
  fns.delete_framebuffers = reinterpret_cast<VrGlFns::DeleteFramebuffersFn>(SDL_GL_GetProcAddress("glDeleteFramebuffers"));
  fns.bind_framebuffer = reinterpret_cast<VrGlFns::BindFramebufferFn>(SDL_GL_GetProcAddress("glBindFramebuffer"));
  fns.framebuffer_texture_2d = reinterpret_cast<VrGlFns::FramebufferTexture2DFn>(SDL_GL_GetProcAddress("glFramebufferTexture2D"));
  fns.check_framebuffer_status = reinterpret_cast<VrGlFns::CheckFramebufferStatusFn>(SDL_GL_GetProcAddress("glCheckFramebufferStatus"));
  
  fns.loaded = fns.gen_textures && fns.delete_textures && fns.bind_texture && fns.tex_parameteri && fns.tex_image_2d &&
               fns.tex_sub_image_2d && fns.gen_framebuffers && fns.delete_framebuffers && fns.bind_framebuffer &&
               fns.framebuffer_texture_2d && fns.check_framebuffer_status;
  return fns;
}

int VR_NextPowerOfTwo(int value) {
  int size = 1;
  while (size < value) {
    size <<= 1;
  }
  return size;
}

// In VR_EnsureMenuBitmap(), change the desired size:
void VR_EnsureMenuBitmap() {
  // Use VR target resolution for higher quality
  const int desired_width = Max_window_w;
  const int desired_height = Max_window_h;
  const int desired_texture_size = VR_NextPowerOfTwo(std::max(desired_width, desired_height));

  if (Vr_menu_width == desired_width && Vr_menu_height == desired_height && Vr_menu_texture_size == desired_texture_size &&
      Vr_menu_bitmap >= 0 && Vr_menu_fbo != 0) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.gen_framebuffers || !gl.gen_textures || !gl.bind_framebuffer || !gl.bind_texture || 
      !gl.tex_parameteri || !gl.tex_image_2d || !gl.framebuffer_texture_2d || !gl.check_framebuffer_status) {
    static bool warned = false;
    if (!warned) {
      LOG_WARNING << "VR: Missing GL functions for framebuffer creation";
      warned = true;
    }
    return;
  }

  // Clean up old resources
  if (Vr_menu_fbo != 0) {
    gl.delete_framebuffers(1, &Vr_menu_fbo);
    Vr_menu_fbo = 0;
  }
  if (Vr_menu_fbo_texture != 0) {
    gl.delete_textures(1, &Vr_menu_fbo_texture);
    Vr_menu_fbo_texture = 0;
  }
  if (Vr_menu_bitmap >= 0) {
    if (Vr_menu_texture_registered) {
      rend_UnregisterExternalTexture(Vr_menu_bitmap);
      Vr_menu_texture_registered = false;
    }
    bm_FreeBitmap(Vr_menu_bitmap);
    Vr_menu_bitmap = -1;
  }

  Vr_menu_width = desired_width;
  Vr_menu_height = desired_height;
  Vr_menu_texture_size = desired_texture_size;

  LOG_INFO.printf("VR: Creating menu FBO at %dx%d (texture size %d)", 
                  Vr_menu_width, Vr_menu_height, Vr_menu_texture_size);

  // Create FBO texture at VR resolution
  gl.gen_textures(1, &Vr_menu_fbo_texture);
  gl.bind_texture(GL_TEXTURE_2D, Vr_menu_fbo_texture);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gl.tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA8, Vr_menu_width, Vr_menu_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  // Create FBO
  gl.gen_framebuffers(1, &Vr_menu_fbo);
  gl.bind_framebuffer(GL_FRAMEBUFFER, Vr_menu_fbo);
  gl.framebuffer_texture_2d(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Vr_menu_fbo_texture, 0);

  GLenum status = gl.check_framebuffer_status(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    LOG_WARNING << "VR: Menu framebuffer is not complete!";
    gl.delete_framebuffers(1, &Vr_menu_fbo);
    gl.delete_textures(1, &Vr_menu_fbo_texture);
    Vr_menu_fbo = 0;
    Vr_menu_fbo_texture = 0;
    gl.bind_framebuffer(GL_FRAMEBUFFER, 0);
    return;
  }

  gl.bind_framebuffer(GL_FRAMEBUFFER, 0);

  // Create bitmap
  Vr_menu_bitmap = bm_AllocBitmap(Vr_menu_texture_size, Vr_menu_texture_size, 0);
  if (Vr_menu_bitmap < 0) {
    LOG_WARNING << "VR: Unable to allocate menu bitmap for cinema screen.";
  } else {
    Vr_menu_texture_registered =
        rend_RegisterExternalTexture(Vr_menu_bitmap, static_cast<unsigned int>(Vr_menu_fbo_texture),
                                     Vr_menu_width, Vr_menu_height);
    if (!Vr_menu_texture_registered) {
      LOG_WARNING << "VR: Failed to register menu framebuffer texture with renderer.";
    }
  }
}

void VR_BlitMenuTextureToWindow() {
  if (Vr_menu_fbo_texture == 0) {
    return;
  }

  // Render the menu FBO texture to the window as a fullscreen quad
  // This displays what was rendered to the FBO
  StartFrame(0, 0, Max_window_w, Max_window_h);
  
  // Here we'd render a fullscreen textured quad with Vr_menu_fbo_texture
  // For now, we can use the game's existing rendering system
  // The game's renderer should be able to blit the texture
  
  EndFrame();
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

  uint32_t target_w = kVrTargetWidth;
  uint32_t target_h = kVrTargetHeight;

  if (Vr_submit_width == target_w && Vr_submit_height == target_h) {
    return;
  }

  if (Vr_submit_width != 0 || Vr_submit_height != 0) {
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

void VR_UpdateSubmitSurface(const NewBitmap &screenshot, VrSubmitSurface &surface, bool allow_stereo_crop) {
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

  for (uint32_t y = 0; y < Vr_submit_height; ++y) {
    const uint32_t src_y = (y * sample_height) / Vr_submit_height;
    for (uint32_t x = 0; x < Vr_submit_width; ++x) {
      const uint32_t src_x = (x * sample_width) / Vr_submit_width;
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

void VR_DrawCinemaScreen(int texture_handle, float u_max, float v_max, float arc_degrees, float radius, float height) {
  const int num_segments = 32;
  const float arc_radians = arc_degrees * (3.14159265f / 180.0f);
  const float angle_step = arc_radians / num_segments;
  const float start_angle = -arc_radians / 2.0f;
  
  g3Point points[(num_segments + 1) * 2];
  g3Point *top_strip[num_segments + 1];
  g3Point *bottom_strip[num_segments + 1];
  
  for (int i = 0; i <= num_segments; ++i) {
    float angle = start_angle + (i * angle_step);
    float x = sinf(angle) * radius;
    float z = cosf(angle) * radius;
    
    vector top_pos{x, height / 2.0f, z};
    vector bottom_pos{x, -height / 2.0f, z};
    
    int top_idx = i * 2;
    int bottom_idx = i * 2 + 1;
    
    g3_RotatePoint(&points[top_idx], &top_pos);
    g3_RotatePoint(&points[bottom_idx], &bottom_pos);
    
    points[top_idx].p3_flags |= PF_UV;
    points[bottom_idx].p3_flags |= PF_UV;
    
    float u = (float)i / num_segments * u_max;
    // FLIP V: top gets v_max, bottom gets 0
    points[top_idx].p3_u = u;
    points[top_idx].p3_v = v_max;  // Changed from 0.0f
    points[bottom_idx].p3_u = u;
    points[bottom_idx].p3_v = 0.0f;  // Changed from v_max
    
    top_strip[i] = &points[top_idx];
    bottom_strip[i] = &points[bottom_idx];
  }
  
  rend_SetTextureType(TT_LINEAR);
  rend_SetLighting(LS_NONE);
  rend_SetAlphaType(AT_CONSTANT);
  rend_SetAlphaValue(255);
  
  for (int i = 0; i < num_segments; ++i) {
    g3Point *quad[4] = {
      top_strip[i],
      top_strip[i + 1],
      bottom_strip[i + 1],
      bottom_strip[i]
    };
    g3_DrawPoly(4, quad, texture_handle);
  }
}

void VR_DrawTestQuad(int texture_handle, float u_max, float v_max) {
  constexpr float width = 3.0f;
  constexpr float height = 2.25f;  // 4:3 aspect ratio
  constexpr float distance = 5.0f;
  
  // Quad in front of camera, camera looks down -Z
  vector p0{-width/2, height/2, -distance};
  vector p1{width/2, height/2, -distance};
  vector p2{width/2, -height/2, -distance};
  vector p3{-width/2, -height/2, -distance};

  g3Point points[4];
  g3Point *point_list[4] = {&points[0], &points[1], &points[2], &points[3]};

  g3_RotatePoint(&points[0], &p0);
  g3_RotatePoint(&points[1], &p1);
  g3_RotatePoint(&points[2], &p2);
  g3_RotatePoint(&points[3], &p3);
  
  LOG_INFO.printf("VR: Transformed points - p0=(%f,%f,%f) p1=(%f,%f,%f)", 
                  points[0].p3_sx, points[0].p3_sy, points[0].p3_z,
                  points[1].p3_sx, points[1].p3_sy, points[1].p3_z);

  // Set texture coordinates with V flipped
  points[0].p3_flags |= PF_UV;
  points[1].p3_flags |= PF_UV;
  points[2].p3_flags |= PF_UV;
  points[3].p3_flags |= PF_UV;

  points[0].p3_u = 0.0f;
  points[0].p3_v = v_max;
  points[1].p3_u = u_max;
  points[1].p3_v = v_max;
  points[2].p3_u = u_max;
  points[2].p3_v = 0.0f;
  points[3].p3_u = 0.0f;
  points[3].p3_v = 0.0f;

  rend_SetZBufferState(0);
  rend_SetTextureType(TT_LINEAR);
  rend_SetLighting(LS_NONE);
  rend_SetAlphaType(AT_CONSTANT_TEXTURE);
  rend_SetAlphaValue(255);
  
  g3_DrawPoly(4, point_list, texture_handle);
}

// Helper function to render the 3D cinema screen for one eye
// Call this ONCE during VR initialization (in VR_Init or similar)
void VR_InitStereoFrustums() {
  if (!Vr_system) return;
  
  g3StereoFrustum left_frustum, right_frustum;
  
  Vr_system->GetProjectionRaw(vr::Eye_Left, &left_frustum.left, &left_frustum.right,
                               &left_frustum.top, &left_frustum.bottom);
  Vr_system->GetProjectionRaw(vr::Eye_Right, &right_frustum.left, &right_frustum.right,
                               &right_frustum.top, &right_frustum.bottom);
  
  LOG_INFO.printf("VR: Setting stereo frustums - Left(L=%f,R=%f,T=%f,B=%f) Right(L=%f,R=%f,T=%f,B=%f)",
                  left_frustum.left, left_frustum.right, left_frustum.top, left_frustum.bottom,
                  right_frustum.left, right_frustum.right, right_frustum.top, right_frustum.bottom);
  
  g3_SetStereoFrustum(&left_frustum, &right_frustum);
}

// Then in your render function:
void VR_RenderCinemaScreenForEye(VrSubmitSurface &surface, bool is_left_eye) {
  if (!VR_EnsureSubmitSurface(surface)) {
    return;
  }

  if (Vr_menu_fbo_texture == 0 || Vr_menu_bitmap < 0) {
    return;
  }

  constexpr float kCinemaArcDegrees = 120.0f;
  constexpr float kCinemaScreenRadius = 5.0f;
  constexpr float kCinemaScreenHeight = 3.0f;

  int render_width = Max_window_w;
  int render_height = Max_window_h;

  StartFrame(0, 0, render_width, render_height);
  rend_ClearScreen(GR_BLACK);

  // Keep both eyes centered on the same head-space midpoint and let the stereo
  // projection offset each eye/frustum. This ensures the convergence midpoint
  // stays in the overlap region between both eyes instead of being recentered
  // per-eye.
  vector camera_pos{0.0f, 0.0f, 0.0f};
  matrix camera_orient = Identity_matrix;
  float zoom = D3_DEFAULT_ZOOM;
  const bool stereo_projection = (Vr_render_mode == VrRenderMode::Stereo);

  if (stereo_projection) {
    // Converge at the center of the curved cinema display (angle = 0 section).
    // Use forward-axis depth from the shared head midpoint so both eyes use the
    // same convergence plane.
    const vector cinema_center{0.0f, 0.0f, kCinemaScreenRadius};
    const vector to_cinema_center = cinema_center - camera_pos;
    const float convergence_distance = to_cinema_center * camera_orient.fvec;
    g3_StartFrameStereo(&camera_pos, &camera_orient, zoom, is_left_eye, VR_GetStereoEyeSeparation(),
                        convergence_distance);
  } else {
    g3_StartFrame(&camera_pos, &camera_orient, zoom);
  }

  float u_max = Vr_menu_texture_registered
                    ? 1.0f
                    : static_cast<float>(Vr_menu_width) / static_cast<float>(Vr_menu_texture_size);
  float v_max = Vr_menu_texture_registered
                    ? 1.0f
                    : static_cast<float>(Vr_menu_height) / static_cast<float>(Vr_menu_texture_size);

  VR_DrawCinemaScreen(Vr_menu_bitmap, u_max, v_max, kCinemaArcDegrees, kCinemaScreenRadius, kCinemaScreenHeight);

  g3_EndFrame();
  EndFrame();

  auto screenshot = rend_Screenshot();
  if (screenshot && screenshot->getData()) {
    VR_UpdateSubmitSurface(*screenshot, surface, false);
  }
}


} // namespace

void VR_BeginMenuFramebufferRender() {
  if (Vr_menu_fbo == 0) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.bind_framebuffer) {
    return;
  }

  // Bind the menu FBO so all subsequent rendering goes to our texture
  gl.bind_framebuffer(GL_FRAMEBUFFER, Vr_menu_fbo);
}

void VR_EndMenuFramebufferRender() {
  auto &gl = VR_GetGlFns();
  if (!gl.bind_framebuffer) {
    return;
  }

  // Unbind the FBO, returning to normal window rendering
  gl.bind_framebuffer(GL_FRAMEBUFFER, 0);
}

void VR_InitFromCommandLine() {
  Vr_enabled = FindArg("-vr") != 0;
  Vr_render_mode = VrRenderMode::Stereo;
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

    float left_l = 0.0f;
    float left_r = 0.0f;
    float left_t = 0.0f;
    float left_b = 0.0f;
    float right_l = 0.0f;
    float right_r = 0.0f;
    float right_t = 0.0f;
    float right_b = 0.0f;
    Vr_system->GetProjectionRaw(vr::Eye_Left, &left_l, &left_r, &left_t, &left_b);
    Vr_system->GetProjectionRaw(vr::Eye_Right, &right_l, &right_r, &right_t, &right_b);

    g3StereoFrustum left_frustum{left_l, left_r, -left_t, -left_b};
    g3StereoFrustum right_frustum{right_l, right_r, -right_t, -right_b};
    g3_SetStereoFrustum(&left_frustum, &right_frustum);
  }

  if (Vr_openvr_ready) {
    uint32_t target_w = kVrTargetWidth;
    uint32_t target_h = kVrTargetHeight;
    LOG_INFO.printf("OpenVR enabled via -vr. Recommended render target %ux%u.", target_w, target_h);
  }
}

bool VR_IsEnabled() {
  return Vr_enabled;
}

VrRenderMode VR_GetRenderMode() {
  return Vr_render_mode;
}

bool VR_IsStereoRendering() {
  return Vr_enabled && Vr_render_mode == VrRenderMode::Stereo;
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
  if (Vr_menu_bitmap < 0 || Vr_menu_fbo == 0) {
    return;
  }

  // The menu has already been rendered to Vr_menu_fbo_texture
  
  // Render the curved cinema screen for each eye
  if (Vr_render_mode == VrRenderMode::Stereo) {
    VR_RenderCinemaScreenForEye(Vr_submit_left, true);
    VR_RenderCinemaScreenForEye(Vr_submit_right, false);
  } else {
    // Cinema mode: both eyes see the same centered view
    VR_RenderCinemaScreenForEye(Vr_submit_left, true);
    VR_RenderCinemaScreenForEye(Vr_submit_right, false);
  }

  // Blit the menu texture to the monitor window
  VR_BlitMenuTextureToWindow();

  // Submit to OpenVR
  if (Vr_openvr_ready && vr::VRCompositor()) {
    if (Vr_submit_left.texture != 0 && Vr_submit_right.texture != 0) {
      vr::Texture_t left_texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_left.texture)),
                                    vr::TextureType_OpenGL, vr::ColorSpace_Auto};
      vr::Texture_t right_texture = {reinterpret_cast<void *>(static_cast<uintptr_t>(Vr_submit_right.texture)),
                                     vr::TextureType_OpenGL, vr::ColorSpace_Auto};
      vr::VRCompositor()->Submit(vr::Eye_Left, &left_texture);
      vr::VRCompositor()->Submit(vr::Eye_Right, &right_texture);
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

  VR_UpdateSubmitSurface(left, Vr_submit_left, false);
  VR_UpdateSubmitSurface(right, Vr_submit_right, false);

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

  auto &gl = VR_GetGlFns();
  
  VR_DeleteSubmitSurface(Vr_submit_left);
  VR_DeleteSubmitSurface(Vr_submit_right);
  Vr_submit_width = 0;
  Vr_submit_height = 0;

  if (Vr_menu_fbo != 0 && gl.delete_framebuffers) {
    gl.delete_framebuffers(1, &Vr_menu_fbo);
    Vr_menu_fbo = 0;
  }
  
  if (Vr_menu_fbo_texture != 0 && gl.delete_textures) {
    gl.delete_textures(1, &Vr_menu_fbo_texture);
    Vr_menu_fbo_texture = 0;
  }

  if (Vr_menu_bitmap >= 0) {
    if (Vr_menu_texture_registered) {
      rend_UnregisterExternalTexture(Vr_menu_bitmap);
      Vr_menu_texture_registered = false;
    }
    bm_FreeBitmap(Vr_menu_bitmap);
    Vr_menu_bitmap = -1;
  }
  Vr_menu_width = 0;
  Vr_menu_height = 0;
  Vr_menu_texture_size = 0;
}

void VR_Shutdown() {
  auto &gl = VR_GetGlFns();
  
  VR_DeleteSubmitSurface(Vr_submit_left);
  VR_DeleteSubmitSurface(Vr_submit_right);
  Vr_submit_width = 0;
  Vr_submit_height = 0;
  
  if (Vr_menu_fbo != 0 && gl.delete_framebuffers) {
    gl.delete_framebuffers(1, &Vr_menu_fbo);
    Vr_menu_fbo = 0;
  }
  
  if (Vr_menu_fbo_texture != 0 && gl.delete_textures) {
    gl.delete_textures(1, &Vr_menu_fbo_texture);
    Vr_menu_fbo_texture = 0;
  }

  if (Vr_menu_bitmap >= 0 && Vr_menu_texture_registered) {
    rend_UnregisterExternalTexture(Vr_menu_bitmap);
    Vr_menu_texture_registered = false;
  }
  
  if (Vr_system) {
    vr::VR_Shutdown();
    Vr_system = nullptr;
  }
  Vr_openvr_ready = false;
  Vr_enabled = false;
}
