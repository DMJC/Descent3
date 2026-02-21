/*
* Descent 3
* Copyright (C) 2024 Parallax Software
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#include "vr.h"
#include "3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "args.h"
#include "bitmap.h"
#include "descent.h"
#include "game.h"
#include "log.h"
#include "NewBitmap.h"
#include "renderer.h"
#include "vecmat.h"
#include "../renderer/dyna_gl.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <openvr/openvr.h>

namespace {
bool Vr_enabled = false;
bool Vr_openvr_ready = false;
VrRenderMode Vr_render_mode = VrRenderMode::Stereo;
vr::IVRSystem *Vr_system = nullptr;
vr::IVRCompositor *Vr_compositor = nullptr;
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
GLuint Vr_submit_fbo = 0;
int Vr_menu_bitmap = -1;
int Vr_menu_width = 0;
int Vr_menu_height = 0;
int Vr_menu_texture_size = 0;
bool Vr_menu_texture_registered = false;
bool Vr_menu_fbo_support_missing_logged = false;
bool Vr_curved_menu_unavailable_logged = false;

struct VrGlFns {
  bool loaded = false;
  using GenTexturesFn = decltype(&glGenTextures);
  using DeleteTexturesFn = decltype(&glDeleteTextures);
  using BindTextureFn = decltype(&glBindTexture);
  using TexParameteriFn = decltype(&glTexParameteri);
  using TexImage2DFn = decltype(&glTexImage2D);
  using TexSubImage2DFn = decltype(&glTexSubImage2D);
  using CopyTexSubImage2DFn = decltype(&glCopyTexSubImage2D);
  using GenFramebuffersFn = void (*)(GLsizei, GLuint *);
  using DeleteFramebuffersFn = void (*)(GLsizei, const GLuint *);
  using BindFramebufferFn = void (*)(GLenum, GLuint);
  using FramebufferTexture2DFn = void (*)(GLenum, GLenum, GLenum, GLuint, GLint);
  using CheckFramebufferStatusFn = GLenum (*)(GLenum);
  using GetIntegervFn = void (*)(GLenum, GLint *);
  using ViewportFn = void (*)(GLint, GLint, GLsizei, GLsizei);
  using ClearColorFn = decltype(&glClearColor);
  using ClearFn = decltype(&glClear);
  using DisableFn = decltype(&glDisable);
  using EnableFn = decltype(&glEnable);
  using MatrixModeFn = decltype(&glMatrixMode);
  using PushMatrixFn = decltype(&glPushMatrix);
  using PopMatrixFn = decltype(&glPopMatrix);
  using LoadIdentityFn = decltype(&glLoadIdentity);
  using FrustumFn = decltype(&glFrustum);
  using TranslatefFn = decltype(&glTranslatef);
  using LoadMatrixfFn = decltype(&glLoadMatrixf);
  using MultMatrixfFn = decltype(&glMultMatrixf);
  using BeginFn = decltype(&glBegin);
  using EndFn = decltype(&glEnd);
  using TexCoord2fFn = decltype(&glTexCoord2f);
  using Vertex3fFn = decltype(&glVertex3f);

  GenTexturesFn gen_textures = nullptr;
  DeleteTexturesFn delete_textures = nullptr;
  BindTextureFn bind_texture = nullptr;
  TexParameteriFn tex_parameteri = nullptr;
  TexImage2DFn tex_image_2d = nullptr;
  TexSubImage2DFn tex_sub_image_2d = nullptr;
  CopyTexSubImage2DFn copy_tex_sub_image_2d = nullptr;
  GenFramebuffersFn gen_framebuffers = nullptr;
  DeleteFramebuffersFn delete_framebuffers = nullptr;
  BindFramebufferFn bind_framebuffer = nullptr;
  FramebufferTexture2DFn framebuffer_texture_2d = nullptr;
  CheckFramebufferStatusFn check_framebuffer_status = nullptr;
  GetIntegervFn get_integerv = nullptr;
  ViewportFn viewport = nullptr;
  ClearColorFn clear_color = nullptr;
  ClearFn clear = nullptr;
  DisableFn disable = nullptr;
  EnableFn enable = nullptr;
  MatrixModeFn matrix_mode = nullptr;
  PushMatrixFn push_matrix = nullptr;
  PopMatrixFn pop_matrix = nullptr;
  LoadIdentityFn load_identity = nullptr;
  FrustumFn frustum = nullptr;
  TranslatefFn translatef = nullptr;
  LoadMatrixfFn load_matrixf = nullptr;
  MultMatrixfFn mult_matrixf = nullptr;
  BeginFn begin = nullptr;
  EndFn end = nullptr;
  TexCoord2fFn tex_coord2f = nullptr;
  Vertex3fFn vertex3f = nullptr;
};

std::array<GLint, 4> Vr_saved_menu_viewport = {0, 0, 0, 0};
bool Vr_saved_menu_viewport_valid = false;

VrGlFns &VR_GetGlFns() {
  static VrGlFns fns;
  if (fns.loaded) {
    return fns;
  }

  // The first VR setup call can happen before the OpenGL context is active.
  // In that case SDL can't resolve GL entry points yet, so keep retrying
  // until a context is available instead of caching nullptr function pointers.
  if (SDL_GL_GetCurrentContext() == nullptr) {
    return fns;
  }

  fns = VrGlFns{};

  const auto load_proc = [](const char *primary, const char *fallback_ext = nullptr, const char *fallback_arb = nullptr) {
    void *proc = SDL_GL_GetProcAddress(primary);
    if (!proc && fallback_ext) {
      proc = SDL_GL_GetProcAddress(fallback_ext);
    }
    if (!proc && fallback_arb) {
      proc = SDL_GL_GetProcAddress(fallback_arb);
    }
    return proc;
  };

  fns.gen_textures = reinterpret_cast<VrGlFns::GenTexturesFn>(load_proc("glGenTextures"));
  fns.delete_textures = reinterpret_cast<VrGlFns::DeleteTexturesFn>(load_proc("glDeleteTextures"));
  fns.bind_texture = reinterpret_cast<VrGlFns::BindTextureFn>(load_proc("glBindTexture"));
  fns.tex_parameteri = reinterpret_cast<VrGlFns::TexParameteriFn>(load_proc("glTexParameteri"));
  fns.tex_image_2d = reinterpret_cast<VrGlFns::TexImage2DFn>(load_proc("glTexImage2D"));
  fns.tex_sub_image_2d = reinterpret_cast<VrGlFns::TexSubImage2DFn>(load_proc("glTexSubImage2D"));
  fns.copy_tex_sub_image_2d = reinterpret_cast<VrGlFns::CopyTexSubImage2DFn>(load_proc("glCopyTexSubImage2D"));
  fns.gen_framebuffers =
      reinterpret_cast<VrGlFns::GenFramebuffersFn>(load_proc("glGenFramebuffers", "glGenFramebuffersEXT", "glGenFramebuffersARB"));
  fns.delete_framebuffers = reinterpret_cast<VrGlFns::DeleteFramebuffersFn>(
      load_proc("glDeleteFramebuffers", "glDeleteFramebuffersEXT", "glDeleteFramebuffersARB"));
  fns.bind_framebuffer =
      reinterpret_cast<VrGlFns::BindFramebufferFn>(load_proc("glBindFramebuffer", "glBindFramebufferEXT", "glBindFramebufferARB"));
  fns.framebuffer_texture_2d = reinterpret_cast<VrGlFns::FramebufferTexture2DFn>(
      load_proc("glFramebufferTexture2D", "glFramebufferTexture2DEXT", "glFramebufferTexture2DARB"));
  fns.check_framebuffer_status = reinterpret_cast<VrGlFns::CheckFramebufferStatusFn>(
      load_proc("glCheckFramebufferStatus", "glCheckFramebufferStatusEXT", "glCheckFramebufferStatusARB"));
  fns.get_integerv = reinterpret_cast<VrGlFns::GetIntegervFn>(load_proc("glGetIntegerv"));
  fns.viewport = reinterpret_cast<VrGlFns::ViewportFn>(load_proc("glViewport"));
  fns.clear_color = reinterpret_cast<VrGlFns::ClearColorFn>(load_proc("glClearColor"));
  fns.clear = reinterpret_cast<VrGlFns::ClearFn>(load_proc("glClear"));
  fns.disable = reinterpret_cast<VrGlFns::DisableFn>(load_proc("glDisable"));
  fns.enable = reinterpret_cast<VrGlFns::EnableFn>(load_proc("glEnable"));
  fns.matrix_mode = reinterpret_cast<VrGlFns::MatrixModeFn>(load_proc("glMatrixMode"));
  fns.push_matrix = reinterpret_cast<VrGlFns::PushMatrixFn>(load_proc("glPushMatrix"));
  fns.pop_matrix = reinterpret_cast<VrGlFns::PopMatrixFn>(load_proc("glPopMatrix"));
  fns.load_identity = reinterpret_cast<VrGlFns::LoadIdentityFn>(load_proc("glLoadIdentity"));
  fns.frustum = reinterpret_cast<VrGlFns::FrustumFn>(load_proc("glFrustum"));
  fns.translatef = reinterpret_cast<VrGlFns::TranslatefFn>(load_proc("glTranslatef"));
  fns.load_matrixf = reinterpret_cast<VrGlFns::LoadMatrixfFn>(load_proc("glLoadMatrixf"));
  fns.mult_matrixf = reinterpret_cast<VrGlFns::MultMatrixfFn>(load_proc("glMultMatrixf"));
  fns.begin = reinterpret_cast<VrGlFns::BeginFn>(load_proc("glBegin"));
  fns.end = reinterpret_cast<VrGlFns::EndFn>(load_proc("glEnd"));
  fns.tex_coord2f = reinterpret_cast<VrGlFns::TexCoord2fFn>(load_proc("glTexCoord2f"));
  fns.vertex3f = reinterpret_cast<VrGlFns::Vertex3fFn>(load_proc("glVertex3f"));
  fns.loaded = true;
  return fns;
}

int VR_NextPowerOfTwo(int value) {
  if (value <= 0) {
    return 1;
  }

  int power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

void VR_DeleteSubmitSurface(VrSubmitSurface &surface) {
  auto &gl = VR_GetGlFns();
  if (surface.texture != 0 && gl.delete_textures) {
    gl.delete_textures(1, &surface.texture);
    surface.texture = 0;
  }
  surface.buffer.clear();
}

bool VR_EnsureSubmitSurface(VrSubmitSurface &surface) {
  if (Vr_submit_width == 0 || Vr_submit_height == 0) {
    return false;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.gen_textures || !gl.bind_texture || !gl.tex_parameteri || !gl.tex_image_2d) {
    return false;
  }

  if (surface.texture == 0) {
    gl.gen_textures(1, &surface.texture);
    gl.bind_texture(GL_TEXTURE_2D, surface.texture);
    gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA8, Vr_submit_width, Vr_submit_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  }

  if (surface.buffer.size() != static_cast<size_t>(Vr_submit_width * Vr_submit_height)) {
    surface.buffer.assign(static_cast<size_t>(Vr_submit_width * Vr_submit_height), 0u);
  }

  return true;
}

void VR_UpdateSubmitSurface(const NewBitmap &screenshot, VrSubmitSurface &surface) {
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
  if (!src_data || src_w == 0 || src_h == 0) {
    return;
  }

  for (uint32_t y = 0; y < Vr_submit_height; ++y) {
    const uint32_t src_y = (y * src_h) / Vr_submit_height;
    for (uint32_t x = 0; x < Vr_submit_width; ++x) {
      const uint32_t src_x = (x * src_w) / Vr_submit_width;
      surface.buffer[y * Vr_submit_width + x] = src_data[src_y * src_w + src_x];
    }
  }

  gl.bind_texture(GL_TEXTURE_2D, surface.texture);
  gl.tex_sub_image_2d(GL_TEXTURE_2D, 0, 0, 0, Vr_submit_width, Vr_submit_height, GL_RGBA, GL_UNSIGNED_BYTE, surface.buffer.data());
}

void VR_EnsureMenuBitmap() {
  if (!Vr_openvr_ready || Renderer_type != RENDERER_OPENGL || Vr_submit_width == 0 || Vr_submit_height == 0) {
    return;
  }

  const int desired_width = static_cast<int>(Vr_submit_width);
  const int desired_height = static_cast<int>(Vr_submit_height);
  const int desired_texture_size = VR_NextPowerOfTwo(std::max(desired_width, desired_height));

  if (Vr_menu_bitmap >= 0 && Vr_menu_width == desired_width && Vr_menu_height == desired_height &&
      Vr_menu_texture_size == desired_texture_size && Vr_menu_fbo != 0) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.gen_textures || !gl.delete_textures || !gl.bind_texture || !gl.tex_parameteri || !gl.tex_image_2d ||
      !gl.gen_framebuffers || !gl.delete_framebuffers || !gl.bind_framebuffer || !gl.framebuffer_texture_2d ||
      !gl.check_framebuffer_status) {
    if (!Vr_menu_fbo_support_missing_logged) {
      LOG_WARNING << "VR: Missing GL functions for framebuffer creation";
      Vr_menu_fbo_support_missing_logged = true;
    }
    return;
  }
  Vr_menu_fbo_support_missing_logged = false;

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

  gl.gen_textures(1, &Vr_menu_fbo_texture);
  gl.bind_texture(GL_TEXTURE_2D, Vr_menu_fbo_texture);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl.tex_parameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl.tex_image_2d(GL_TEXTURE_2D, 0, GL_RGBA8, desired_texture_size, desired_texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  gl.gen_framebuffers(1, &Vr_menu_fbo);
  gl.bind_framebuffer(GL_FRAMEBUFFER, Vr_menu_fbo);
  gl.framebuffer_texture_2d(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Vr_menu_fbo_texture, 0);
  if (gl.check_framebuffer_status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG_WARNING << "VR: Menu framebuffer is not complete!";
  }
  gl.bind_framebuffer(GL_FRAMEBUFFER, 0);

  Vr_menu_bitmap = bm_AllocBitmap(desired_texture_size, desired_texture_size, 0);
  if (Vr_menu_bitmap >= 0) {
    rend_RegisterExternalTexture(Vr_menu_bitmap, static_cast<unsigned int>(Vr_menu_fbo_texture), desired_texture_size,
                                 desired_texture_size);
    Vr_menu_texture_registered = true;
  }

  Vr_menu_width = desired_width;
  Vr_menu_height = desired_height;
  Vr_menu_texture_size = desired_texture_size;
}

bool VR_SubmitOpenVrFrame(GLuint left_texture, GLuint right_texture, bool wait_for_poses) {
  if (!Vr_openvr_ready || Vr_compositor == nullptr || left_texture == 0 || right_texture == 0) {
    return false;
  }

  if (wait_for_poses) {
    vr::TrackedDevicePose_t tracked_device_pose[vr::k_unMaxTrackedDeviceCount]{};
    const auto wait_err = Vr_compositor->WaitGetPoses(tracked_device_pose, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
    if (wait_err != vr::VRCompositorError_None) {
      LOG_WARNING.printf("OpenVR WaitGetPoses failed with error code %d", static_cast<int>(wait_err));
      return false;
    }
  }

  vr::Texture_t left = {reinterpret_cast<void *>(static_cast<uintptr_t>(left_texture)), vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
  vr::Texture_t right = {reinterpret_cast<void *>(static_cast<uintptr_t>(right_texture)), vr::TextureType_OpenGL, vr::ColorSpace_Gamma};

  const auto left_err = Vr_compositor->Submit(vr::Eye_Left, &left);
  const auto right_err = Vr_compositor->Submit(vr::Eye_Right, &right);

  if (left_err != vr::VRCompositorError_None || right_err != vr::VRCompositorError_None) {
    LOG_WARNING.printf("OpenVR Submit failed: left=%d right=%d", static_cast<int>(left_err), static_cast<int>(right_err));
  }

  Vr_compositor->PostPresentHandoff();

  return left_err == vr::VRCompositorError_None && right_err == vr::VRCompositorError_None;
}

struct VrPoseMatrices {
  bool valid = false;
  std::array<float, 16> left_eye_view = {0.0f};
  std::array<float, 16> right_eye_view = {0.0f};
  std::array<float, 16> menu_model = {0.0f};
};

struct VrRigidTransform {
  float m[4][4];
};

VrRigidTransform VR_IdentityTransform() {
  VrRigidTransform out{};
  std::memset(out.m, 0, sizeof(out.m));
  out.m[0][0] = 1.0f;
  out.m[1][1] = 1.0f;
  out.m[2][2] = 1.0f;
  out.m[3][3] = 1.0f;
  return out;
}

VrRigidTransform VR_FromOpenVr34(const vr::HmdMatrix34_t &src) {
  VrRigidTransform out = VR_IdentityTransform();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      out.m[row][col] = src.m[row][col];
    }
  }
  return out;
}

VrRigidTransform VR_MultiplyTransform(const VrRigidTransform &a, const VrRigidTransform &b) {
  VrRigidTransform out{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      out.m[row][col] = 0.0f;
      for (int k = 0; k < 4; ++k) {
        out.m[row][col] += a.m[row][k] * b.m[k][col];
      }
    }
  }
  return out;
}

VrRigidTransform VR_InvertRigidTransform(const VrRigidTransform &src) {
  VrRigidTransform out = VR_IdentityTransform();

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out.m[row][col] = src.m[col][row];
    }
  }

  out.m[0][3] = -(out.m[0][0] * src.m[0][3] + out.m[0][1] * src.m[1][3] + out.m[0][2] * src.m[2][3]);
  out.m[1][3] = -(out.m[1][0] * src.m[0][3] + out.m[1][1] * src.m[1][3] + out.m[1][2] * src.m[2][3]);
  out.m[2][3] = -(out.m[2][0] * src.m[0][3] + out.m[2][1] * src.m[1][3] + out.m[2][2] * src.m[2][3]);

  return out;
}

std::array<float, 16> VR_ToOpenGlColumnMajor(const VrRigidTransform &src) {
  std::array<float, 16> out{};
  int idx = 0;
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      out[idx++] = src.m[row][col];
    }
  }
  return out;
}

VrRigidTransform VR_Translation(float x, float y, float z) {
  VrRigidTransform out = VR_IdentityTransform();
  out.m[0][3] = x;
  out.m[1][3] = y;
  out.m[2][3] = z;
  return out;
}

bool VR_BuildMenuPoseMatrices(VrPoseMatrices &out_pose) {
  if (!Vr_openvr_ready || Vr_compositor == nullptr || Vr_system == nullptr) {
    return false;
  }

  vr::TrackedDevicePose_t tracked_device_pose[vr::k_unMaxTrackedDeviceCount]{};
  const auto wait_err = Vr_compositor->WaitGetPoses(tracked_device_pose, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
  if (wait_err != vr::VRCompositorError_None) {
    LOG_WARNING.printf("OpenVR WaitGetPoses failed with error code %d", static_cast<int>(wait_err));
    return false;
  }

  const auto &hmd_pose = tracked_device_pose[vr::k_unTrackedDeviceIndex_Hmd];
  if (!hmd_pose.bPoseIsValid) {
    return false;
  }

  const VrRigidTransform head_to_absolute = VR_FromOpenVr34(hmd_pose.mDeviceToAbsoluteTracking);
  const VrRigidTransform eye_to_head_left = VR_FromOpenVr34(Vr_system->GetEyeToHeadTransform(vr::Eye_Left));
  const VrRigidTransform eye_to_head_right = VR_FromOpenVr34(Vr_system->GetEyeToHeadTransform(vr::Eye_Right));

  const VrRigidTransform eye_left_to_absolute = VR_MultiplyTransform(head_to_absolute, eye_to_head_left);
  const VrRigidTransform eye_right_to_absolute = VR_MultiplyTransform(head_to_absolute, eye_to_head_right);

  const VrRigidTransform left_eye_view = VR_InvertRigidTransform(eye_left_to_absolute);
  const VrRigidTransform right_eye_view = VR_InvertRigidTransform(eye_right_to_absolute);

  // Head-anchored cinema screen in front of the user, avoiding direct eye-buffer UI compositing.
  const VrRigidTransform menu_model = VR_MultiplyTransform(head_to_absolute, VR_Translation(0.0f, 0.0f, -1.25f));

  out_pose.left_eye_view = VR_ToOpenGlColumnMajor(left_eye_view);
  out_pose.right_eye_view = VR_ToOpenGlColumnMajor(right_eye_view);
  out_pose.menu_model = VR_ToOpenGlColumnMajor(menu_model);
  out_pose.valid = true;
  return true;
}

bool VR_RenderCurvedMenuToSurface(const VrSubmitSurface &surface, const std::array<float, 16> &eye_view,
                                  const std::array<float, 16> &menu_model) {
  if (surface.texture == 0 || Vr_menu_fbo_texture == 0 || Vr_submit_width == 0 || Vr_submit_height == 0) {
    return false;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.bind_framebuffer || !gl.framebuffer_texture_2d || !gl.viewport || !gl.check_framebuffer_status ||
      !gl.clear_color || !gl.clear || !gl.disable || !gl.enable || !gl.matrix_mode || !gl.push_matrix ||
      !gl.pop_matrix || !gl.load_identity || !gl.load_matrixf || !gl.mult_matrixf || !gl.frustum || !gl.begin || !gl.end ||
      !gl.tex_coord2f || !gl.vertex3f || !gl.bind_texture) {
    return false;
  }

  gl.bind_framebuffer(GL_FRAMEBUFFER, Vr_submit_fbo);
  gl.framebuffer_texture_2d(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, surface.texture, 0);
  if (gl.check_framebuffer_status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    return false;
  }

  gl.viewport(0, 0, static_cast<GLsizei>(Vr_submit_width), static_cast<GLsizei>(Vr_submit_height));
  gl.clear_color(0.f, 0.f, 0.f, 1.f);
  gl.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  gl.disable(GL_DEPTH_TEST);
  gl.disable(GL_CULL_FACE);
  gl.enable(GL_TEXTURE_2D);

  gl.matrix_mode(GL_PROJECTION);
  gl.push_matrix();
  gl.load_identity();
  const float near_plane = 0.1f;
  const float far_plane = 10.0f;
  const float top = near_plane;
  const float right = top * (static_cast<float>(Vr_submit_width) / static_cast<float>(Vr_submit_height));
  gl.frustum(-right, right, -top, top, near_plane, far_plane);

  gl.matrix_mode(GL_MODELVIEW);
  gl.push_matrix();
  gl.load_matrixf(eye_view.data());
  gl.mult_matrixf(menu_model.data());

  const float radius = 1.15f;
  const float arc_half_angle = 0.85f;
  const float screen_height = 1.15f;
  const int segments = 64;

  gl.bind_texture(GL_TEXTURE_2D, Vr_menu_fbo_texture);
  gl.begin(GL_QUAD_STRIP);
  for (int i = 0; i <= segments; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(segments);
    const float angle = (t * 2.0f - 1.0f) * arc_half_angle;
    const float x = std::sin(angle) * radius;
    const float z = (std::cos(angle) * radius) - radius;

    gl.tex_coord2f(t, 1.0f);
    gl.vertex3f(x, screen_height * 0.5f, z);
    gl.tex_coord2f(t, 0.0f);
    gl.vertex3f(x, -screen_height * 0.5f, z);
  }
  gl.end();

  gl.pop_matrix();
  gl.matrix_mode(GL_PROJECTION);
  gl.pop_matrix();
  gl.matrix_mode(GL_MODELVIEW);
  return true;
}

bool VR_CopyMenuToSubmitSurface(const VrSubmitSurface &surface) {
  if (surface.texture == 0 || Vr_menu_fbo == 0 || Vr_submit_width == 0 || Vr_submit_height == 0) {
    return false;
  }

  auto &gl = VR_GetGlFns();
  if (!gl.bind_framebuffer || !gl.bind_texture || !gl.copy_tex_sub_image_2d) {
    return false;
  }

  gl.bind_framebuffer(GL_FRAMEBUFFER, Vr_menu_fbo);
  gl.bind_texture(GL_TEXTURE_2D, surface.texture);
  gl.copy_tex_sub_image_2d(GL_TEXTURE_2D, 0, 0, 0, 0, 0, Vr_submit_width, Vr_submit_height);
  return true;
}

void VR_InitStereoFrustums() {
  if (!Vr_openvr_ready || Vr_system == nullptr) {
    return;
  }

  g3StereoFrustum left_frustum{};
  g3StereoFrustum right_frustum{};

  float l = 0.0f, r = 0.0f, t = 0.0f, b = 0.0f;
  Vr_system->GetProjectionRaw(vr::Eye_Left, &l, &r, &t, &b);
  left_frustum.left = l;
  left_frustum.right = r;
  left_frustum.top = -t;
  left_frustum.bottom = -b;

  Vr_system->GetProjectionRaw(vr::Eye_Right, &l, &r, &t, &b);
  right_frustum.left = l;
  right_frustum.right = r;
  right_frustum.top = -t;
  right_frustum.bottom = -b;

  const auto eye_to_head_left = Vr_system->GetEyeToHeadTransform(vr::Eye_Left);
  const auto eye_to_head_right = Vr_system->GetEyeToHeadTransform(vr::Eye_Right);
  Vr_eye_separation = std::fabs(eye_to_head_right.m[0][3] - eye_to_head_left.m[0][3]);

  g3_SetStereoFrustum(&left_frustum, &right_frustum);
}

} // namespace

void VR_BeginMenuFramebufferRender() {
  if (Vr_menu_fbo == 0) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (gl.bind_framebuffer) {
    if (gl.get_integerv) {
      gl.get_integerv(GL_VIEWPORT, Vr_saved_menu_viewport.data());
      Vr_saved_menu_viewport_valid = true;
    }

    gl.bind_framebuffer(GL_FRAMEBUFFER, Vr_menu_fbo);

    if (gl.viewport && Vr_menu_width > 0 && Vr_menu_height > 0) {
      gl.viewport(0, 0, Vr_menu_width, Vr_menu_height);
    }
  }
}

void VR_EndMenuFramebufferRender() {
  auto &gl = VR_GetGlFns();
  if (gl.bind_framebuffer) {
    gl.bind_framebuffer(GL_FRAMEBUFFER, 0);

    if (gl.viewport && Vr_saved_menu_viewport_valid) {
      gl.viewport(Vr_saved_menu_viewport[0], Vr_saved_menu_viewport[1],
                  Vr_saved_menu_viewport[2], Vr_saved_menu_viewport[3]);
    }

    Vr_saved_menu_viewport_valid = false;
  }
}

void VR_InitFromCommandLine() {
  Vr_enabled = FindArg("-vr") != 0;
  Vr_render_mode = VrRenderMode::Stereo;
  if (!Vr_enabled) {
    return;
  }

  vr::EVRInitError init_error = vr::VRInitError_None;
  Vr_system = vr::VR_Init(&init_error, vr::VRApplication_Scene);
  if (init_error != vr::VRInitError_None || Vr_system == nullptr) {
    LOG_WARNING.printf("OpenVR init failed: %s", vr::VR_GetVRInitErrorAsEnglishDescription(init_error));
    Vr_enabled = false;
    return;
  }

  Vr_compositor = vr::VRCompositor();
  if (Vr_compositor == nullptr) {
    LOG_WARNING << "OpenVR compositor unavailable; VR disabled.";
    vr::VR_Shutdown();
    Vr_system = nullptr;
    Vr_enabled = false;
    return;
  }

  Vr_system->GetRecommendedRenderTargetSize(&Vr_submit_width, &Vr_submit_height);
  if (Vr_submit_width == 0 || Vr_submit_height == 0) {
    Vr_submit_width = 2000;
    Vr_submit_height = 2000;
  }

  Vr_openvr_ready = true;
  VR_InitStereoFrustums();
  LOG_INFO.printf("OpenVR enabled via -vr. Render target %ux%u.", Vr_submit_width, Vr_submit_height);
}

bool VR_IsEnabled() { return Vr_enabled; }
VrRenderMode VR_GetRenderMode() { return Vr_render_mode; }
bool VR_IsStereoRendering() { return Vr_enabled && Vr_render_mode == VrRenderMode::Stereo; }
float VR_GetStereoEyeSeparation() { return Vr_enabled ? Vr_eye_separation : 0.0f; }

void VR_RenderMenuFrame() {
  if (!Vr_enabled || !Vr_openvr_ready || Renderer_type != RENDERER_OPENGL) {
    return;
  }

  VR_EnsureMenuBitmap();
  if (Vr_menu_fbo_texture == 0) {
    return;
  }

  if (!VR_EnsureSubmitSurface(Vr_submit_left) || !VR_EnsureSubmitSurface(Vr_submit_right)) {
    return;
  }

  auto &gl = VR_GetGlFns();
  if (Vr_submit_fbo == 0 && gl.gen_framebuffers) {
    gl.gen_framebuffers(1, &Vr_submit_fbo);
  }

  bool left_curved = false;
  bool right_curved = false;
  VrPoseMatrices pose_matrices;

  const bool have_pose = VR_BuildMenuPoseMatrices(pose_matrices);

  if (Vr_submit_fbo != 0 && have_pose) {
    left_curved = VR_RenderCurvedMenuToSurface(Vr_submit_left, pose_matrices.left_eye_view, pose_matrices.menu_model);
    right_curved = VR_RenderCurvedMenuToSurface(Vr_submit_right, pose_matrices.right_eye_view, pose_matrices.menu_model);
  }

  if (!left_curved || !right_curved) {
    if (!Vr_curved_menu_unavailable_logged) {
      LOG_WARNING << "VR: Curved menu render path unavailable; falling back to flat menu submit.";
      Vr_curved_menu_unavailable_logged = true;
    }

    if (!VR_CopyMenuToSubmitSurface(Vr_submit_left) || !VR_CopyMenuToSubmitSurface(Vr_submit_right)) {
      if (gl.bind_framebuffer) {
        gl.bind_framebuffer(GL_FRAMEBUFFER, 0);
      }
      return;
    }
  }

  if (gl.bind_framebuffer) {
    gl.bind_framebuffer(GL_FRAMEBUFFER, 0);
  }

  VR_SubmitOpenVrFrame(Vr_submit_left.texture, Vr_submit_right.texture, false);
}

void VR_SubmitStereoFrame(const NewBitmap &left, const NewBitmap &right) {
  if (!VR_IsStereoRendering() || !Vr_openvr_ready || Renderer_type != RENDERER_OPENGL) {
    return;
  }

  if (!VR_EnsureSubmitSurface(Vr_submit_left) || !VR_EnsureSubmitSurface(Vr_submit_right)) {
    return;
  }

  VR_UpdateSubmitSurface(left, Vr_submit_left);
  VR_UpdateSubmitSurface(right, Vr_submit_right);
  VR_SubmitOpenVrFrame(Vr_submit_left.texture, Vr_submit_right.texture, true);
}

void VR_ResetGraphicsResources() {
  auto &gl = VR_GetGlFns();
  VR_DeleteSubmitSurface(Vr_submit_left);
  VR_DeleteSubmitSurface(Vr_submit_right);

  if (Vr_menu_fbo != 0 && gl.delete_framebuffers) gl.delete_framebuffers(1, &Vr_menu_fbo);
  if (Vr_submit_fbo != 0 && gl.delete_framebuffers) gl.delete_framebuffers(1, &Vr_submit_fbo);
  if (Vr_menu_fbo_texture != 0 && gl.delete_textures) gl.delete_textures(1, &Vr_menu_fbo_texture);
  Vr_menu_fbo = 0;
  Vr_submit_fbo = 0;
  Vr_menu_fbo_texture = 0;

  if (Vr_menu_bitmap >= 0) {
    if (Vr_menu_texture_registered) {
      rend_UnregisterExternalTexture(Vr_menu_bitmap);
      Vr_menu_texture_registered = false;
    }
    bm_FreeBitmap(Vr_menu_bitmap);
    Vr_menu_bitmap = -1;
  }
}

void VR_Shutdown() {
  VR_ResetGraphicsResources();

  if (Vr_system != nullptr) {
    vr::VR_Shutdown();
  }
  Vr_system = nullptr;
  Vr_compositor = nullptr;
  Vr_openvr_ready = false;
  Vr_enabled = false;
}
