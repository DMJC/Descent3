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

#include "log.h"

#if defined(ENABLE_OPENVR)

#include "3d.h"
#include "vecmat.h"

#include "../renderer/HardwareInternal.h"
#include "../renderer/dyna_gl.h"

#include <SDL3/SDL_opengl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <openvr.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

constexpr float kMetersToGameUnits = 1.0f;
constexpr float kNearClip = 0.05f;
constexpr float kFarClip = 500.0f;

struct EyeFramebuffer {
  GLuint framebuffer = 0;
  GLuint color = 0;
  GLuint depth = 0;
};

bool requested_ = false;
bool initialized_ = false;
bool init_attempted_ = false;

vr::IVRSystem *vr_system_ = nullptr;
EyeFramebuffer eye_targets_[2];

glm::quat initial_orientation_{};
glm::vec3 initial_position_{};
bool have_initial_pose_ = false;

glm::quat base_orientation_quat_{};
glm::mat3 base_orientation_matrix_{};
glm::vec3 base_position_{};

glm::quat relative_orientation_{};
glm::vec3 head_offset_{};

int render_width_ = 0;
int render_height_ = 0;

bool frame_active_ = false;

std::array<d3vr::EyeRenderData, 2> eye_data_{};
std::array<bool, 2> eye_data_valid_{};
const d3vr::EyeRenderData *current_eye_data_ = nullptr;
d3vr::EyeRenderData scratch_eye_data_{};

GLint previous_framebuffer_ = 0;
GLint previous_viewport_[4] = {0, 0, 0, 0};
GLint previous_scissor_[4] = {0, 0, 0, 0};
int bound_eye_index_ = -1;

GLint mirror_framebuffer_ = 0;
GLint mirror_viewport_[4] = {0, 0, 0, 0};
GLint mirror_scissor_[4] = {0, 0, 0, 0};

bool warned_missing_compositor_ = false;

glm::vec3 ToGlm(const vector &v) {
  return glm::vec3(v.x(), v.y(), v.z());
}

vector FromGlm(const glm::vec3 &v) {
  vector out{};
  out.x() = v.x;
  out.y() = v.y;
  out.z() = v.z;
  return out;
}

glm::mat3 MatrixToGlm(const matrix &m) {
  glm::mat3 mat{};
  mat[0][0] = m.rvec.x();
  mat[0][1] = m.uvec.x();
  mat[0][2] = m.fvec.x();
  mat[1][0] = m.rvec.y();
  mat[1][1] = m.uvec.y();
  mat[1][2] = m.fvec.y();
  mat[2][0] = m.rvec.z();
  mat[2][1] = m.uvec.z();
  mat[2][2] = m.fvec.z();
  return mat;
}

matrix GlmToMatrix(const glm::mat3 &m) {
  matrix out{};
  out.rvec.x() = m[0][0];
  out.rvec.y() = m[1][0];
  out.rvec.z() = m[2][0];
  out.uvec.x() = m[0][1];
  out.uvec.y() = m[1][1];
  out.uvec.z() = m[2][1];
  out.fvec.x() = m[0][2];
  out.fvec.y() = m[1][2];
  out.fvec.z() = m[2][2];
  return out;
}

glm::quat ConvertRotation(const vr::HmdMatrix34_t &m) {
  glm::vec3 r(m.m[0][0], m.m[1][0], m.m[2][0]);
  glm::vec3 u(m.m[0][1], m.m[1][1], m.m[2][1]);
  glm::vec3 f(-m.m[0][2], -m.m[1][2], -m.m[2][2]);

  glm::mat3 rot{};
  rot[0] = r;
  rot[1] = u;
  rot[2] = f;

  return glm::quat_cast(rot);
}

glm::vec3 ConvertTranslation(const vr::HmdMatrix34_t &m) {
  glm::vec3 translation(m.m[0][3], m.m[1][3], m.m[2][3]);
  translation.z = -translation.z;
  return translation;
}

glm::mat4 ConvertProjection(const vr::HmdMatrix44_t &m) {
  return glm::mat4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1],
                   m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]);
}

void FillProjectionMatrix(const glm::mat4 &mat, float out[4][4]) {
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      out[row][col] = mat[col][row];
    }
  }
}

void DestroyFramebuffer(EyeFramebuffer &buffer) {
  if (buffer.color != 0) {
    dglDeleteTextures(1, &buffer.color);
    buffer.color = 0;
  }
  if (buffer.depth != 0) {
    dglDeleteRenderbuffers(1, &buffer.depth);
    buffer.depth = 0;
  }
  if (buffer.framebuffer != 0) {
    dglDeleteFramebuffers(1, &buffer.framebuffer);
    buffer.framebuffer = 0;
  }
}

bool CreateFramebuffer(EyeFramebuffer &buffer) {
  GLint prev_fbo = 0;
  dglGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

  dglGenFramebuffers(1, &buffer.framebuffer);
  dglBindFramebuffer(GL_FRAMEBUFFER, buffer.framebuffer);

  dglGenTextures(1, &buffer.color);
  dglBindTexture(GL_TEXTURE_2D, buffer.color);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  dglTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, render_width_, render_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  dglFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buffer.color, 0);

  dglGenRenderbuffers(1, &buffer.depth);
  dglBindRenderbuffer(GL_RENDERBUFFER, buffer.depth);
  dglRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, render_width_, render_height_);
  dglFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer.depth);

  bool complete = dglCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  if (!complete) {
    LOG_ERROR << "OpenVR: Failed to create framebuffer for eye.";
    DestroyFramebuffer(buffer);
  }

  dglBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
  return complete;
}

} // namespace

namespace d3vr {

void SetRequested(bool requested) { requested_ = requested; }

bool IsRequested() { return requested_; }

bool Initialize() {
  if (!requested_) {
    return false;
  }

  if (initialized_) {
    return true;
  }

  vr::EVRInitError init_error = vr::VRInitError_None;
  vr_system_ = vr::VR_Init(&init_error, vr::VRApplication_Scene);
  if (init_error != vr::VRInitError_None) {
    LOG_ERROR << "OpenVR initialization failed: " << vr::VR_GetVRInitErrorAsEnglishDescription(init_error);
    vr_system_ = nullptr;
    init_attempted_ = true;
    return false;
  }

  if (vr::VRCompositor() == nullptr) {
    if (!warned_missing_compositor_) {
      LOG_ERROR << "OpenVR: VR compositor is unavailable.";
      warned_missing_compositor_ = true;
    }
    vr::VR_Shutdown();
    vr_system_ = nullptr;
    init_attempted_ = true;
    return false;
  }

  uint32_t width = 0;
  uint32_t height = 0;
  vr_system_->GetRecommendedRenderTargetSize(&width, &height);
  render_width_ = static_cast<int>(width);
  render_height_ = static_cast<int>(height);

  if (render_width_ <= 0 || render_height_ <= 0) {
    LOG_ERROR << "OpenVR: Invalid recommended render target size.";
    vr::VR_Shutdown();
    vr_system_ = nullptr;
    init_attempted_ = true;
    return false;
  }

  vr::VRCompositor()->SetTrackingSpace(vr::TrackingUniverseStanding);

  if (!CreateFramebuffer(eye_targets_[0]) || !CreateFramebuffer(eye_targets_[1])) {
    vr::VR_Shutdown();
    vr_system_ = nullptr;
    init_attempted_ = true;
    return false;
  }

  have_initial_pose_ = false;
  frame_active_ = false;
  mirror_framebuffer_ = 0;
  bound_eye_index_ = -1;

  initialized_ = true;
  init_attempted_ = true;

  LOG_INFO.printf("OpenVR initialized with render target %dx%d", render_width_, render_height_);
  return true;
}

void Shutdown() {
  if (!initialized_) {
    requested_ = false;
    return;
  }

  DestroyFramebuffer(eye_targets_[0]);
  DestroyFramebuffer(eye_targets_[1]);

  vr::VR_Shutdown();
  vr_system_ = nullptr;

  initialized_ = false;
  have_initial_pose_ = false;
  frame_active_ = false;
  mirror_framebuffer_ = 0;
  bound_eye_index_ = -1;
  eye_data_valid_.fill(false);
  current_eye_data_ = nullptr;
}

bool IsEnabled() { return requested_ && initialized_; }

bool BeginFrame(const vector &basePos, const matrix &baseOrient) {
  if (!IsEnabled()) {
    return false;
  }

  auto *compositor = vr::VRCompositor();
  if (!compositor) {
    if (!warned_missing_compositor_) {
      LOG_ERROR << "OpenVR: compositor became unavailable.";
      warned_missing_compositor_ = true;
    }
    return false;
  }

  std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> poses{};
  auto pose_error = compositor->WaitGetPoses(poses.data(), static_cast<uint32_t>(poses.size()), nullptr, 0);
  if (pose_error != vr::VRCompositorError_None) {
    LOG_WARNING << "OpenVR: WaitGetPoses failed with error " << pose_error;
    return false;
  }

  const vr::TrackedDevicePose_t &hmd_pose = poses[vr::k_unTrackedDeviceIndex_Hmd];
  if (!hmd_pose.bPoseIsValid) {
    LOG_WARNING << "OpenVR: HMD pose is invalid.";
    return false;
  }

  glm::quat current_orientation = ConvertRotation(hmd_pose.mDeviceToAbsoluteTracking);
  glm::vec3 current_position = ConvertTranslation(hmd_pose.mDeviceToAbsoluteTracking);

  if (!have_initial_pose_) {
    initial_orientation_ = current_orientation;
    initial_position_ = current_position;
    have_initial_pose_ = true;
  }

  relative_orientation_ = current_orientation * glm::inverse(initial_orientation_);
  relative_orientation_ = glm::normalize(relative_orientation_);
  head_offset_ = (current_position - initial_position_) * kMetersToGameUnits;

  base_orientation_matrix_ = MatrixToGlm(baseOrient);
  base_orientation_quat_ = glm::quat_cast(base_orientation_matrix_);
  base_position_ = ToGlm(basePos);

  frame_active_ = true;
  mirror_framebuffer_ = 0;
  bound_eye_index_ = -1;
  eye_data_valid_.fill(false);
  current_eye_data_ = nullptr;

  return true;
}

bool GetEyeData(Eye eye, EyeRenderData &out) {
  if (!frame_active_ || !IsEnabled()) {
    return false;
  }

  int index = static_cast<int>(eye);
  vr::HmdMatrix34_t eye_to_head = vr_system_->GetEyeToHeadTransform(static_cast<vr::Hmd_Eye>(index));

  glm::vec3 eye_offset = -ConvertTranslation(eye_to_head);
  glm::vec3 rotated_eye_offset = glm::mat3_cast(relative_orientation_) * eye_offset;
  glm::vec3 local_offset = head_offset_ + rotated_eye_offset;

  glm::vec3 world_offset = base_orientation_matrix_ * local_offset;
  glm::vec3 world_position = base_position_ + world_offset;

  glm::quat final_orientation = glm::normalize(base_orientation_quat_ * relative_orientation_);
  glm::mat3 final_matrix = glm::mat3_cast(final_orientation);

  vr::HmdMatrix44_t projection = vr_system_->GetProjectionMatrix(static_cast<vr::Hmd_Eye>(index), kNearClip, kFarClip);
  glm::mat4 projection_matrix = ConvertProjection(projection);

  float tan_half_fov = 1.0f / projection_matrix[1][1];
  float zoom = (4.0f / 3.0f) * tan_half_fov;

  EyeRenderData data{};
  data.position = FromGlm(world_position);
  data.orientation = GlmToMatrix(final_matrix);
  data.zoom = zoom;
  data.renderWidth = render_width_;
  data.renderHeight = render_height_;
  FillProjectionMatrix(projection_matrix, data.projection);

  eye_data_[index] = data;
  eye_data_valid_[index] = true;
  out = data;
  return true;
}

bool BindEye(Eye eye, const EyeRenderData &data) {
  if (!frame_active_ || !IsEnabled()) {
    return false;
  }

  int index = static_cast<int>(eye);
  if (eye_targets_[index].framebuffer == 0) {
    return false;
  }

  dglGetIntegerv(GL_FRAMEBUFFER_BINDING, &previous_framebuffer_);
  dglGetIntegerv(GL_VIEWPORT, previous_viewport_);
  dglGetIntegerv(GL_SCISSOR_BOX, previous_scissor_);

  if (mirror_framebuffer_ == 0) {
    mirror_framebuffer_ = previous_framebuffer_;
    mirror_viewport_[0] = previous_viewport_[0];
    mirror_viewport_[1] = previous_viewport_[1];
    mirror_viewport_[2] = previous_viewport_[2];
    mirror_viewport_[3] = previous_viewport_[3];
    mirror_scissor_[0] = previous_scissor_[0];
    mirror_scissor_[1] = previous_scissor_[1];
    mirror_scissor_[2] = previous_scissor_[2];
    mirror_scissor_[3] = previous_scissor_[3];
  }

  bound_eye_index_ = index;

  dglBindFramebuffer(GL_FRAMEBUFFER, eye_targets_[index].framebuffer);
  dglViewport(0, 0, data.renderWidth, data.renderHeight);
  dglScissor(0, 0, data.renderWidth, data.renderHeight);
  dglClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  return true;
}

void UnbindEye() {
  if (bound_eye_index_ < 0) {
    return;
  }

  dglBindFramebuffer(GL_FRAMEBUFFER, previous_framebuffer_);
  dglViewport(previous_viewport_[0], previous_viewport_[1], previous_viewport_[2], previous_viewport_[3]);
  dglScissor(previous_scissor_[0], previous_scissor_[1], previous_scissor_[2], previous_scissor_[3]);

  bound_eye_index_ = -1;
}

void SetCurrentEyeData(const EyeRenderData *data) {
  if (data == nullptr) {
    current_eye_data_ = nullptr;
    return;
  }

  for (std::size_t i = 0; i < eye_data_.size(); ++i) {
    if (eye_data_valid_[i] && data == &eye_data_[i]) {
      current_eye_data_ = data;
      return;
    }
  }

  scratch_eye_data_ = *data;
  current_eye_data_ = &scratch_eye_data_;
}

void ClearCurrentEyeData() { current_eye_data_ = nullptr; }

void OnGameRenderWorldStart() {
  if (!current_eye_data_) {
    return;
  }

  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      gTransformProjection[row][col] = current_eye_data_->projection[row][col];
    }
  }
  g3_UpdateFullTransform();
}

void SubmitFrame() {
  if (!frame_active_ || !IsEnabled()) {
    return;
  }

  vr::Texture_t left_texture{reinterpret_cast<void *>(static_cast<uintptr_t>(eye_targets_[0].color)), vr::TextureType_OpenGL,
                              vr::ColorSpace_Gamma};
  vr::Texture_t right_texture{reinterpret_cast<void *>(static_cast<uintptr_t>(eye_targets_[1].color)), vr::TextureType_OpenGL,
                               vr::ColorSpace_Gamma};

  vr::VRCompositor()->Submit(vr::Eye_Left, &left_texture);
  vr::VRCompositor()->Submit(vr::Eye_Right, &right_texture);
  vr::VRCompositor()->PostPresentHandoff();

  frame_active_ = false;
  eye_data_valid_.fill(false);
}

void MirrorToBackbuffer() {
  if (mirror_framebuffer_ == 0 || eye_targets_[0].framebuffer == 0) {
    return;
  }

  dglBindFramebuffer(GL_READ_FRAMEBUFFER, eye_targets_[0].framebuffer);
  dglBindFramebuffer(GL_DRAW_FRAMEBUFFER, mirror_framebuffer_);
  dglBlitFramebuffer(0, 0, render_width_, render_height_, mirror_viewport_[0], mirror_viewport_[1],
                     mirror_viewport_[0] + mirror_viewport_[2], mirror_viewport_[1] + mirror_viewport_[3],
                     GL_COLOR_BUFFER_BIT, GL_LINEAR);
  dglBindFramebuffer(GL_FRAMEBUFFER, mirror_framebuffer_);
  dglViewport(mirror_viewport_[0], mirror_viewport_[1], mirror_viewport_[2], mirror_viewport_[3]);
  dglScissor(mirror_scissor_[0], mirror_scissor_[1], mirror_scissor_[2], mirror_scissor_[3]);
}

void CancelFrame() {
  if (bound_eye_index_ >= 0) {
    dglBindFramebuffer(GL_FRAMEBUFFER, previous_framebuffer_);
    dglViewport(previous_viewport_[0], previous_viewport_[1], previous_viewport_[2], previous_viewport_[3]);
    dglScissor(previous_scissor_[0], previous_scissor_[1], previous_scissor_[2], previous_scissor_[3]);
    bound_eye_index_ = -1;
  }

  frame_active_ = false;
  mirror_framebuffer_ = 0;
  current_eye_data_ = nullptr;
  eye_data_valid_.fill(false);
}

} // namespace d3vr

#else // !defined(ENABLE_OPENVR)

namespace {
bool requested_stub_ = false;
bool warned_stub_ = false;
} // namespace

namespace d3vr {

void SetRequested(bool requested) {
  requested_stub_ = requested;
  if (requested && !warned_stub_) {
    LOG_WARNING << "OpenVR support was requested but is not available in this build.";
    warned_stub_ = true;
  }
}

bool IsRequested() { return requested_stub_; }

bool Initialize() { return false; }

void Shutdown() { requested_stub_ = false; }

bool IsEnabled() { return false; }

bool BeginFrame(const vector &, const matrix &) { return false; }

bool GetEyeData(Eye, EyeRenderData &) { return false; }

bool BindEye(Eye, const EyeRenderData &) { return false; }

void UnbindEye() {}

void SetCurrentEyeData(const EyeRenderData *) {}

void ClearCurrentEyeData() {}

void OnGameRenderWorldStart() {}

void SubmitFrame() {}

void MirrorToBackbuffer() {}

void CancelFrame() {}

} // namespace d3vr

#endif // defined(ENABLE_OPENVR)
