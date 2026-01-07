#include "vr/VRSystem.h"

#include <cmath>
#include <vector>

#include <GL/gl.h>

#include "game.h"
#include "log.h"
#include "timer.h"

namespace {
constexpr float kCinemaRadius = 2.0f;
constexpr float kCinemaHeight = 1.2f;
constexpr float kCinemaArcDegrees = 90.0f;
constexpr float kCinemaDistance = 1.5f;
constexpr int kCinemaSegments = 48;
constexpr float kCinemaRecenterSpeed = 1.5f;
constexpr float kNearZ = 0.1f;
constexpr float kFarZ = 1000.0f;
constexpr float kPi = 3.14159265358979323846f;

float WrapAngle(float angle) {
  while (angle > kPi)
    angle -= kPi * 2.0f;
  while (angle < -kPi)
    angle += kPi * 2.0f;
  return angle;
}
} // namespace

VRSystem &VRSystem::Get() {
  static VRSystem instance;
  return instance;
}

VRSystem::VRSystem() {
  base_orientation_ = Identity_matrix;
  base_position_ = Zero_vector;
  hmd_orientation_ = Identity_matrix;
  hmd_position_ = Zero_vector;
}

VRSystem::~VRSystem() { Shutdown(); }

bool VRSystem::Initialize(bool enable_vr) {
  if (!enable_vr) {
    enabled_ = false;
    return false;
  }

  if (enabled_) {
    return true;
  }

  if (!InitializeOpenVR()) {
    enabled_ = false;
    return false;
  }

  CreateEyeTargets();
  CreateCinemaTarget();
  BuildCinemaMesh();

  enabled_ = true;
  LOG_INFO << "OpenVR initialized.";
  return true;
}

void VRSystem::Shutdown() {
  if (!enabled_) {
    ShutdownOpenVR();
    return;
  }

  DestroyTargets();
  ShutdownOpenVR();

  enabled_ = false;
}

bool VRSystem::Enabled() const { return enabled_; }

bool VRSystem::InCinemaMode() const { return cinema_mode_; }

bool VRSystem::InitializeOpenVR() {
  vr::EVRInitError error = vr::VRInitError_None;
  hmd_ = vr::VR_Init(&error, vr::VRApplication_Scene);
  if (error != vr::VRInitError_None) {
    hmd_ = nullptr;
    LOG_WARNING << "OpenVR init failed: " << vr::VR_GetVRInitErrorAsEnglishDescription(error);
    return false;
  }

  compositor_ = vr::VRCompositor();
  if (!compositor_) {
    LOG_WARNING << "OpenVR compositor unavailable.";
    vr::VR_Shutdown();
    hmd_ = nullptr;
    return false;
  }

  uint32_t width = 0;
  uint32_t height = 0;
  hmd_->GetRecommendedRenderTargetSize(&width, &height);
  eye_targets_[0].width = static_cast<int>(width);
  eye_targets_[0].height = static_cast<int>(height);
  eye_targets_[1].width = static_cast<int>(width);
  eye_targets_[1].height = static_cast<int>(height);

  for (int eye = 0; eye < 2; ++eye) {
    float left = 0.0f;
    float right = 0.0f;
    float top = 0.0f;
    float bottom = 0.0f;
    hmd_->GetProjectionRaw(static_cast<vr::Hmd_Eye>(eye), &left, &right, &top, &bottom);
    float vertical_fov = std::max(std::abs(top), std::abs(bottom));
    eye_zoom_[eye] = vertical_fov * 4.0f / 3.0f;
  }

  return true;
}

void VRSystem::ShutdownOpenVR() {
  if (hmd_) {
    vr::VR_Shutdown();
    hmd_ = nullptr;
    compositor_ = nullptr;
  }
}

void VRSystem::CreateEyeTargets() {
  for (int eye = 0; eye < 2; ++eye) {
    EyeRenderTarget &target = eye_targets_[eye];
    if (target.fbo != 0) {
      continue;
    }

    dglGenFramebuffers(1, &target.fbo);
    dglBindFramebuffer(GL_FRAMEBUFFER, target.fbo);

    dglGenTextures(1, &target.color_tex);
    dglBindTexture(GL_TEXTURE_2D, target.color_tex);
    dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    dglTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, target.width, target.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    dglGenRenderbuffers(1, &target.depth_rb);
    dglBindRenderbuffer(GL_RENDERBUFFER, target.depth_rb);
    dglRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, target.width, target.height);

    dglFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target.color_tex, 0);
    dglFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, target.depth_rb);

    if (dglCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      LOG_WARNING << "OpenVR eye framebuffer incomplete.";
    }
  }

  dglBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void VRSystem::CreateCinemaTarget() {
  if (cinema_fbo_ != 0) {
    return;
  }

  dglGenFramebuffers(1, &cinema_fbo_);
  dglBindFramebuffer(GL_FRAMEBUFFER, cinema_fbo_);

  dglGenTextures(1, &cinema_color_tex_);
  dglBindTexture(GL_TEXTURE_2D, cinema_color_tex_);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  dglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  dglTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cinema_width_, cinema_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

  dglGenRenderbuffers(1, &cinema_depth_rb_);
  dglBindRenderbuffer(GL_RENDERBUFFER, cinema_depth_rb_);
  dglRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, cinema_width_, cinema_height_);

  dglFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cinema_color_tex_, 0);
  dglFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, cinema_depth_rb_);

  if (dglCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG_WARNING << "OpenVR cinema framebuffer incomplete.";
  }

  dglBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void VRSystem::DestroyTargets() {
  for (auto &target : eye_targets_) {
    if (target.depth_rb) {
      dglDeleteRenderbuffers(1, &target.depth_rb);
      target.depth_rb = 0;
    }
    if (target.color_tex) {
      dglDeleteTextures(1, &target.color_tex);
      target.color_tex = 0;
    }
    if (target.fbo) {
      dglDeleteFramebuffers(1, &target.fbo);
      target.fbo = 0;
    }
  }

  if (cinema_depth_rb_) {
    dglDeleteRenderbuffers(1, &cinema_depth_rb_);
    cinema_depth_rb_ = 0;
  }
  if (cinema_color_tex_) {
    dglDeleteTextures(1, &cinema_color_tex_);
    cinema_color_tex_ = 0;
  }
  if (cinema_fbo_) {
    dglDeleteFramebuffers(1, &cinema_fbo_);
    cinema_fbo_ = 0;
  }
}

void VRSystem::BuildCinemaMesh() {
  cinema_vertices_.clear();
  cinema_vertices_.reserve((kCinemaSegments + 1) * 2);

  float arc_rad = kCinemaArcDegrees * kPi / 180.0f;
  float start_angle = -arc_rad * 0.5f;
  float step = arc_rad / static_cast<float>(kCinemaSegments);
  float center_z = -(kCinemaDistance + kCinemaRadius);

  for (int i = 0; i <= kCinemaSegments; ++i) {
    float angle = start_angle + step * static_cast<float>(i);
    float x = std::sin(angle) * kCinemaRadius;
    float z = center_z + std::cos(angle) * kCinemaRadius;
    float u = static_cast<float>(i) / static_cast<float>(kCinemaSegments);

    cinema_vertices_.push_back({x, -kCinemaHeight * 0.5f, z, u, 0.0f});
    cinema_vertices_.push_back({x, kCinemaHeight * 0.5f, z, u, 1.0f});
  }
}

void VRSystem::BeginFrame() {
  if (!enabled_ || !compositor_) {
    return;
  }

  compositor_->WaitGetPoses(tracked_poses_.data(), static_cast<uint32_t>(tracked_poses_.size()), nullptr, 0);

  if (tracked_poses_[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
    const vr::HmdMatrix34_t &pose = tracked_poses_[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking;
    hmd_orientation_ = HmdMatrixToMatrix(pose);
    hmd_position_ = HmdMatrixGetPosition(pose);
  }

  UpdateEyeMatrices();
  UpdateCinemaPointer();

  double now = timer_GetTime();
  if (last_update_time_ == 0.0) {
    last_update_time_ = now;
  }
  float dt = static_cast<float>(now - last_update_time_);
  last_update_time_ = now;
  UpdateScreenRecenter(dt);
}

void VRSystem::RenderEye(Eye eye, const std::function<void()> &renderScene) {
  if (!enabled_) {
    renderScene();
    return;
  }

  current_eye_ = eye;
  rendering_eye_ = true;
  rendering_cinema_ = false;

  const EyeRenderTarget &target = eye_targets_[static_cast<int>(eye)];
  dglBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
  dglViewport(0, 0, target.width, target.height);
  dglScissor(0, 0, target.width, target.height);
  dglClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  dglClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  renderScene();

  rendering_eye_ = false;
}

void VRSystem::SubmitEyes() {
  if (!enabled_ || !compositor_) {
    return;
  }

  vr::Texture_t left_tex = {reinterpret_cast<void *>(static_cast<uintptr_t>(eye_targets_[0].color_tex)),
                            vr::TextureType_OpenGL, vr::ColorSpace_Gamma};
  vr::Texture_t right_tex = {reinterpret_cast<void *>(static_cast<uintptr_t>(eye_targets_[1].color_tex)),
                             vr::TextureType_OpenGL, vr::ColorSpace_Gamma};

  compositor_->Submit(vr::Eye_Left, &left_tex);
  compositor_->Submit(vr::Eye_Right, &right_tex);

  GLuint source_fbo = cinema_mode_ ? cinema_fbo_ : eye_targets_[0].fbo;
  int src_w = cinema_mode_ ? cinema_width_ : eye_targets_[0].width;
  int src_h = cinema_mode_ ? cinema_height_ : eye_targets_[0].height;

  int window_w = Max_window_w;
  int window_h = Max_window_h;

  dglBindFramebuffer(GL_READ_FRAMEBUFFER, source_fbo);
  dglBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  dglBlitFramebuffer(0, 0, src_w, src_h, 0, 0, window_w, window_h, GL_COLOR_BUFFER_BIT, GL_LINEAR);
  dglBindFramebuffer(GL_FRAMEBUFFER, 0);

  cinema_mode_ = false;
}

void VRSystem::BeginCinema() {
  if (!enabled_) {
    return;
  }

  cinema_mode_ = true;
  rendering_cinema_ = true;
  rendering_eye_ = false;

  dglBindFramebuffer(GL_FRAMEBUFFER, cinema_fbo_);
  dglViewport(0, 0, cinema_width_, cinema_height_);
  dglScissor(0, 0, cinema_width_, cinema_height_);
  dglClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  dglClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

GLuint VRSystem::GetCinemaColorTex() const { return cinema_color_tex_; }

void VRSystem::EndCinema() {
  if (!enabled_) {
    return;
  }

  dglBindFramebuffer(GL_FRAMEBUFFER, 0);
  rendering_cinema_ = false;
}

void VRSystem::RenderCinemaScreen() {
  if (!enabled_) {
    return;
  }

  const int eye_index = static_cast<int>(current_eye_);
  matrix4 proj = eye_projection_[eye_index];

  float proj_mat[16] = {
      proj.rvec.x(), proj.rvec.y(), proj.rvec.z(), proj.rvec.w(),
      proj.uvec.x(), proj.uvec.y(), proj.uvec.z(), proj.uvec.w(),
      proj.fvec.x(), proj.fvec.y(), proj.fvec.z(), proj.fvec.w(),
      proj.pos.x(),  proj.pos.y(),  proj.pos.z(),  proj.pos.w(),
  };

  dglDisable(GL_CULL_FACE);
  dglDisable(GL_LIGHTING);
  dglDisable(GL_BLEND);
  dglDisable(GL_DEPTH_TEST);
  dglDepthMask(GL_FALSE);
  dglEnable(GL_TEXTURE_2D);
  dglBindTexture(GL_TEXTURE_2D, cinema_color_tex_);

  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(proj_mat);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glRotatef(screen_yaw_ * 180.0f / kPi, 0.0f, 1.0f, 0.0f);

  glBegin(GL_TRIANGLE_STRIP);
  for (const auto &v : cinema_vertices_) {
    glTexCoord2f(v.u, v.v);
    glVertex3f(v.x, v.y, v.z);
  }
  glEnd();
}

bool VRSystem::IsRenderingEye() const { return rendering_eye_; }

bool VRSystem::IsRenderingCinema() const { return rendering_cinema_; }

bool VRSystem::GetActiveViewport(int &out_width, int &out_height) const {
  if (rendering_eye_) {
    const EyeRenderTarget &target = eye_targets_[static_cast<int>(current_eye_)];
    out_width = target.width;
    out_height = target.height;
    return true;
  }

  if (rendering_cinema_) {
    out_width = cinema_width_;
    out_height = cinema_height_;
    return true;
  }

  return false;
}

void VRSystem::BindActiveRenderTarget() const {
  if (rendering_eye_) {
    const EyeRenderTarget &target = eye_targets_[static_cast<int>(current_eye_)];
    dglBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
    dglViewport(0, 0, target.width, target.height);
    dglScissor(0, 0, target.width, target.height);
    return;
  }

  if (rendering_cinema_) {
    dglBindFramebuffer(GL_FRAMEBUFFER, cinema_fbo_);
    dglViewport(0, 0, cinema_width_, cinema_height_);
    dglScissor(0, 0, cinema_width_, cinema_height_);
  }
}

void VRSystem::SetBasePose(const vector &position, const matrix &orientation) {
  base_position_ = position;
  base_orientation_ = orientation;
}

const matrix4 &VRSystem::GetEyeProjection(Eye eye) const { return eye_projection_[static_cast<int>(eye)]; }

const matrix &VRSystem::GetEyeView(Eye eye) const { return eye_view_[static_cast<int>(eye)]; }

const vector &VRSystem::GetEyePosition(Eye eye) const { return eye_positions_[static_cast<int>(eye)]; }

float VRSystem::GetEyeZoom(Eye eye) const { return eye_zoom_[static_cast<int>(eye)]; }

void VRSystem::GetCinemaSize(int &out_width, int &out_height) const {
  out_width = cinema_width_;
  out_height = cinema_height_;
}

bool VRSystem::GetCinemaPointer(int &out_x, int &out_y, bool &out_click_down) const {
  if (!pointer_valid_) {
    return false;
  }

  out_x = pointer_x_;
  out_y = pointer_y_;
  out_click_down = pointer_click_down_;
  return true;
}

void VRSystem::UpdateEyeMatrices() {
  if (!hmd_) {
    return;
  }

  for (int eye = 0; eye < 2; ++eye) {
    vr::Hmd_Eye vr_eye = (eye == 0) ? vr::Eye_Left : vr::Eye_Right;
    vr::HmdMatrix44_t proj = hmd_->GetProjectionMatrix(vr_eye, kNearZ, kFarZ);
    eye_projection_[eye] = HmdMatrixToMatrix4(proj);

    vr::HmdMatrix34_t eye_to_head = hmd_->GetEyeToHeadTransform(vr_eye);
    matrix eye_matrix = HmdMatrixToMatrix(eye_to_head);
    vector eye_pos = HmdMatrixGetPosition(eye_to_head);

    matrix eye_matrix_inv = eye_matrix;
    vm_TransposeMatrix(&eye_matrix_inv);
    vector neg_pos = eye_pos * -1.0f;
    vector head_to_eye = TransformPoint(eye_matrix_inv, neg_pos);
    eye_offsets_[eye] = head_to_eye;

    matrix final_orient = base_orientation_ * hmd_orientation_;
    vector offset = hmd_position_ + TransformPoint(hmd_orientation_, head_to_eye);
    vector final_pos = base_position_ + TransformPoint(base_orientation_, offset);

    eye_view_[eye] = final_orient;
    eye_positions_[eye] = final_pos;
  }
}

void VRSystem::UpdateCinemaPointer() {
  pointer_valid_ = false;
  pointer_click_down_ = false;

  if (!hmd_) {
    return;
  }

  vector ray_dir = hmd_orientation_.fvec * -1.0f;
  if (ray_dir.x() == 0.0f && ray_dir.y() == 0.0f && ray_dir.z() == 0.0f) {
    return;
  }

  float yaw = screen_yaw_;
  float cos_y = std::cos(-yaw);
  float sin_y = std::sin(-yaw);
  vector local_dir;
  local_dir.x() = ray_dir.x() * cos_y - ray_dir.z() * sin_y;
  local_dir.y() = ray_dir.y();
  local_dir.z() = ray_dir.x() * sin_y + ray_dir.z() * cos_y;

  float center_z = -(kCinemaDistance + kCinemaRadius);
  float a = local_dir.x() * local_dir.x() + local_dir.z() * local_dir.z();
  float b = 2.0f * (local_dir.x() * 0.0f + local_dir.z() * -center_z);
  float c = center_z * center_z - kCinemaRadius * kCinemaRadius;
  float disc = b * b - 4.0f * a * c;
  if (disc < 0.0f) {
    return;
  }

  float t = (-b - std::sqrt(disc)) / (2.0f * a);
  if (t <= 0.0f) {
    return;
  }

  vector hit = local_dir * t;
  float angle = std::atan2(hit.x(), hit.z() - center_z);
  float arc_rad = kCinemaArcDegrees * kPi / 180.0f;
  float start_angle = -arc_rad * 0.5f;
  float end_angle = arc_rad * 0.5f;

  if (angle < start_angle || angle > end_angle) {
    return;
  }

  float u = (angle - start_angle) / arc_rad;
  float v = (hit.y() + kCinemaHeight * 0.5f) / kCinemaHeight;
  if (v < 0.0f || v > 1.0f) {
    return;
  }

  pointer_x_ = static_cast<int>(u * static_cast<float>(cinema_width_));
  pointer_y_ = static_cast<int>((1.0f - v) * static_cast<float>(cinema_height_));
  pointer_valid_ = true;

  for (vr::TrackedDeviceIndex_t device = 0; device < vr::k_unMaxTrackedDeviceCount; ++device) {
    if (!tracked_poses_[device].bPoseIsValid) {
      continue;
    }
    if (hmd_->GetTrackedDeviceClass(device) != vr::TrackedDeviceClass_Controller) {
      continue;
    }
    vr::VRControllerState_t state{};
    if (hmd_->GetControllerState(device, &state, sizeof(state))) {
      if (state.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Trigger)) {
        pointer_click_down_ = true;
        break;
      }
      if (state.rAxis[1].x > 0.5f) {
        pointer_click_down_ = true;
        break;
      }
    }
  }
}

void VRSystem::UpdateScreenRecenter(float dt) {
  vector forward = hmd_orientation_.fvec;
  float yaw = std::atan2(forward.x(), -forward.z());
  screen_yaw_target_ = yaw;
  float delta = WrapAngle(screen_yaw_target_ - screen_yaw_);
  screen_yaw_ += delta * std::min(1.0f, dt * kCinemaRecenterSpeed);
}

matrix VRSystem::HmdMatrixToMatrix(const vr::HmdMatrix34_t &mat) const {
  matrix result;
  result.rvec = {mat.m[0][0], mat.m[1][0], mat.m[2][0]};
  result.uvec = {mat.m[0][1], mat.m[1][1], mat.m[2][1]};
  result.fvec = {mat.m[0][2], mat.m[1][2], mat.m[2][2]};
  return result;
}

matrix4 VRSystem::HmdMatrixToMatrix4(const vr::HmdMatrix44_t &mat) const {
  matrix4 result;
  result.rvec = {mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0]};
  result.uvec = {mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1]};
  result.fvec = {mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2]};
  result.pos = {mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]};
  return result;
}

vector VRSystem::HmdMatrixGetPosition(const vr::HmdMatrix34_t &mat) const {
  return {mat.m[0][3], mat.m[1][3], mat.m[2][3]};
}

vector VRSystem::TransformPoint(const matrix &basis, const vector &pos) const {
  vector out;
  vm_MatrixMulVector(&out, &pos, &basis);
  return out;
}
