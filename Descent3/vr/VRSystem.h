#pragma once

#include <array>
#include <functional>
#include <vector>

#include <openvr.h>

#include "renderer/dyna_gl.h"
#include "vecmat.h"

class VRSystem {
public:
  enum class Eye { Left = 0, Right = 1 };

  static VRSystem &Get();

  bool Initialize(bool enable_vr);
  void Shutdown();

  bool Enabled() const;
  bool InCinemaMode() const;

  void BeginFrame();
  void RenderEye(Eye eye, const std::function<void()> &renderScene);
  void SubmitEyes();

  void BeginCinema();
  GLuint GetCinemaColorTex() const;
  void EndCinema();
  void RenderCinemaScreen();

  bool IsRenderingEye() const;
  bool IsRenderingCinema() const;
  bool GetActiveViewport(int &out_width, int &out_height) const;
  void BindActiveRenderTarget() const;

  void SetBasePose(const vector &position, const matrix &orientation);
  const matrix4 &GetEyeProjection(Eye eye) const;
  const matrix &GetEyeView(Eye eye) const;
  const vector &GetEyePosition(Eye eye) const;
  float GetEyeZoom(Eye eye) const;

  void GetCinemaSize(int &out_width, int &out_height) const;
  bool GetCinemaPointer(int &out_x, int &out_y, bool &out_click_down) const;

private:
  VRSystem();
  ~VRSystem();

  struct EyeRenderTarget {
    GLuint fbo = 0;
    GLuint color_tex = 0;
    GLuint depth_rb = 0;
    int width = 0;
    int height = 0;
  };

  struct CinemaVertex {
    float x;
    float y;
    float z;
    float u;
    float v;
  };

  bool InitializeOpenVR();
  void ShutdownOpenVR();
  void CreateEyeTargets();
  void CreateCinemaTarget();
  void DestroyTargets();
  void BuildCinemaMesh();
  void UpdateEyeMatrices();
  void UpdateCinemaPointer();
  void UpdateScreenRecenter(float dt);

  matrix HmdMatrixToMatrix(const vr::HmdMatrix34_t &mat) const;
  matrix4 HmdMatrixToMatrix4(const vr::HmdMatrix44_t &mat) const;
  vector HmdMatrixGetPosition(const vr::HmdMatrix34_t &mat) const;

  vector TransformPoint(const matrix &basis, const vector &pos) const;

  bool enabled_ = false;
  bool cinema_mode_ = false;
  bool rendering_eye_ = false;
  bool rendering_cinema_ = false;
  Eye current_eye_ = Eye::Left;

  vr::IVRSystem *hmd_ = nullptr;
  vr::IVRCompositor *compositor_ = nullptr;

  std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> tracked_poses_{};
  matrix hmd_orientation_{};
  vector hmd_position_{};

  matrix base_orientation_{};
  vector base_position_{};

  std::array<matrix4, 2> eye_projection_{};
  std::array<matrix, 2> eye_view_{};
  std::array<vector, 2> eye_positions_{};
  std::array<float, 2> eye_zoom_{};
  std::array<vector, 2> eye_offsets_{};

  std::array<EyeRenderTarget, 2> eye_targets_{};

  GLuint cinema_fbo_ = 0;
  GLuint cinema_color_tex_ = 0;
  GLuint cinema_depth_rb_ = 0;
  int cinema_width_ = 2048;
  int cinema_height_ = 2048;

  std::vector<CinemaVertex> cinema_vertices_{};

  float screen_yaw_ = 0.0f;
  float screen_yaw_target_ = 0.0f;
  double last_update_time_ = 0.0;

  bool pointer_valid_ = false;
  int pointer_x_ = 0;
  int pointer_y_ = 0;
  bool pointer_click_down_ = false;
};
