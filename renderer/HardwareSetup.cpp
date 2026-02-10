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

#include <cstring>

#include "3d.h"
#include "log.h"
#include "HardwareInternal.h"
#include "renderer.h"

// User-specified aspect ratio, stored as w/h
static float sAspect = 0.0f;
static g3StereoFrustum sStereoFrustum[2];
bool sStereoFrustumValid = false;

// allows the user to specify an aspect ratio that overrides the renderer's
// The parameter is the w/h of the screen pixels
void g3_SetAspectRatio(float aspect) { sAspect = aspect; }
// returns the user-specified aspect ratio used to override the renderer's
float g3_GetAspectRatio() { return sAspect; }

void g3_SetStereoFrustum(const g3StereoFrustum *left, const g3StereoFrustum *right) {
  if (!left || !right) {
    LOG_INFO << "VR-MENU-TRACE: g3_SetStereoFrustum invalid input";
    sStereoFrustumValid = false;
    return;
  }

  sStereoFrustum[0] = *left;
  sStereoFrustum[1] = *right;
  sStereoFrustumValid = true;

  LOG_INFO.printf("VR-MENU-TRACE: g3_SetStereoFrustum L(l=%f r=%f t=%f b=%f) R(l=%f r=%f t=%f b=%f)", left->left,
                  left->right, left->top, left->bottom, right->left, right->right, right->top, right->bottom);
}

void g3_GetViewPortMatrix(float *viewMat) {
  // extract the viewport data from the renderer
  int viewportWidth, viewportHeight;
  int viewportX, viewportY;
  rend_GetProjectionScreenParameters(viewportX, viewportY, viewportWidth, viewportHeight);
  LOG_INFO.printf("VR-MENU-TRACE: g3_GetViewPortMatrix viewport x=%d y=%d w=%d h=%d", viewportX, viewportY,
                  viewportWidth, viewportHeight);

  float viewportWidthOverTwo = ((float)viewportWidth) * 0.5f;
  float viewportHeightOverTwo = ((float)viewportHeight) * 0.5f;

  // setup the matrix
  memset(viewMat, 0, sizeof(float) * 16);
  viewMat[0] = viewportWidthOverTwo;
  viewMat[5] = -viewportHeightOverTwo;
  viewMat[12] = viewportWidthOverTwo + (float)viewportX;
  viewMat[13] = viewportHeightOverTwo + (float)viewportY;
  viewMat[10] = viewMat[15] = 1.0f;
}

void g3_GetProjectionMatrix(float zoom, float *projMat) {
  // get window size
  int viewportWidth, viewportHeight;
  rend_GetProjectionParameters(&viewportWidth, &viewportHeight);

  float s = ((float)viewportWidth) / ((float)viewportHeight);
  float vertical_fov = zoom * 3.0f / 4.0f;

  // setup the matrix
  memset(projMat, 0, sizeof(float) * 16);

  // calculate 1/tan(fov)
  float oOT = 1.0f / vertical_fov;

  // fill in the matrix
  // Go read https://www.songho.ca/opengl/gl_projectionmatrix.html
  // if you feel like doing the math again :)
	if (s <= 1.0f)
	{
		projMat[0] = oOT;
		projMat[5] = oOT * s;
	}
	else
	{
		projMat[0] = oOT / s;
		projMat[5] = oOT;
	}

  projMat[10] = 1.0f;
  projMat[11] = 1.0f;
  projMat[14] = -1.0f;

  LOG_INFO.printf("VR-MENU-TRACE: g3_GetProjectionMatrix zoom=%f viewport_w=%d viewport_h=%d proj00=%f proj11=%f", zoom,
                  viewportWidth, viewportHeight, projMat[0], projMat[5]);
}

// start the frame
void g3_StartFrame(vector *view_pos, matrix *view_matrix, float zoom) {
  // initialize the viewport transform
  LOG_INFO.printf("VR-MENU-TRACE: g3_StartFrame zoom=%f view_pos=(%f,%f,%f)", zoom, view_pos->x(), view_pos->y(),
                  view_pos->z());
  g3_GetViewPortMatrix((float *)gTransformViewPort);
  g3_GetProjectionMatrix(zoom, (float *)gTransformProjection);
  if (sStereoFrustumValid) {
    LOG_INFO.printf("VR-MENU-TRACE: g3_StartFrame stereo-frustum-active L(l=%f r=%f t=%f b=%f) R(l=%f r=%f t=%f b=%f)",
                    sStereoFrustum[0].left, sStereoFrustum[0].right, sStereoFrustum[0].top, sStereoFrustum[0].bottom,
                    sStereoFrustum[1].left, sStereoFrustum[1].right, sStereoFrustum[1].top, sStereoFrustum[1].bottom);
    LOG_INFO << "VR-MENU-TRACE: g3_StartFrame using mono g3_GetProjectionMatrix while stereo frusta are active";
  }
  g3_GetModelViewMatrix(view_pos, view_matrix, (float *)gTransformModelView);
  g3_UpdateFullTransform();

  // get window size
  rend_GetProjectionParameters(&Window_width, &Window_height);

  // Set vars for projection
  Window_w2 = ((scalar)Window_width) * 0.5f;
  Window_h2 = ((scalar)Window_height) * 0.5f;
  LOG_INFO.printf("VR-MENU-TRACE: g3_StartFrame projection window_w=%d window_h=%d window_w2=%f window_h2=%f", Window_width,
                  Window_height, Window_w2, Window_h2);

  // ISB trick: use the window aspect only, screen aspect ratio
  // is not important because we assume pixels are square
  scalar s = (scalar)Window_height / (scalar)Window_width;

  Matrix_scale = { s <= 1.0f ? s : 1.0f / s, 1.0f };


  //ISB: Convert zoom into vertical FOV for convenience
  zoom *= 3.f / 4.f;

  Matrix_scale.z() = 1.0f;

  // Set the view variables
  View_position = *view_pos;
  View_zoom = zoom;
  Unscaled_matrix = *view_matrix;

  // Scale x and y to zoom in or out;
  float oOZ = 1.0f / View_zoom;
  Matrix_scale.x() = Matrix_scale.x() * oOZ;
  Matrix_scale.y() = Matrix_scale.y() * oOZ;

  // Scale the matrix elements
  View_matrix.rvec = Unscaled_matrix.rvec * Matrix_scale.x();
  View_matrix.uvec = Unscaled_matrix.uvec * Matrix_scale.y();
  View_matrix.fvec = Unscaled_matrix.fvec * Matrix_scale.z();

  // Reset the list of free points
  InitFreePoints();

  // Reset the far clip plane
  g3_ResetFarClipZ();
}

// Add this new function
void g3_GetStereoProjectionMatrix(float zoom, bool is_left_eye, float eye_separation, float convergence_distance, float *projMat) {
  if (sStereoFrustumValid) {
    const g3StereoFrustum &frustum = sStereoFrustum[is_left_eye ? 0 : 1];
    const g3StereoFrustum &left_eye_frustum = sStereoFrustum[0];
    const g3StereoFrustum &right_eye_frustum = sStereoFrustum[1];

    memset(projMat, 0, sizeof(float) * 16);

    // Normalize incoming bounds so projection math is stable, then use a
    // shared width/height large enough for both eyes to avoid one eye being
    // horizontally cropped/stretched when eye frusta differ.
    const float left = (frustum.left < frustum.right) ? frustum.left : frustum.right;
    const float right = (frustum.left < frustum.right) ? frustum.right : frustum.left;
    const float bottom = (frustum.bottom < frustum.top) ? frustum.bottom : frustum.top;
    const float top = (frustum.bottom < frustum.top) ? frustum.top : frustum.bottom;

    const float left_eye_width = (left_eye_frustum.left < left_eye_frustum.right)
                                     ? (left_eye_frustum.right - left_eye_frustum.left)
                                     : (left_eye_frustum.left - left_eye_frustum.right);
    const float right_eye_width = (right_eye_frustum.left < right_eye_frustum.right)
                                      ? (right_eye_frustum.right - right_eye_frustum.left)
                                      : (right_eye_frustum.left - right_eye_frustum.right);
    const float left_eye_height = (left_eye_frustum.bottom < left_eye_frustum.top)
                                      ? (left_eye_frustum.top - left_eye_frustum.bottom)
                                      : (left_eye_frustum.bottom - left_eye_frustum.top);
    const float right_eye_height = (right_eye_frustum.bottom < right_eye_frustum.top)
                                       ? (right_eye_frustum.top - right_eye_frustum.bottom)
                                       : (right_eye_frustum.bottom - right_eye_frustum.top);

    const float width = (left_eye_width > right_eye_width) ? left_eye_width : right_eye_width;
    const float height = (left_eye_height > right_eye_height) ? left_eye_height : right_eye_height;

    LOG_INFO.printf("VR-MENU-TRACE: g3_GetStereoProjectionMatrix eye=%s frustum(l=%f r=%f t=%f b=%f) shared_w=%f shared_h=%f",
                    is_left_eye ? "LEFT" : "RIGHT", left, right, top, bottom, width, height);

    if (width == 0.0f || height == 0.0f) {
      LOG_INFO << "VR-MENU-TRACE: g3_GetStereoProjectionMatrix falling back to default projection";
      // Don't invalidate globally, just fall through to default projection
    } else {
      projMat[0] = 2.0f / width;
      projMat[5] = 2.0f / height;
      projMat[8] = (right + left) / width;
      projMat[9] = (top + bottom) / height;
      projMat[10] = 1.0f;
      projMat[11] = 1.0f;
      projMat[14] = -1.0f;
      LOG_INFO.printf("VR-MENU-TRACE: g3_GetStereoProjectionMatrix eye=%s proj00=%f proj11=%f proj20=%f proj21=%f",
                      is_left_eye ? "LEFT" : "RIGHT", projMat[0], projMat[5], projMat[8], projMat[9]);
      return;
    }
  }

  // get window size
  int viewportWidth, viewportHeight;
  rend_GetProjectionParameters(&viewportWidth, &viewportHeight);

  float s = ((float)viewportWidth) / ((float)viewportHeight);
  float vertical_fov = zoom * 3.0f / 4.0f;

  // setup the matrix
  memset(projMat, 0, sizeof(float) * 16);

  // calculate 1/tan(fov)
  float oOT = 1.0f / vertical_fov;

  // Calculate the eye offset for this eye
  float eye_offset = is_left_eye ? -eye_separation / 2.0f : eye_separation / 2.0f;
  float frustum_shift = 0.0f;
  if (convergence_distance > 0.0f && eye_separation != 0.0f) {
    // Shift the frustum opposite of the eye offset so the convergence plane remains centered.
    frustum_shift = -eye_offset / convergence_distance;
  }
  
  // Apply the asymmetric frustum offset
  if (s <= 1.0f) {
    projMat[0] = oOT;
    projMat[5] = oOT * s;
  } else {
    projMat[0] = oOT / s;
    projMat[5] = oOT;
  }

  // THIS IS THE KEY: Add horizontal offset for stereo (projMat[8])
  projMat[8] = frustum_shift * projMat[0]; // Scale by horizontal FOV factor

  projMat[10] = 1.0f;
  projMat[11] = 1.0f;
  projMat[14] = -1.0f;
}

// Add stereo version of g3_StartFrame
void g3_StartFrameStereo(vector *view_pos, matrix *view_matrix, float zoom, bool is_left_eye, float eye_separation, float convergence_distance) {
  // initialize the viewport transform
  g3_GetViewPortMatrix((float *)gTransformViewPort);
  
  // USE STEREO PROJECTION instead of regular
  LOG_INFO.printf("VR-MENU-TRACE: g3_StartFrameStereo eye=%s zoom=%f eye_sep=%f conv=%f", is_left_eye ? "LEFT" : "RIGHT",
                  zoom, eye_separation, convergence_distance);
  g3_GetStereoProjectionMatrix(zoom, is_left_eye, eye_separation, convergence_distance, (float *)gTransformProjection);

  LOG_INFO.printf("VR-MENU-TRACE: g3_StartFrameStereo eye=%s proj00=%f proj11=%f proj20=%f proj21=%f stereoValid=%d",
                  is_left_eye ? "LEFT" : "RIGHT", gTransformProjection[0][0], gTransformProjection[1][1],
                  gTransformProjection[2][0], gTransformProjection[2][1], sStereoFrustumValid ? 1 : 0);
  
  g3_GetModelViewMatrix(view_pos, view_matrix, (float *)gTransformModelView);
  g3_UpdateFullTransform();

  // ... rest of the function stays the same as g3_StartFrame ...
  // (copy the remaining code from g3_StartFrame)
    // get window size
  rend_GetProjectionParameters(&Window_width, &Window_height);

  // Set vars for projection
  Window_w2 = ((scalar)Window_width) * 0.5f;
  Window_h2 = ((scalar)Window_height) * 0.5f;

  // ISB trick: use the window aspect only, screen aspect ratio
  // is not important because we assume pixels are square
  scalar s = (scalar)Window_height / (scalar)Window_width;

  Matrix_scale = { s <= 1.0f ? s : 1.0f / s, 1.0f };

  //ISB: Convert zoom into vertical FOV for convenience
  zoom *= 3.f / 4.f;

  Matrix_scale.z() = 1.0f;

  // Set the view variables
  View_position = *view_pos;
  View_zoom = zoom;
  Unscaled_matrix = *view_matrix;

  // Scale x and y to zoom in or out;
  float oOZ = 1.0f / View_zoom;
  Matrix_scale.x() = Matrix_scale.x() * oOZ;
  Matrix_scale.y() = Matrix_scale.y() * oOZ;

  // Scale the matrix elements
  View_matrix.rvec = Unscaled_matrix.rvec * Matrix_scale.x();
  View_matrix.uvec = Unscaled_matrix.uvec * Matrix_scale.y();
  View_matrix.fvec = Unscaled_matrix.fvec * Matrix_scale.z();

  // Reset the list of free points
  InitFreePoints();

  // Reset the far clip plane
  g3_ResetFarClipZ();
}

// this doesn't do anything, but is here for completeness
void g3_EndFrame(void) {
  // make sure temp points are free
  CheckTempPoints();
}

// get the current view position
void g3_GetViewPosition(vector *vp) { *vp = View_position; }

void g3_GetViewMatrix(matrix *mat) { *mat = View_matrix; }

void g3_GetUnscaledMatrix(matrix *mat) { *mat = Unscaled_matrix; }

// Gets the matrix scale vector
void g3_GetMatrixScale(vector *matrix_scale) { *matrix_scale = Matrix_scale; }
