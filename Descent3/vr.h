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

#pragma once

#include "vecmat.h"

namespace d3vr {

enum class Eye : int { Left = 0, Right = 1 };

struct EyeRenderData {
  vector position;
  matrix orientation;
  float projection[4][4];
  float zoom;
  int renderWidth;
  int renderHeight;
};

void SetRequested(bool requested);
bool IsRequested();
bool Initialize();
void Shutdown();
bool IsEnabled();

bool BeginFrame(const vector &basePos, const matrix &baseOrient);
bool GetEyeData(Eye eye, EyeRenderData &out);
bool BindEye(Eye eye, const EyeRenderData &data);
void UnbindEye();
void SetCurrentEyeData(const EyeRenderData *data);
void ClearCurrentEyeData();
void OnGameRenderWorldStart();
void SubmitFrame();
void MirrorToBackbuffer();
void CancelFrame();

} // namespace d3vr
