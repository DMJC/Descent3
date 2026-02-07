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

#ifndef D3_VR_H
#define D3_VR_H

class NewBitmap;

enum class VrRenderMode {
  Cinema,
  Stereo,
};

void VR_InitFromCommandLine();
bool VR_IsEnabled();
VrRenderMode VR_GetRenderMode();
bool VR_IsStereoRendering();
float VR_GetStereoEyeSeparation();
void VR_RenderMenuFrame();
void VR_SubmitStereoFrame(const NewBitmap &left, const NewBitmap &right);
void VR_ResetGraphicsResources();
void VR_Shutdown();

#endif
