/*
 * Descent 3
 * Copyright (C) 2024 Descent Developers
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

#include "movie_sound.h"

namespace D3 {

MovieSoundDevice::MovieSoundDevice(int sample_rate, uint16_t sample_size, uint8_t channels, bool is_compressed) {
  SDL_AudioSpec spec{};
  spec.freq = sample_rate;
  spec.format = (sample_size == 2) ? SDL_AUDIO_S16LE : SDL_AUDIO_U8;
  spec.channels = channels;

  spec.samples = 1024;
  spec.callback = nullptr;

  this->device = SDL_OpenAudioDevice(nullptr, 0, &spec, nullptr, 0);
  this->m_is_compressed = is_compressed;
  this->m_sample_size = sample_size;
};

MovieSoundDevice::~MovieSoundDevice() {
  if (this->device != 0) {
    SDL_CloseAudioDevice(this->device);
    this->device = 0;
  }
}

void MovieSoundDevice::FillBuffer(char *buffer, int len) const {
  if (this->device != 0) {
    SDL_QueueAudio(this->device, buffer, len);
  }
};

void MovieSoundDevice::Play() {
  if (this->device != 0) {
    SDL_PauseAudioDevice(this->device, 0);
  }
}

void MovieSoundDevice::Stop() {
  if (this->device != 0) {
    SDL_ClearQueuedAudio(this->device);
    SDL_PauseAudioDevice(this->device, 1);
  }
}

void MovieSoundDevice::Lock() {}

void MovieSoundDevice::Unlock() {}

} // namespace D3
