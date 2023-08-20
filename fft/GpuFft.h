#ifndef GPU_UTILS_GPU_FFT_H
#define GPU_UTILS_GPU_FFT_H
#pragma once

#include <vector>
#include <complex>
#include <memory>
#include <future>

#include "FftTypes.h"

namespace GpuUtils
{
using Signals::Complex;
using Signals::ComplexVec;

class FftEngine
{
public:

  FftEngine(int fftSize, int blockSize, bool debug=false);
  ~FftEngine();
  std::future<ComplexVec> dofft(ComplexVec& samples);
  bool busy() const;

  class Impl;
private:
  std::unique_ptr<Impl> impl_;
};
}

#endif
