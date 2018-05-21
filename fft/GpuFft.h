#ifndef GPU_UTILS_GPU_FFT_H
#define GPU_UTILS_GPU_FFT_H
#pragma once

#include <vector>
#include <complex>
#include <memory>

#include "FftTypes.h"

namespace GpuUtils
{
   using Signals::Complex;

   class FftEngine
   {
   public:

      FftEngine(int size);
      ~FftEngine();

      void debug(bool enableDisable);

      void operator()(std::vector<Complex>& samples);

   private:
      class Impl;
      std::unique_ptr<Impl> impl_;
   };
}

#endif
