#pragma once
#include <vector>
#include <complex>
#include <memory>

namespace GpuUtils
{
   class FftEngine
   {
   public:

      typedef std::complex<float> Complex;

      FftEngine(int size);
      ~FftEngine();

      void debug(bool enableDisable);

      void operator()(std::vector<Complex>& samples);

   private:
      class Impl;
      std::unique_ptr<Impl> impl_;
   };
}
