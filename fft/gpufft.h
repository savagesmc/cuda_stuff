#pragma once
#include <vector>
#include <complex>

namespace GpuUtils
{
   typedef std::complex<float> MyComplex;
   std::vector<MyComplex> fft(const std::vector<MyComplex>& samples);
}
