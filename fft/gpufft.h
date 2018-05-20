#pragma once
#include <vector>
#include <complex>

namespace GpuUtils
{
   typedef std::complex<float> MyComplex;
   typedef std::vector<MyComplex> MyComplexVect;
   std::vector<MyComplexVect> fft(const std::vector<MyComplexVect>& samples);
}
