#pragma once
#include <vector>
#include <complex>

namespace GpuUtils
{
   typedef std::complex<float> MyComplex;
   typedef std::vector<MyComplex> MyComplexVect;
   void fft(MyComplexVect& samples, bool debug=false);
}
