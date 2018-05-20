#include "gpufft.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <fstream>
#include "TimeStat.h"

using namespace std;
using namespace std::chrono;

typedef GpuUtils::FftEngine::Complex Complex;

const float pi = atan2(1.0, 1.0) * 4.0;

const float freq     = 100.;
const float T        = 0.001;
const float omega    = 2 * pi * freq;

void doTest(float frequency, int numSamples, bool printTime = false, bool outputToFile = false)
{
   ostringstream ostr;
   ostr << "complexFft(" << setw(8) << numSamples << ")";

   ofstream ofile("output.csv");
   ofstream ifile("input.csv");

   std::vector<Complex> samples;
   for (auto i = 0; i < numSamples; ++i)
   {
      auto t = i * T;
      auto rad = omega * t;
      samples.push_back(Complex(cos(rad), sin(rad)));
   }

   if (outputToFile)
   {
      for (auto&& s : samples)
      {
         ifile << s.real() << ", " << s.imag() << "\n";
      }
   }

   {
      GpuUtils::FftEngine fft(samples.size());

      const int numTimes = 50;
      TimeStat ts(ostr.str(), numTimes);
      for (auto i=0; i<numTimes; ++i)
      {
         fft(samples);
      }
   }

   if (outputToFile)
   {
      for (auto s : samples)
      {
         ofile << s.real() << ", " << s.imag() << "\n";
      }
      ofile.flush();
   }
}


int main(int argc, char* argv[])
{
   doTest(5.0, 16384);

   doTest(5.0, 1024);
   doTest(5.0, 1024);
   doTest(5.0, 2048);
   doTest(5.0, 4096);
   doTest(5.0, 8192);
   doTest(5.0, 16384);
   doTest(5.0, 16384*2);
   doTest(5.0, 16384*4);
   doTest(5.0, 16384*8);
   doTest(5.0, 16384*16);
   doTest(5.0, 16384*32);
   doTest(5.0, 16384*64);
   doTest(5.0, 16384*64, true, true);
}
