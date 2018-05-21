#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <fstream>
#include "TimeStat.h"
#include "Fft.h"

using namespace std;
using namespace std::chrono;

typedef Signals::Complex Complex;

const float pi = atan2(1.0, 1.0) * 4.0;

const float freq     = 100.;
const float T        = 0.001;
const float omega    = 2 * pi * freq;

void doTest(Signals::Fft fft, float frequency, int numSamples, bool printTime = false, bool outputToFile = false)
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
      const int numTimes = 50;
      TimeStat ts(ostr.str(), numTimes);
      for (auto i=0; i<numTimes; ++i)
      {
         std::vector<Complex> cpySamps = samples;
         fft.submit(cpySamps);
      }
   }


   std::vector<Complex> resultSamps;
   {
      const int numTimes = 50;
      TimeStat ts(ostr.str(), numTimes);
      for (auto i=0; i<numTimes; ++i)
      {
         std::vector<Complex> cpySamps = fft.result();
         resultSamps = cpySamps;
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
   doTest(Signals::Fft(1024), 5.0, 1024, false);
   // doTest(fft, 5.0, 1024);
   // doTest(fft, 5.0, 2048);
   // doTest(fft, 5.0, 4096);
   // doTest(fft, 5.0, 8192);
   doTest(Signals::Fft(16384), 5.0, 16384);
   // doTest(fft, 5.0, 16384*2);
   // doTest(fft, 5.0, 16384*4);
   // doTest(fft, 5.0, 16384*8);
   // doTest(fft, 5.0, 16384*16);
   // doTest(fft, 5.0, 16384*32);
   doTest(Signals::Fft(16384*64), 5.0, 16384*64);
   // doTest(fft, 5.0, 16384*64, true, true);
}
