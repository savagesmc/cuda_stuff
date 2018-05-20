#include "gpufft.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <fstream>
#include "TimeStat.h"

using namespace std;
using namespace std::chrono;
using GpuUtils::MyComplex;

const float pi = atan2(1.0, 1.0) * 4.0;

const float freq     = 100.;
const float T        = 0.001;
const float omega    = 2 * pi * freq;

void doTest(float frequency, int numSamples, int numBlocks = 1, bool printTime = true, bool outputToFile = false)
{
   ostringstream ostr;
   ostr << "complexFft(" << setw(8) << numSamples << ")";

   ofstream ofile("output.csv");
   ofstream ifile("input.csv");

   std::vector<MyComplex> samples;
   for (auto i = 0; i < numSamples; ++i)
   {
      auto t = i * T;
      auto rad = omega * t;
      samples.push_back(MyComplex(cos(rad), sin(rad)));
   }

   std::vector<std::vector<MyComplex> > blocks;
   for (auto i = 0; i < numBlocks; ++i)
   {
      blocks.push_back(samples);
   }

   if (outputToFile)
   {
      for (auto&& s : samples)
      {
         ifile << s.real() << ", " << s.imag() << "\n";
      }
   }

   std::vector<std::vector<MyComplex> > result;
   {
      if (printTime)
      {
         TimeStat ts(ostr.str(), blocks.size());
         result = GpuUtils::fft(blocks);
      }
      else
      {
         result = GpuUtils::fft(blocks);
      }
   }

   if (outputToFile)
   {
      for (auto&& s : result[0])
      {
         ofile << s.real() << ", " << s.imag() << "\n";
      }
      ofile.flush();
   }
}


int main(int argc, char* argv[])
{
   doTest(5.0, 16384, 1, false);

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
   doTest(5.0, 16384*64, 10, true, true);
}
