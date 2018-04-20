#include "gpufft.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;
using GpuUtils::MyComplex;

const float pi = atan2(1.0, 1.0) * 4.0;

const int numSamples = 1024;
const float freq     = 100.;
const float T        = 0.001;
const float omega    = 2 * pi * freq;

class TimeStat
{
   steady_clock::time_point start;
   string label;
public:
   TimeStat(const string& s) : label(s), start(steady_clock::now()) { }
   ~TimeStat() {
      steady_clock::time_point stop = steady_clock::now();
      duration<double> dur = stop - start;
      cout << label << " " << (dur.count()) << endl;
   }
};

void doTest(float frequency, int numSamples, bool printTime = true, bool outputToFile = false)
{
   ostringstream ostr;
   ostr << "complexFft(" << setw(6) << numSamples << ")";

   ofstream ofile("output.csv");
   ofstream ifile("input.csv");

   std::vector<MyComplex> samples;
   for (auto i = 0; i < numSamples; ++i)
   {
      auto t = i * T;
      auto rad = omega * t;
      samples.push_back(MyComplex(cos(rad), sin(rad)));
   }

   if (outputToFile)
   {
      for (auto&& s : samples)
      {
         ifile << s.real() << ", " << s.imag() << "\n";
      }
   }

   std::vector<MyComplex> result;
   {
      if (printTime)
      {
         TimeStat ts(ostr.str());
      }
      result = GpuUtils::fft(samples);
   }

   if (outputToFile)
   {
      for (auto&& s : result)
      {
         ofile << s.real() << ", " << s.imag() << "\n";
      }
      ofile.flush();
   }
}


int main(int argc, char* argv[])
{
   doTest(5.0, 16384, false);

   doTest(5.0, 1024);
   doTest(5.0, 2048);
   doTest(5.0, 4096);
   doTest(5.0, 8192);
   doTest(5.0, 16384, true, true);
}
