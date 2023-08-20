#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <list>
#include "TimeStat.h"
#include "Fft.h"

using namespace std;
using namespace std::chrono;

using Signals::Complex;
using Signals::ComplexVec;

const float pi = atan2(1.0, 1.0) * 4.0;

const float freq     = 100.;
const float T        = 0.001;
const float omega    = 2 * pi * freq;

void doTest(Signals::Fft fft, float frequency, int numSamples, int numFfts, bool printTime = false, bool outputToFile = false)
{
  ostringstream ostr;
  ostr << "complexFft(" << setw(8) << numSamples << ")";

  ofstream ofile("output.csv");
  ofstream ifile("input.csv");

  ComplexVec samples;
  ComplexVec resultSamps;
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

  steady_clock::duration totalTime;
  {
    TimeStat ts("complexFft["+to_string(numSamples)+"]:", numFfts);
    typedef std::future<ComplexVec> Future;
    std::list<Future> futures;
    int numSubmitted = 0;
    int numComplete = 0;
    while (numComplete < numFfts)
    {
      while (!fft.busy() && (numSubmitted < numFfts)) {
        auto f = fft.submit(samples);
        futures.emplace_back(std::move(f));
        numSubmitted += 1;
      }
      std::list<Future>::iterator iter;
      for (iter = futures.begin(); iter != futures.end();)
      {
        auto temp = iter++;
        if (temp->wait_for(milliseconds(0)) == future_status::ready)
        {
          numComplete += 1;
          resultSamps = std::move(temp->get());
          futures.erase(temp);
        }
      }
    }
    totalTime = ts.elapsed();
  }

  std::cout << "FFT [" << fft.fftSize() << "," << numSamples << "]::Samples Per Second: " << numFfts * numSamples / duration_cast<duration<double>>(totalTime).count() << std::endl;

  if (outputToFile)
  {
    for (auto s : resultSamps)
    {
      ofile << s.real() << ", " << s.imag() << "\n";
    }
    ofile.flush();
  }
}


int main(int argc, char* argv[])
{
  doTest(Signals::Fft(1024, 128*1024), 5.0, 128*1024, 50, true, true);
  doTest(Signals::Fft(4096, 128*1024), 5.0, 128*1024, 25, true, true);
  doTest(Signals::Fft(16384, 128*1024), 5.0, 128*1024, 10, true, true);
}
