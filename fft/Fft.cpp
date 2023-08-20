#include "Fft.h"
#include <vector>
#include <complex>
#include <memory>
#include <thread>
#include <future>
#include <chrono>
#include "FftTypes.h"
#include "GpuFft.h"

#include <iostream>
#include <iomanip>
#include <string>

using namespace std;

namespace Signals
{

using GpuUtils::FftEngine;

struct Fft::Impl
{
  const int numThreads_;
  int fftSize_;
  int blkSize_;
  std::vector<std::unique_ptr<FftEngine>> engines_;
  Impl(int fftSz, int blkSz, bool debug)
  : numThreads_(10), fftSize_(fftSz), blkSize_(blkSz)
  {
    for (int i = 0; i < numThreads_; ++i)
    {
      engines_.push_back(std::make_unique<FftEngine>(fftSz, blkSz, debug));
    }
  }

  ~Impl()
  { }

  std::future<ComplexVec> submit(std::vector<Complex> samples)
  {
    for (std::vector<std::unique_ptr<FftEngine>>::iterator eng=engines_.begin();
         eng != engines_.end();
         ++eng)
    {
      if (!(*eng)->busy())
      {
        return (*eng)->dofft(samples);
      }
    }
    throw std::runtime_error("No available fft engine.");
  }

  bool busy() const
  {
    for (std::vector<std::unique_ptr<FftEngine>>::const_iterator eng=engines_.begin();
         eng != engines_.end();
         ++eng)
    {
      if (!(*eng)->busy())
      {
        return false;
      }
    }
    return true;
  }

};

Fft::Fft(int fftSz, int blkSz, bool debug) : impl_(new Impl(fftSz, blkSz, debug)) { }
Fft::~Fft() { }

std::future<ComplexVec> Fft::submit(ComplexVec& samples)
{
  return impl_->submit(samples);
}

int Fft::fftSize() const
{
  return impl_->fftSize_;
}

bool Fft::busy() const
{
  return impl_->busy();
}

}
