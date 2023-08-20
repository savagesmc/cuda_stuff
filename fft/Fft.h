#ifndef SIGNALS_FFT_H
#define SIGNALS_FFT_H
#pragma once

#include <vector>
#include <complex>
#include <memory>
#include <future>
#include "FftTypes.h"

namespace Signals
{
class Fft
{
public:
  Fft(int fftSize, int blockSz, bool debug=false);
  ~Fft();
  std::future<ComplexVec> submit(ComplexVec& samples);
  int fftSize() const;
  bool busy() const;
private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}

#endif
