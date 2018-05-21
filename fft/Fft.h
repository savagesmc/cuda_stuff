#ifndef SIGNALS_FFT_H
#define SIGNALS_FFT_H
#pragma once

#include <vector>
#include <complex>
#include <memory>
#include "FftTypes.h"

namespace Signals
{
   class Fft
   {
   public:

      Fft(int blockSz);
      ~Fft();

      void debug(bool enableDisable);

      void submit(std::vector<Complex> samples);

      // TODO: Replace with futures
      std::vector<Complex> result();

   private:
      class Impl;
      std::shared_ptr<Impl> impl_;
   };
}

#endif
