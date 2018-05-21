#include "Fft.h"
#include <vector>
#include <complex>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <list>
#include <chrono>
#include "FftTypes.h"
#include "GpuFft.h"
#include "Queue.h"
#include <boost/optional.hpp>

using namespace std;

using boost::optional;
using boost::none;

namespace Signals
{

typedef Util::Queue<vector<Complex>> Queue;
using GpuUtils::FftEngine;

class FftThread
{
   bool exit_;
   Queue& inQ_;
   Queue& outQ_;
   std::thread thd_;
   FftEngine eng_;
public:

   FftThread(int sz, Queue& in, Queue& out)
   : exit_(false)
   , inQ_(in)
   , outQ_(out)
   , thd_(std::bind(&FftThread::exec, this))
   , eng_(sz)
   { }

   ~FftThread()
   {
      exit_ = true;
      thd_.join();
   }

   void exec()
   {
      while (!exit_)
      {
         auto v = inQ_.pop();
         if (v) {
            eng_(*v);
            outQ_.push(std::move(*v));
         }
      }
   }
};

class Fft::Impl
{
   Queue inQ_;
   Queue outQ_;
   const int numThreads_;
   std::vector<FftThread*> thdPool_;
public:
   Impl(int sz)
   : inQ_()
     , outQ_()
     , numThreads_(10)
     ,thdPool_(numThreads_)
   {
      for (int i = 0; i < numThreads_; ++i)
      {
         thdPool_[i] = new FftThread(sz, inQ_, outQ_);
      }
   }

   ~Impl()
   {
      for (int i = 0; i < numThreads_; ++i)
      {
         delete thdPool_[i];
      }
   }

   void debug(bool enableDisable)
   {
      //TODO
   }

   void submit(std::vector<Complex> samples)
   {
      inQ_.push(std::move(samples));
   }

   // TODO: Replace with futures
   std::vector<Complex> result()
   {
      auto v = outQ_.pop();
      while(!v) {
         v = outQ_.pop();
      }
      return *v;
   }
};

Fft::Fft(int sz) : impl_(new Impl(sz)) { }
Fft::~Fft() { }

void Fft::debug(bool enableDisable)
{
   impl_->debug(enableDisable);
}

void Fft::submit(std::vector<Complex> samples)
{
   impl_->submit(std::move(samples));
}

std::vector<Complex> Fft::result()
{
   return impl_->result();
}

}
