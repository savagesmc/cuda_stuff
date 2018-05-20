#include "gpufft.h"

#include <memory>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "CudaUtils.h"
#include "TimeStat.h"

using namespace std;

namespace
{
   typedef float2 GpuComplex;
}

namespace GpuUtils
{

class FftEngine::Impl
{
   bool debug_;
   int devMemSize_;
   GpuComplex *devMem_;
   cufftHandle plan_;
public:
   Impl(int size)
   : debug_(false)
     , devMemSize_(sizeof(GpuComplex) * size)
     , devMem_(nullptr)
   {
      checkCudaErrors(cudaMalloc((void **)&devMem_, devMemSize_));
      checkCudaErrors(cufftPlan1d(&plan_, size, CUFFT_C2C, 1));
   }

   ~Impl()
   {
      checkCudaErrors(cufftDestroy(plan_));
      checkCudaErrors(cudaFree(devMem_));
   }

   void debug(bool enableDisable)
   {
      debug_ = enableDisable;
   }

   void operator()(std::vector<Complex>& samples)
   {
      checkCudaErrors(cudaMemcpy(devMem_, &samples[0], devMemSize_, cudaMemcpyHostToDevice));
      checkCudaErrors(cufftExecC2C(plan_, (cufftComplex *)devMem_, (cufftComplex *)devMem_, CUFFT_FORWARD));
      checkCudaErrors(cudaMemcpy(&samples[0], devMem_, devMemSize_, cudaMemcpyDeviceToHost));
   }
};

FftEngine::FftEngine(int size)
: impl_(new FftEngine::Impl(size))
{ }

FftEngine::~FftEngine()
{ }

void FftEngine::debug(bool enableDisable)
{
   impl_->debug(enableDisable);
}

void FftEngine::operator()(std::vector<Complex>& samples)
{
   impl_->operator()(samples);
}

}
