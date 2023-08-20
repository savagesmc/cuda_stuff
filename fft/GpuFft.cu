#include "GpuFft.h"

#include <memory>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "CudaUtils.h"
#include "TimeStat.h"

using namespace std;
using namespace std::chrono;

namespace
{
typedef float2 GpuComplex;
constexpr int MaxBlock{4096*1024}; // 4 Megs
}

namespace GpuUtils
{

struct CompleteInfo
{
  void* engine;
  std::promise<ComplexVec>* promise;
};


void CUDART_CB cudaOperationComplete(void* userData);

class FftEngine::Impl
{
  bool debug_{false};
  bool busy_{false};
  int blockSize_;
  int devMemSize_;
  cudaStream_t stream_{};
  cufftHandle plan_;
  GpuComplex* devMemIn_;
  GpuComplex* devMemOut_;
  std::array<Complex, MaxBlock> blockIn_;
  std::array<Complex, MaxBlock> blockOut_;
public:

  Impl(int fftSize, int blockSize, bool debug)
  : debug_(debug),
    blockSize_(blockSize),
    devMemSize_(sizeof(GpuComplex) * blockSize),
    devMemIn_(nullptr),
    devMemOut_(nullptr)
  {
    if (blockSize > MaxBlock)
    {
      throw std::runtime_error("BlockSize > FftEngine MaxBlock size");
    }
    if (blockSize % fftSize)
    {
      throw std::runtime_error("BlockSize not a multiple of FftSize");
    }
    checkCudaErrors(cudaHostRegister(&blockIn_[0], MaxBlock*sizeof(Complex), cudaHostRegisterPortable));
    checkCudaErrors(cudaHostRegister(&blockOut_[0], MaxBlock*sizeof(Complex), cudaHostRegisterPortable));
    checkCudaErrors(cudaMalloc((void**)&devMemIn_, devMemSize_));
    checkCudaErrors(cudaMalloc((void**)&devMemOut_, devMemSize_));
    checkCudaErrors(cudaStreamCreate(&stream_));
    checkCudaErrors(cufftPlan1d(&plan_, fftSize, CUFFT_C2C, blockSize/fftSize));
    checkCudaErrors(cufftSetStream(plan_, stream_));
  }

  ~Impl()
  {
    checkCudaErrors(cudaStreamDestroy(stream_));
    checkCudaErrors(cufftDestroy(plan_));
    checkCudaErrors(cudaFree(devMemIn_));
    checkCudaErrors(cudaFree(devMemOut_));
    checkCudaErrors(cudaHostUnregister(&blockIn_[0]));
    checkCudaErrors(cudaHostUnregister(&blockOut_[0]));
  }

  std::future<ComplexVec> dofft(ComplexVec& samples)
  {
    if (samples.size() != blockSize_)
    {
      throw std::runtime_error("BlockSize not a multiple of FftSize");
    }
    int hostSize = samples.size() * sizeof(Complex);
    std::copy(samples.begin(), samples.end(), blockIn_.begin()); // copy into host block
    checkCudaErrors(cudaMemcpyAsync(devMemIn_, &blockIn_[0], devMemSize_, cudaMemcpyHostToDevice, stream_));
    cufftExecC2C(plan_, (cufftComplex*)devMemIn_, (cufftComplex*)devMemOut_, CUFFT_FORWARD);
    checkCudaErrors(cudaMemcpyAsync(&blockOut_[0], devMemOut_, devMemSize_, cudaMemcpyDeviceToHost, stream_));
    std::promise<ComplexVec>* p = new std::promise<ComplexVec>();
    CompleteInfo* info = new CompleteInfo;
    info->engine = this;
    info->promise = p;
    checkCudaErrors(cudaLaunchHostFunc(stream_, cudaOperationComplete, info));
    busy_ = true;
    return p->get_future();
  }

  void operationComplete(std::promise<ComplexVec>* promise) {
    ComplexVec samples(blockSize_);
    std::copy(blockOut_.begin(), blockOut_.begin()+blockSize_, samples.begin()); // copy back out of host block
    promise->set_value(std::move(samples));
    delete promise;
    busy_ = false;
  }

  bool busy() const {
    return busy_;
  }
};

FftEngine::FftEngine(int fftSize, int blockSize, bool debug)
  : impl_(new FftEngine::Impl(fftSize, blockSize, debug))
{ }

FftEngine::~FftEngine()
{
}

std::future<ComplexVec> FftEngine::dofft(ComplexVec& samples)
{
  return impl_->dofft(samples);
}

bool FftEngine::busy() const {
  return impl_->busy();
}

void CUDART_CB cudaOperationComplete(void* userData)
{
  CompleteInfo* info = static_cast<CompleteInfo*>(userData);
  FftEngine::Impl* engine = static_cast<FftEngine::Impl*>(info->engine);
  engine->operationComplete(info->promise);
  delete info;
}

}
