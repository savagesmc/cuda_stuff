/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include "gpufft.h"

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "CudaUtils.h"
#include "TimeStat.h"

using namespace std;

namespace GpuUtils
{

typedef float2 Complex;

void fft(vector<MyComplex>& samples, bool debug)
{
   const int mem_size = sizeof(MyComplex) * samples.size();

   const int numIter = (debug) ? 1 : 0;

   Complex *d_signal;
   {
      TimeStat("              malloc:", numIter);
      checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
   }
   {
      TimeStat("    memcpy to device:", numIter);
      checkCudaErrors(cudaMemcpy(d_signal, &samples[0], mem_size, cudaMemcpyHostToDevice));
   }

   // CUFFT plan simple API
   cufftHandle plan;
   {
      TimeStat("               cufft:", numIter);
      checkCudaErrors(cufftPlan1d(&plan, samples.size(), CUFFT_C2C, 1));
      checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));
   }

   {
      TimeStat("  memcpy from device:", numIter);
      checkCudaErrors(cudaMemcpy(&samples[0], d_signal, mem_size, cudaMemcpyDeviceToHost));
   }

   {
      TimeStat("  destroy fft plan", numIter);
      checkCudaErrors(cufftDestroy(plan));
   }

   // Deallocate
   {
      TimeStat("  free device memory", numIter);
      checkCudaErrors(cudaFree(d_signal));
   }
}

}
