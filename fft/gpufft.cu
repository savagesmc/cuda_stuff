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

vector<vector<MyComplex> > fft(const vector<vector<MyComplex> >& in)
{
   vector<vector<MyComplex> > out;
   if (in.size() < 1)
      return out;

   const int numSamples = in[0].size();
   const int mem_size = sizeof(Complex) * numSamples * in.size();

   // Allocate device memory for signal
   Complex *h_signal = (Complex *)malloc(mem_size);
   Complex *h_ptr = h_signal;
   for (int i=0; i<in.size(); ++i)
   {
      vector<MyComplex>::const_iterator it = in[i].begin();
      vector<MyComplex>::const_iterator end = in[i].end();
      for (; it != end; ++it)
      {
         h_ptr->x = it->real();
         h_ptr->y = it->imag();
         ++h_ptr;
      }
   }

   Complex *d_signal;
   Complex *d_signal2;
   {
      TimeStat("              malloc:");
      checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
      checkCudaErrors(cudaMalloc((void **)&d_signal2, mem_size));
   }
   {
      TimeStat("    memcpy to device:");
      checkCudaErrors(cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));
   }

   // CUFFT plan simple API
   cufftHandle plan;
   {
      TimeStat("               cufft:");
      checkCudaErrors(cufftPlan1d(&plan, numSamples, CUFFT_C2C, in.size()));
      checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal2, CUFFT_FORWARD));
   }

   {
      TimeStat("  memcpy from device:");
      checkCudaErrors(cudaMemcpy(h_signal, d_signal2, mem_size, cudaMemcpyDeviceToHost));
   }

   h_ptr = h_signal;
   for (int i=0; i<in.size(); ++i)
   {
      out.push_back(vector<MyComplex>(numSamples));
      vector<MyComplex>::iterator oit = out[i].begin();
      vector<MyComplex>::iterator oend = out[i].end();
      for (; oit != oend; ++oit)
      {
         *oit = MyComplex(h_ptr->x, h_ptr->y);
         ++h_ptr;
      }
   }

   // Deallocate
   checkCudaErrors(cufftDestroy(plan));
   free(h_signal);
   checkCudaErrors(cudaFree(d_signal));
   checkCudaErrors(cudaFree(d_signal2));

   return out;
}

}
