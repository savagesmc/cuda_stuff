// Compile with:
// nvcc --std=c++11 fft_stream.cu -o fft_stream -lcufft

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cufft.h>

#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Print file name, line number, and error code when a CUDA error occurs.
#define check_cuda_errors(val)  __check_cuda_errors__ ( (val), #val, __FILE__, __LINE__ )

template <typename T>
inline void __check_cuda_errors__(T code, const char *func, const char *file, int line) {
    if (code) {
    std::cout << "CUDA error at "
          << file << ":" << line << std::endl
          << "error code: " << (unsigned int) code
          << " type: \""  << cudaGetErrorString(cudaGetLastError()) << "\"" << std::endl
          << "func: \"" << func << "\""
          << std::endl;
    cudaDeviceReset();
    exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {

    // Number of FFTs to compute.
    const int NUM_DATAs[] = {64*1024, 64*1024, 64*1024, 64*1024, 64*1024, 64*1024, 64*1024, 64*1024, 64*1024};

    // Length of each FFT.
    const int Ns[] = {128, 256, 512, 1024, 4096, 8192, 16636, 32768, 65536};

    // Number of GPU streams across which to distribute the FFTs.
    const int NUM_STREAMSs[] = { 256, 256, 256, 256, 256, 256, 256, 256, 256 };


    for (auto nn = 0; nn < 9; ++nn)
    {
       auto dbg = 0;

       auto NUM_DATA = NUM_DATAs[nn];
       auto N = Ns[nn];
       auto NUM_STREAMS = NUM_STREAMSs[nn];

       // Allocate and initialize host input data.
       float2 **h_in = new float2* [NUM_STREAMS];
       for (int ii = 0; ii < NUM_STREAMS; ii++) {
           check_cuda_errors(cudaHostAlloc((void**)&h_in[ii],N*sizeof(float2), cudaHostAllocWriteCombined | cudaHostAllocMapped));
           for (int jj = 0; jj < N; ++jj) {
               h_in[ii][jj].x = (float) 1.f;
               h_in[ii][jj].y = (float) 0.f;
           }
       }

       // cout << dbg++ << endl;

       // Allocate and initialize host output data.
       float2 **h_out = new float2* [NUM_STREAMS];
       for (int ii = 0; ii < NUM_STREAMS; ii++) {
           check_cuda_errors(cudaHostAlloc((void**)&h_out[ii],N*sizeof(float2), cudaHostAllocWriteCombined | cudaHostAllocMapped));
           for (int jj = 0; jj < N; ++jj) {
               h_out[ii][jj].x = 0.f;
               h_out[ii][jj].y = 0.f;
           }
       }

       // cout << dbg++ << endl;

       // Allocate pointers to device input and output arrays.
       float2 **d_in = new float2* [NUM_STREAMS];
       float2 **d_out = new float2* [NUM_STREAMS];

       // Allocate input and output arrays on device.
       for (int ii = 0; ii < NUM_STREAMS; ii++) {
           check_cuda_errors(cudaHostGetDevicePointer((void**)&d_in[ii], h_in[ii], 0));
           check_cuda_errors(cudaHostGetDevicePointer((void**)&d_out[ii], h_out[ii], 0));
       }

       // cout << dbg++ << endl;

       // Create CUDA streams.
       cudaStream_t streams[NUM_STREAMS];
       for (int ii = 0; ii < NUM_STREAMS; ii++) {
           check_cuda_errors(cudaStreamCreate(&streams[ii]));
       }

       // cout << dbg++ << endl;

       // Creates cuFFT plans and sets them in streams
       cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS);
       for (int ii = 0; ii < NUM_STREAMS; ii++) {
           cufftPlan1d(&plans[ii], N, CUFFT_C2C, 1);
           cufftSetStream(plans[ii], streams[ii]);
       }

       // cout << dbg++ << endl;

       steady_clock::time_point before = steady_clock::now();

       // Fill streams with async memcopies and FFTs.
       for (int ii = 0; ii < NUM_DATA; ii++) {
           int jj = ii % NUM_STREAMS;
           check_cuda_errors(cudaMemcpyAsync(d_in[jj], h_in[jj], N*sizeof(float2), cudaMemcpyHostToDevice, streams[jj]));
           cufftExecC2C(plans[jj], (cufftComplex*)d_in[jj], (cufftComplex*)d_out[jj], CUFFT_FORWARD);
           check_cuda_errors(cudaMemcpyAsync(h_out[jj], d_out[jj], N*sizeof(float2), cudaMemcpyDeviceToHost, streams[jj]));
       }

       // cout << dbg++ << endl;

       // Wait for calculations to complete.
       for(int ii = 0; ii < NUM_STREAMS; ii++) {
           check_cuda_errors(cudaStreamSynchronize(streams[ii]));
       }

       // cout << dbg++ << endl;

       steady_clock::time_point after = steady_clock::now();

       // Free memory and streams.
       for (int ii = 0; ii < NUM_STREAMS; ii++) {
           check_cuda_errors(cudaFreeHost(h_in[ii]));
           // cout << ii << " " << dbg++ << endl;
           check_cuda_errors(cudaFreeHost(h_out[ii]));
           // cout << ii << " " << dbg++ << endl;
           /* delete[] h_in[ii]; */
           /* cout << ii << " " << dbg++ << endl; */
           /* delete[] h_out[ii]; */
           /* cout << ii << " " << dbg++ << endl; */
           check_cuda_errors(cudaStreamDestroy(streams[ii]));
           // cout << ii << " " << dbg++ << endl;
       }

       // cout << dbg++ << endl;

       delete plans;

       // cout << dbg++ << endl;

       cudaDeviceReset();

       // cout << dbg++ << endl;

       auto totalTime = duration<double>(after - before).count();
       auto timePer = totalTime / NUM_DATA;
       auto sampPerSec = N / timePer;

       cout << "===================================================" << endl;
       cout << "NUM_DATA:       " << NUM_DATA << endl;
       cout << "NUM_STREAMS:    " << NUM_STREAMS << endl;
       cout << "N:              " << N << endl;
       cout << "Total Time:     " << totalTime << endl;
       cout << "Time Per FFT:   " << timePer << endl;
       cout << "Samps Per Sec:  " << sampPerSec << endl;
    }

    return 0;
}
