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

void doFft(int N, int BATCH, int NUM_STREAMS, int NUM_PER_STREAM)
{
    auto NUM_SAMP = BATCH*N;

    // Initialize host input data (only need 1 copy)
    float2 h_in[NUM_SAMP];
    for (int jj = 0; jj < NUM_SAMP; ++jj) {
        h_in[jj].x = (float) 1.f;
        h_in[jj].y = (float) 0.f;
    }

    // Allocate and initialize host output data.
    float2 **h_out = new float2 *[NUM_STREAMS*NUM_PER_STREAM];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
       for (int jj = 0; jj < NUM_PER_STREAM; ++jj) {
          auto idx = ii*NUM_PER_STREAM + jj;
          h_out[idx] = new float2[NUM_SAMP];
          for (int kk=0; kk<NUM_SAMP; ++kk)
          {
             h_out[idx][kk].x = 0.f;
             h_out[idx][kk].y = 0.f;
          }
       }
    }

    // Pin host input and output memory for cudaMemcpyAsync.
    check_cuda_errors(cudaHostRegister(h_in, NUM_SAMP*sizeof(float2), cudaHostRegisterPortable));
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
       for (int jj = 0; jj < NUM_PER_STREAM; ++jj) {
          auto idx = ii*NUM_PER_STREAM + jj;
          check_cuda_errors(cudaHostRegister(h_out[idx], NUM_SAMP*sizeof(float2), cudaHostRegisterPortable));
       }
    }

    // Allocate pointers to device input and output arrays.
    float2 **d_in = new float2 *[NUM_STREAMS * NUM_PER_STREAM];
    float2 **d_out = new float2 *[NUM_STREAMS * NUM_PER_STREAM];

    // Allocate intput and output arrays on device.
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
       for (int jj = 0; jj < NUM_PER_STREAM; ++jj) {
          auto idx = ii*NUM_PER_STREAM + jj;
          check_cuda_errors(cudaMalloc((void**)&d_in[idx], NUM_SAMP*sizeof(float2)));
          check_cuda_errors(cudaMalloc((void**)&d_out[idx], NUM_SAMP*sizeof(float2)));
       }
    }

    // Create CUDA streams.
    cudaStream_t streams[NUM_STREAMS];
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        check_cuda_errors(cudaStreamCreate(&streams[ii]));
    }

    // Creates cuFFT plans and sets them in streams
    cufftHandle* plans = (cufftHandle*) malloc(sizeof(cufftHandle)*NUM_STREAMS);
    for (int ii = 0; ii < NUM_STREAMS; ii++) {
        cufftPlan1d(&plans[ii], N, CUFFT_C2C, BATCH);
        cufftSetStream(plans[ii], streams[ii]);
    }

    steady_clock::time_point before = steady_clock::now();

    // Fill streams with async memcopies and FFTs.
    for (int ii = 0; ii < NUM_STREAMS * NUM_PER_STREAM; ii++) {
        int strmIdx = ii % NUM_STREAMS;
        check_cuda_errors(cudaMemcpyAsync(d_in[ii], h_in, NUM_SAMP*sizeof(float2), cudaMemcpyHostToDevice, streams[strmIdx]));
        cufftExecC2C(plans[strmIdx], (cufftComplex*)d_in[ii], (cufftComplex*)d_out[ii], CUFFT_FORWARD);
        check_cuda_errors(cudaMemcpyAsync(h_out[ii], d_out[ii], NUM_SAMP*sizeof(float2), cudaMemcpyDeviceToHost, streams[strmIdx]));
    }

    // Wait for calculations to complete.
    for(int ii = 0; ii < NUM_STREAMS; ii++) {
        check_cuda_errors(cudaStreamSynchronize(streams[ii]));
    }

    steady_clock::time_point after = steady_clock::now();

    auto totalTime = duration<double>(after - before).count();
    auto totalFfts = NUM_STREAMS * NUM_PER_STREAM * BATCH;
    auto timePer = totalTime / totalFfts;
    auto sampPerSec = N / timePer;

    cout << "===================================================" << endl;
    cout << "N:                 " << N << endl;
    cout << "NUM_SAMP:          " << NUM_SAMP << endl;
    cout << "NUM_FFT_PER_BATCH: " << NUM_SAMP/N << endl;
    cout << "NUM_STREAMS:       " << NUM_STREAMS << endl;
    cout << "NUM_PER_STREAM:    " << NUM_PER_STREAM << endl;
    cout << "Total Time:        " << totalTime << endl;
    cout << "Time Per FFT:      " << timePer << endl;
    cout << "Samps Per Sec:     " << sampPerSec << endl;

    // Free memory and streams.
    check_cuda_errors(cudaHostUnregister(h_in));
    for (int ii = 0; ii < NUM_STREAMS*NUM_PER_STREAM; ii++) {
        check_cuda_errors(cudaHostUnregister(h_out[ii]));
        check_cuda_errors(cudaFree(d_in[ii]));
        check_cuda_errors(cudaFree(d_out[ii]));
        delete[] h_out[ii];
    }

    for (int ii = 0; ii < NUM_STREAMS; ii++) {
       check_cuda_errors(cudaStreamDestroy(streams[ii]));
    }

    delete plans;


}

int main(int argc, char *argv[]) {

    doFfts(1024, 256, 1, 1);
    doFfts(4096, 256, 1, 1);
    doFfts(8192, 256, 1, 1);

    cudaDeviceReset();

    return 0;
}
