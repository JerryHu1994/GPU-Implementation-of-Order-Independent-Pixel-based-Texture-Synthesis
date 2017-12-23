/* 
	This class implements the functions associated with CUDA Timer.
	Author: Jieru Hu
*/
#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  //start timing
  void startTimer()
  {
    cudaEventRecord(start, 0);
  }
  //stop timing
  void stopTimer()
  {
    cudaEventRecord(stop, 0);
  }
  //returns the elased time 
  float elapsedTime()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
