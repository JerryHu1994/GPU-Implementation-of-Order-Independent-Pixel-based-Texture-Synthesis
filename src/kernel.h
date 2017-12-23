/*
	kernel.h defines a set of cuda kernel function for the purpose of texture synthesis.
	Author: Jieru Hu
*/

#ifndef KERNEL_H__
#define KERNEL_H__

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Texture.cpp"

#define WHILE_PIXEL 256
#define BLACK_PIXEL 0

/*
	Initialize the curandState in each cuda kernel.
*/
__global__ void setup_kernel(curandState *state, unsigned long seed, int N);

/*
	Randomize the pixel value to either black or white given the ratio from the input texture.
*/
__global__ void random_Initialization(int* initalTex, int N, double ratio, curandState* state);

/*
	Randomize an array for selecting the best candidate pixels, ranging from 0 to 5.
*/
__global__ void random_PixelSelection(Texture randomTex, int N, curandState* state);
/*
	Apply the bilinear method to upscale the size of the image.
*/
__global__ void upscaleImage(Texture inputTex, Texture outputTex);

/*
	The kernel function does computing SSD for each candidate pixels and perform pixel selection.
*/
__global__ void convolutionalKernel(Texture d_pool, Texture d_from, Texture d_to, int d_random, int neighSize);

/*
	Perform a box filter on the final texture to remove the noise.
*/
__global__ void boxFilter(Texture from, Texture to, int matrixSize, int filterSize);

/*
	Apply the texture synthesis given the input pyramids, sizes, number of pyramids, neighborhood size, initial pixel ratio, etc.
*/
void synthesis(Texture** imagePyramids, Texture outputArr, int inputSize, int outputSize, int numPyramids, int neighSize, float sizeRatio, int ite, float initialRatio);

#endif
