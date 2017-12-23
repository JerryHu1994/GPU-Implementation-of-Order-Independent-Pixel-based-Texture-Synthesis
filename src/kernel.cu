#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernel.h"
#include "utils.h"
#include "timer.h"
#include <string>

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024
#define CONVOLUTIONAL_ROWS 32
#define CONVOLUTIONAL_COLS 32

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

using namespace std;

__global__ void setup_kernel(curandState *state, unsigned long seed, int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N * N) {
    	curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void random_Initialization(Texture initalTex, int N, double ratio, curandState* state)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N * N) {
		curandState localState = state[idx];
		int random = (int)(curand(&localState) % 100);
		initalTex.imgArray[idx]= ((double)random/100 < ratio) ? BLACK_PIXEL : WHILE_PIXEL;
	}
}

__global__ void random_PixelSelection(Texture randomTex, int N, curandState* state)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N * N) {
		curandState localState = state[idx];
		int random = (int)(curand(&localState) % 5);
		randomTex.imgArray[idx]= random;
	}
}

__global__ void upscaleImage(Texture inputTex, Texture outputTex)
{
	//2D index of the current thread in the output image
	const int outputX = blockIdx.x * blockDim.x + threadIdx.x;
	const int outputY = blockIdx.y * blockDim.y + threadIdx.y;
	
	int inputWidth = inputTex.cols;
	int inputHeight = inputTex.rows; 
	int outputWidth =  outputTex.cols;
	int outputHeight = outputTex.rows;
	int* input = inputTex.imgArray;
	int* output = outputTex.imgArray;

	//only perform calculation on thread in valid range
	if (outputX < outputWidth && outputY < outputHeight) {
		float x_ratio = ((float)(inputWidth - 1)) / outputWidth;
		float y_ratio = ((float)(inputHeight - 1)) / outputHeight;
		
		int x = (int) (x_ratio * outputX);
		int y = (int) (y_ratio * outputY);
		float x_diff = (x_ratio * outputX) - x;
		float y_diff = (y_ratio * outputY) - y;
	
		int index = y*inputWidth +x;
		int A = input[index];
		int B = input[index + 1];
		int C = input[index + inputWidth];
		int D = input[index + inputWidth + 1];
		
		int result = (int)(A * (1 - x_diff) * (1 - y_diff) + B * x_diff * (1 - y_diff) + C * y_diff * (1 - x_diff) + D * x_diff * y_diff);
		
		int ret = (result >= 128) ? WHILE_PIXEL : BLACK_PIXEL;
		
		output[outputY * outputWidth + outputX] = ret;
	}
}

__global__ void convolutionalKernel(Texture d_pool, Texture d_from, Texture d_to, Texture d_random, int neighSize)
{
	//calculate thread indexes and load texture sizes
	int bidX = blockIdx.x, bidY = blockIdx.y, tidX = threadIdx.x, tidY = threadIdx.y;
	int xIndex = bidX * blockDim.x + tidX;
	int yIndex = bidY * blockDim.y + tidY;
	int threadIdx = tidY * blockDim.x + tidX;
	
	int poolSize = d_pool.rows;
	int fromSize = d_from.rows;
	int toSize = d_to.rows;
	int index = yIndex * toSize + xIndex;

	//calculate how many pixels each thread needs to load
	int unitLoad = poolSize * poolSize / (blockDim.x * blockDim.y) + 1;
	
	__shared__ int pixelPool[110][110];
	//load the candidate pixels into the shared memory
	int load;
	for (load = 0; load < unitLoad; load++) {
		int poolIdx = unitLoad * threadIdx + load;
		if(poolIdx >= poolSize * poolSize)	break;
		pixelPool[poolIdx / poolSize][poolIdx % poolSize] = d_pool.imgArray[poolIdx];
	}
	//synchronize to make sure all candidates are loaded
	__syncthreads();

	int offset = (neighSize - 1)/2;
	
	if (xIndex < toSize && yIndex < toSize) {
				
	//calculate the center index in d_from
	int xCenter = xIndex + offset;
	int yCenter = yIndex + offset;

	//the ssdMap keeps track of the best SSD found, and the bestX and bestY records the best pixel position
	int ssdMap[5];
	int bestX[5];
	int bestY[5];
	int currMapSize = 0;
	int z;
	for(z = 0;z < 5;z++)	ssdMap[z] = 100000000000000;
	
	//calculate the SSD for each target pixel
	int a,b,i,j;
	
	//loop through the candidate pool
	for (a = offset; a <= poolSize-1-offset; a++) {
		for (b = offset; b <= poolSize-1-offset; b++) {
			int currSum = 0;
			for (j = -offset; j <= offset; j++) {
				for(i = -offset; i <= offset; i++) {
					int diff = pixelPool[a+j][b+i] - d_from.imgArray[(yCenter+j) * fromSize + xCenter + i];				
					currSum += diff * diff;		
				}
			}

			int add = 0;
			//maintain a sorted ssdMap for the best candidate pixels
			int currPtr = 0;
			while (currPtr < currMapSize) {
				if (currSum < ssdMap[currPtr]) {
					//shift everything from currPtr to the end
					int j;
					int temp = ssdMap[currPtr], tempx = bestX[currPtr], tempy = bestX[currPtr];
					for (j = currPtr+1; j < 5; j++) {
						int now = temp, nowx = tempx, nowy = tempy;
						temp = ssdMap[j];
						tempx = bestX[j];
						tempy = bestY[j];
						ssdMap[j] = now;
						bestX[j] = nowx;
						bestY[j] = nowy;				
					}
					ssdMap[currPtr] = currSum;
					bestX[currPtr] = b;
					bestY[currPtr] = a;
					if (currMapSize < 5)	currMapSize++;
					add = 1;					
					break;
				}
				currPtr++;
			}
			//there is still empty spot, append to the end
			if (currPtr < 5 && !add)
			{
				ssdMap[currPtr] = currSum;
				bestX[currPtr] = b;
				bestY[currPtr] = a;
 				if (currMapSize < 5)	currMapSize++; 
			}
	
		}
	}
	//pick the best pixel from the candidate pool and fill in the new output
	int random = d_random.imgArray[index];
	int bestIdx = bestY[random] * poolSize + bestX[random];
	d_to.imgArray[index] = d_pool.imgArray[bestIdx];
	}
}

__global__ void boxFilter(Texture from, Texture to, int matrixSize, int filterSize)
{
	//get indexes for threads
	int bidX = blockIdx.x, bidY = blockIdx.y, tidX = threadIdx.x, tidY = threadIdx.y;
	int xIndex = bidX*blockDim.x + tidX;
	int yIndex = bidY*blockDim.y + tidY;
	
	//make sure the index is in the valid range
	if (xIndex < matrixSize && yIndex < matrixSize) {
		//each thread performs a filter for a pixel
		int i,j;
		int outputValue = 0, count = 0;
		int offset = (filterSize-1)/2;
		//filter the current pixel
		for (j = -offset;j <= offset; j++) {
			for (i = -offset;i <= offset; i++) {
				int currIdx = (yIndex + j)*matrixSize + xIndex + i;	
				//make sure the currIdx is in the range			
				if(currIdx >= 0 && currIdx < matrixSize * matrixSize){
					outputValue += from.imgArray[currIdx];
					count ++;
				}		
			}
		}
		float average = (float)outputValue/(float)count;
		to.imgArray[yIndex * matrixSize + xIndex] = (average > 128) ? WHILE_PIXEL : BLACK_PIXEL;
	}
}

void synthesis(Texture** imagePyramids, Texture outputArr, int inputSize, int outputSize, int numPyramids, int neighSize, float sizeRatio, int iterations, float initialRatio)
{
	//calculate sizes
	int initialSize = outputSize / pow(sizeRatio, numPyramids-1) + neighSize - 1;
	cout << "The initial Size is " << initialSize << endl;
	
	Texture d_return = Texture(outputSize, outputSize, 0);
	//allocate the initial texture
	Texture d_main = Texture(initialSize, initialSize, 0);
	CudaSafeCall(cudaMalloc((void**)&(d_main.imgArray), sizeof(int) * initialSize * initialSize));
	
	//allocate memory for curandStates
	curandState* devStates;

  	CudaSafeCall(cudaMalloc((void**)&devStates, initialSize * initialSize * sizeof(curandState)));
    //setup the random seed
	srand(time(NULL));    
	int seed = rand();
	
    cout << "Start seeding..."<<endl;
  	setup_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(devStates, seed, initialSize);
  	CudaCheckError();
  	//start randomize the inital texture
	cout << "Start randomize the inital texture..."<<endl;
	random_Initialization<<<GRID_SIZE, BLOCK_SIZE>>>(d_main, initialSize, initialRatio, devStates);
	CudaCheckError();
		
	int level;
	//iterature through each level of synthesis prramids
	for (level = 0; level < numPyramids; level++) {
		
		cout << "Start synthesizing the " << level << " level" << endl;
		
		//allocate the output texture for the current level (hosting all the picked pixels), can be used repeatedly for multiple iterations
		int currOutputSize = outputSize / pow(sizeRatio, numPyramids - level - 1); 
		Texture d_output = Texture(currOutputSize, currOutputSize, 0);
		CudaSafeCall(cudaMalloc((void**)&d_output.imgArray, sizeof(int) * currOutputSize*currOutputSize));
		
		//allocate the device memory for candidate pixels in current level
		Texture d_pool = Texture(imagePyramids[level]->rows, imagePyramids[level]->cols, 0);
		int poolSize = imagePyramids[level]->cols*imagePyramids[level]->rows*sizeof(int);
		CudaSafeCall(cudaMalloc((void**)&(d_pool.imgArray), poolSize));
			
		//transfer the candidate pixels from host memory to device memory
		CudaSafeCall(cudaMemcpy(d_pool.imgArray, imagePyramids[level]->imgArray, poolSize, cudaMemcpyHostToDevice));
		
		GpuTimer timer;
		
		//on each pyramid level, perform texture synthesis several iterations
		int ite;
		for (ite = 0; ite < iterations; ite++) {
			//compute the convolutional kernel to pick new candidate
			int gridRows = (currOutputSize + CONVOLUTIONAL_ROWS - 1) / CONVOLUTIONAL_ROWS;
			int gridCols = (currOutputSize + CONVOLUTIONAL_COLS - 1) / CONVOLUTIONAL_COLS;
			dim3 dimConvGrid(gridRows, gridCols, 1);
			dim3 dimConvBlock(CONVOLUTIONAL_ROWS, CONVOLUTIONAL_COLS, 1);

			//setup the random seed
			srand(time(NULL));    
			int candidateSeed = rand();
			
			//allocate array for random pixel selection
			curandState* devCandidateStates;
  			CudaSafeCall(cudaMalloc((void**)&devCandidateStates, currOutputSize * currOutputSize * sizeof(curandState)));
			Texture d_random = Texture(currOutputSize,currOutputSize,0);
			CudaSafeCall(cudaMalloc((void**)&d_random.imgArray, sizeof(int) * currOutputSize * currOutputSize));
			//generate the random indexes for the candidate pixel, range from 0-5
			cout << "Start seeding for random candidates..."<<endl;
  			setup_kernel<<<dimConvGrid, dimConvBlock>>>(devCandidateStates, candidateSeed, currOutputSize);
			CudaCheckError();
			cout << "Start randomize the candidate pixel selection..." << endl;
			random_PixelSelection<<<dimConvGrid, dimConvBlock>>>(d_random, currOutputSize, devCandidateStates);
			CudaCheckError();
			
			timer.startTimer();
			int currSize = outputSize / pow(sizeRatio, numPyramids - 1 - level);
			
			//launch the main kernel for texture synthesis
			cout << "Perform synthesizing at level " << level << " iteration " << ite << " Pixel pool size: " << d_pool.rows <<": Input size: " << d_main.rows << " output size: " << d_output.rows<<endl;
			cout << "GridSize :" << gridRows << " " << gridCols<<endl;
			convolutionalKernel<<<dimConvGrid, dimConvBlock>>>(d_pool, d_main, d_output, d_random, neighSize);
			CudaCheckError();
			
			timer.stopTimer();
			cout << "Time spent: " << timer.elapsedTime() << endl << endl;
			
			//after each iteration, resize the image to slightly larger size with boundary neighborhood fiiled in
			//here we use the d_main as the output texture for the rescale process, and thus keeping the d_output as 
			//the output matrix for next synthesis process
			if (ite == iterations - 1) {
				break;
			}
			int rescaleSize = currSize + neighSize - 1;
			int rescaleGridRows = (rescaleSize + CONVOLUTIONAL_ROWS - 1) / CONVOLUTIONAL_ROWS;
			int rescaleGridCols = (rescaleSize + CONVOLUTIONAL_COLS - 1) / CONVOLUTIONAL_COLS;
			dim3 dimRescaleGrid(rescaleGridRows, rescaleGridCols);
			dim3 dimRescaleBlock(CONVOLUTIONAL_ROWS, CONVOLUTIONAL_COLS, 1);
			cout << "Upscaling the texture to next level: Original Size: " << d_output.rows << " After Size: " << d_main.rows << endl << endl;
			upscaleImage<<<dimRescaleGrid,dimRescaleBlock>>>(d_output, d_main);
			CudaCheckError();
		}	
	
		//upscale the size of the image base on the given sizeRatio to the next pyramid level
		if (level != numPyramids - 1) { // skip if it is the last level

			//uncomment following lines to code to save the intermediate synthesis results			
			/*			
			//save the current texture to output
			Texture saveTex = Texture(currOutputSize, currOutputSize, 1);
			CudaSafeCall(cudaMemcpy(saveTex.imgArray, d_output.imgArray, sizeof(int)*currOutputSize*currOutputSize, cudaMemcpyDeviceToHost));
			ostringstream ss;
			ss << level;			
			string str = ss.str() + "out.jpg";
			writeImage(str, &saveTex);
			*/

			//allocate the texture for next level
			int nextSize = outputSize / (pow(sizeRatio, numPyramids - level - 2)) + neighSize - 1;
			Texture d_nextLevel = Texture(nextSize, nextSize, 0);
			CudaSafeCall(cudaMalloc((void**)&d_nextLevel.imgArray, sizeof(int) * nextSize * nextSize));
			
			int levelRescaleRows = (nextSize + CONVOLUTIONAL_ROWS -1) / CONVOLUTIONAL_ROWS;
			int levelRescaleCols = (nextSize + CONVOLUTIONAL_COLS -1) / CONVOLUTIONAL_COLS;
			dim3 dimLevelGrid(levelRescaleRows, levelRescaleCols);
			dim3 dimLevelBlock(CONVOLUTIONAL_ROWS, CONVOLUTIONAL_COLS, 1);
			cout << "Upscaling the texture after level " << level << " From size " << d_output.rows << " to " << d_nextLevel.rows << endl;
			upscaleImage<<<dimLevelGrid, dimLevelBlock>>>(d_output, d_nextLevel);
			CudaCheckError();
				
			//free the output texture in this level
			CudaSafeCall(cudaFree(d_output.imgArray));
			//free the main texture in this level
			CudaSafeCall(cudaFree(d_main.imgArray));
			d_main.copyTexture(&d_nextLevel);
		} else {
			//at the last level of synthesis
			CudaSafeCall(cudaMalloc((void**)&d_return.imgArray, sizeof(int) * outputSize * outputSize));
			int gridRows = (outputSize + CONVOLUTIONAL_ROWS - 1) / CONVOLUTIONAL_ROWS;
			int gridCols = (outputSize + CONVOLUTIONAL_COLS - 1) / CONVOLUTIONAL_COLS;
			dim3 dimFilterGrid(gridRows, gridCols, 1);
			dim3 dimFilterBlock(CONVOLUTIONAL_ROWS, CONVOLUTIONAL_COLS, 1);
			cout << "Start filtering for the ouput" << endl;
			//filter the image, remove the noise pixels			
			boxFilter<<<dimFilterGrid, dimFilterBlock>>>(d_output, d_return, outputSize, neighSize);			
			CudaCheckError();	
			CudaSafeCall(cudaFree(d_output.imgArray));
		}
		
		//free the candidate pixels in this level
		CudaSafeCall(cudaFree(d_pool.imgArray));
	}
	
	//copy the image back to the host memory
	CudaSafeCall(cudaMemcpy(outputArr.imgArray, d_return.imgArray, sizeof(int) * outputSize * outputSize, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaFree(d_return.imgArray));
	return;
}



