/* 
	This is the main class of GPU_based texture synthesis.
	Author: Jieru Hu
*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "timer.h"
#include <string>
#include <stdio.h>
#include <algorithm>
#include "utils.h"
#include "Texture.cpp"
#include "kernel.h"


using namespace std;

int main(int argc, char **argv) {
	
	if (argc != 7) {
		cerr << "usage: ./ts_gpu [image name] [output size] [# pyramid level] [neighborhood size] [size ratio] [#iterations per level]" << endl;
		exit(1);
	}
	string inputFile = argv[1];//name of the input image
	int outputSize = atoi(argv[2]);//size of the output graph
	int numPyramids = atoi(argv[3]);//number of pyramid levels
	int neighSize = atoi(argv[4]);//neighborhood size
	float sizeRatio = atof(argv[5]);//the size ratio between each each level of pyramid
	int ite = atoi(argv[6]);//iterations of synthesis on each pyramid level

	//outputs the synthesis parameters
	cout << "Output Size: " << outputSize << endl << "Number of levels: " << numPyramids << endl << "Neighborhood Size: " << neighSize << endl << "Size Ratio between levels" << sizeRatio << endl << endl << "Iterations: " << ite <<endl;
	//Read the input texture
	cv::Mat inputTex;
	inputTex = readImage(inputFile);
	//outputs the input texture dimensions
	cout << "Input Texture height : " << inputTex.rows << " columns : " << inputTex.cols << endl;
	int size  = inputTex.rows*inputTex.cols;
	int* inputArr = flattenImage(inputTex,size);
	float initialRatio = countRatio(inputArr, size);
	cout << "Initial Ratio: " << initialRatio << endl;

	//Create an array of pointers, with each pointer points to the a level of pyramids
	Texture** imagePyramids = constructPyramids(inputTex, sizeRatio, numPyramids, neighSize);

	Texture outputTex = Texture(outputSize, outputSize, 1);
	int inputSize = inputTex.rows;

	//start timing the GPU part
	GpuTimer timer;
	timer.startTimer();

	synthesis(imagePyramids, outputTex, inputSize, outputSize, numPyramids, neighSize, sizeRatio, ite, initialRatio);
	
	//end timing the GPU part
	timer.stopTimer();

	cout << "Total time spent: " << timer.elapsedTime() << endl;

	//displayIntImage(inputArr, inputSize, inputSize);

    //save the output texture
    string fileName = "output.jpg";

    writeImage(fileName, &outputTex);

    //free the pyramids of textures
    freePyramids(imagePyramids, numPyramids);
	
	return 0;
	
}
	
