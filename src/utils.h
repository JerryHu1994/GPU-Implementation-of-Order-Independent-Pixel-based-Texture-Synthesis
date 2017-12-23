/*
	Utils.h defines a set of helper functions to process the images.
	Author: Jieru Hu
*/
#ifndef UTILS_H__
#define UTILS_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include "Texture.cpp"

using namespace std;

/*
  Reads the image into cv::Mat format, by the input texture image file name.
*/
cv::Mat readImage(const std::string &fileName);

/*
  Refactor the cv::Mat image into a one dimentional int array.
*/
int* flattenImage(cv::Mat image, int size);

/*
  Resize the image given the resize factor.
*/
cv::Mat resizeImage(cv::Mat image, int newRows, int newCols);

/*
  Write the image to output file.
*/
void writeImage(const string &fileName, Texture* outImage);

/*
  Convert an integer array to a float array.
*/
float* intToFloatArray(int* intArr, int size);

/*
	Count the ratio of the black versus white pixels in the given 1-D image array.
*/
double countRatio(int* image, int size);

/*
	Display the texture image with CV library. 
*/
void displayIntImage(Texture* texture);

/*
	Generate an array random initial pixels given the size and ratio. 
*/
int* randomInitial(int size, float ratio);

/*
	Resize the initial input texture into different size texture images, and store them into a list of Textures.
*/
Texture** constructPyramids(cv::Mat inputTex, float sizeRatio, int numPyramids, int neighSize);

/*
	Free the list of textures. 
*/
void freePyramids(Texture** textureList, int numPyramids);

/*
	Geenerate an array of random indexes given the array size.	
*/
int* randomIndex(int size);

#endif
