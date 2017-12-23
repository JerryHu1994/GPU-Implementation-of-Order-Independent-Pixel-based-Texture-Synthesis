/*
	Utils.cpp impletments a set of functions defined in utils.h on image operations.
	Author: Jieru Hu
*/
#include "utils.h"

using namespace std;

cv::Mat readImage(const string &fileName)
{
	cv::Mat image = cv::imread(fileName.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Could not open the iput file: " << fileName << std::endl;
		exit(1);
	}
	//check if the image is continuous
	if (!image.isContinuous()) {
		cerr << "Image isn't continuous!" << endl;
		exit(1);
	}

	//convert the RGB image to greyscale image
	if (image.channels() != 1){
		cout << "Converting RGB image to greyscale image... " << endl;	
		cv::Mat imageGreyScale;
		cv::cvtColor(image, imageGreyScale, CV_BGR2GRAY);
		return imageGreyScale;
	}
	return image;
}

int* flattenImage(cv::Mat image, int size)
{
	int *imgArr;
	if((imgArr = (int *)malloc(sizeof(int) * size)) == NULL){
		cerr << "Failed to malloc image array"<< endl;
		exit(1);
	}
	unsigned char *cvPtr = image.ptr<unsigned char>(0);
	int i;
	for(i=0; i<size ;i++){
		
		if ((int) cvPtr[i] >= 128){
			imgArr[i] = 256;	
		} else {
			imgArr[i] = 0;
		}
	}
	return imgArr;	
}

double countRatio(int* image, int size)
{
	double white=0.0, black=0.0;
	int i;
	for(i=0;i<size;i++){
		if (image[i] == 256) {
			white++;
		} else {
			black++;
		}
	}
	return black/(white+black);
}

cv::Mat resizeImage(cv::Mat image, int newRows, int newCols)
{
	cv::Size newSize(newRows, newCols);
	cv::Mat dst;
	cv::resize(image, dst, newSize);
	return dst;
}

void writeImage(const string &fileName, Texture* outImage)
{
	int size = outImage->cols*outImage->rows;
	float* floatArr = intToFloatArray(outImage->imgArray, size);
	cv:: Mat outputImg(outImage->rows, outImage->cols, CV_32F, floatArr);
	cv::imwrite(fileName.c_str(), outputImg);

}

void writeImage(const string &fileName, float* outImage, size_t numRows, size_t numCols)
{	
	cv:: Mat outputImg(numRows, numCols, CV_32F, outImage);
	cv::imwrite(fileName.c_str(), outputImg);
}

float* intToFloatArray(int* intArr, int size)
{
	float *floatArr;
	int i;
	if((floatArr = (float *)malloc(sizeof(float)*size)) == NULL){
		cerr << "Failed to malloc image array"<< endl;
		exit(1);
	}
	for(i=0;i<size;i++)	floatArr[i] = (float) intArr[i];
	return floatArr;
}

void displayIntImage(Texture* texture)
{
	int size = texture->cols*texture->rows;
	float* floatArr = intToFloatArray(texture->imgArray, size);
	cv::Mat img(texture->rows, texture->cols, CV_32F, floatArr);
	cout << "Image size: " << texture->rows << " " << texture->cols << endl;
	cv::namedWindow( "Display the Image", CV_WINDOW_AUTOSIZE );  
    cv::imshow( "Display window", img);  
    cv::waitKey(5);
}


Texture** constructPyramids(cv::Mat inputTex, float sizeRatio, int numPyramids, int neighSize)
{
	Texture** ptrToPyramids;
	if((ptrToPyramids = (Texture **)malloc(sizeof(Texture *)*numPyramids)) == NULL){
		cerr << "Failed to malloc array"<< endl;
		exit(1);
	}
	int i;
	for(i=0;i<numPyramids;i++){
		//for the last level, just the copy the original texture, no need to resize
		if(i == numPyramids-1){
			int* ptrToArr = flattenImage(inputTex, inputTex.rows*inputTex.cols);
			//Texture curr = Texture(ptrToArr, inputTex.rows, inputTex.cols);			
			ptrToPyramids[i] = new Texture(ptrToArr, inputTex.rows, inputTex.cols);
			break;
		}
		cv::Mat currLevel;
		double currRatio = 1/pow(sizeRatio, numPyramids-1-i);
		int currRows = (int)inputTex.rows*currRatio;
		int currCols = (int)inputTex.cols*currRatio;
		currLevel = resizeImage(inputTex, currRows, currCols);
		int* ptrToArr = flattenImage(currLevel, currRows*currCols);
		ptrToPyramids[i] = new Texture(ptrToArr, currRows, currCols);
		currLevel.release();
	}
	return ptrToPyramids;
}

void freePyramids(Texture** textureList, int numPyramids)
{
	int i;
	for(i=0;i<numPyramids;i++){
		textureList[i]->freeTexture();
	}	
	free(textureList);
	cout << "Free the pyramids image done" << endl;
}

int* randomInitial(int size, float ratio)
{
	int* initialArr;
	if((initialArr = (int*)malloc(sizeof(int)*size)) == NULL){
		cerr << "Failed to malloc array"<< endl;
		exit(1);
	}
	srand(time(NULL));
	int i;
	for(i=0;i<size;i++){
		int random = rand()%100+1;
		initialArr[i] = (random <= ratio*100) ? 0:256;
	}
	return initialArr;
}

int* randomIndex(int size)
{
	int* initialArr;
	if((initialArr = (int*)malloc(sizeof(int)*size)) == NULL){
		cerr << "Failed to malloc array"<< endl;
		exit(1);
	}
	srand(time(NULL));
	int i;
	for(i=0;i<size;i++){
		initialArr[i] = rand()%5;
	}
	return initialArr;
}
