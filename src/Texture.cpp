/*
	This class implements ta a struct as well as properties and functions associated with an texture image.
*/
#ifndef TEXTURE_H__
#define TEXTURE_H__
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//The Texture class constains the image array for the texture, as well as its rows and cols.
class Texture
{
public:	
	int *imgArray;
	int rows;
	int cols;
	Texture(int *arr, int h, int w)
	{
		imgArray = arr;
		rows = h;
		cols = w;
	}
	
	void copyTexture(Texture* T)
	{
		rows = T->rows;
		cols = T->cols;
		imgArray = T->imgArray;
	}

	Texture(int h, int w, int flag)
	{
		rows = h;
		cols = w;
		if (flag) {
			if ((imgArray = (int *)malloc(sizeof(int)*rows*cols)) == NULL) {
				cerr << "Failed to malloc image array"<< endl;
				exit(1);
			}
		}
	}

	void freeTexture()
	{
		free(imgArray);
	}
};
#endif  /* TEXTURE_H__ */
