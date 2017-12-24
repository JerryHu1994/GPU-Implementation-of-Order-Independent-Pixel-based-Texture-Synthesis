# GPU-Implementation-of-Order-Independent-Pixel-based-Texture-Synthesis
Final project for CS 759 High Performance Computing Class taken in UW-Madison

Here is documentation of Jieru Hu's ME 759 Final Project in UW-Madison 2017 Fall.

## Results:
This folder contains a set of texture synthesis results from the GPU Implementation.
## Sample:
This folder contains some sample texture for the program input.
## Src:
This folder contains all the source code.
## Timing:
This folder contains timing results stored in txt, scaling analysis plots, and a python script does all the plotting.

## Instruction for Execution

This code is tested and worked on Euler in 759 class.
To run the program on Euler:
```
cd src
sbatch ts_gpu.sh
```
The result texture image will be produce in src. The execution record will be produced in src/ts_gpu.out


To run the program on another machine, you would need both CUDA and OpenCV support. After obtain those libs, do the following to execute the program:
```
cd src
make
./ts_gpu texture1.jpg 400 3 11 2 3 //[image name] [output size] [# pyramid level] [neighborhood size] [size ratio] [#iterations per level]
```
