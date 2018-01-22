#ifndef __CUDA_HOG__
#define __CUDA_HOG__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <cuda_gl_interop.h>
#include <cuda.h>

#include "HOGDefines.h"
#include "HOGResult.h"

using namespace HOG;
extern "C" __host__ void InitHOG(int width, int height,
								 int avSizeX, int avSizeY,
								 int marginX, int marginY,
								 int cellSizeX, int cellSizeY,
								 int blockSizeX, int blockSizeY,
								 int windowSizeX, int windowSizeY,
								 int noOfHistogramBins, float wtscale,
								 float svmBias, float* svmWeights, int svmWeightsCount,
								 bool useGrayscale);

extern "C" __host__ void CloseHOG();

extern "C" __host__ void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, float minScale, float maxScale);
extern "C" __host__ float* EndHOGProcessing();

extern "C"  __host__ void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
										   int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
										   int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
										   int *cNumberOfWindowsX, int *cNumberOfWindowsY,
										   int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY);

extern "C" __host__ void GetProcessedImage(unsigned char* hostImage, int imageType);

extern "C" __host__ float3* CUDAImageRescale(float3* src, int width, int height, int &rWidth, int &rHeight, float scale);
extern "C" __host__ bool* End_Oxsight_HOG();
extern "C" __host__ HOGResult* getOxwindows();
__host__ void InitCUDAHOG(int cellSizeX, int cellSizeY,
						  int blockSizeX, int blockSizeY,
						  int windowSizeX, int windowSizeY,
						  int noOfHistogramBins, float wtscale,
						  float svmBias, float* svmWeights, int svmWeightsCount,
						  bool useGrayscale);
__host__ void CloseCUDAHOG();

__host__ void InitCFR();

__global__ void CFR_kernel(HOGResult *,float1 *, int ,int, int, int, int, int, int, int, int, int, int, int, int, int *,int *, int *, float);
__device__ int func(int , int );

__device__ float IOU_calc(HOGResult , HOGResult );
__host__ void InitNMS();
__global__ void NMS_GPU(HOGResult *, bool *);
__global__ void bitonic_sort_step(HOGResult *, int, int);
#endif
