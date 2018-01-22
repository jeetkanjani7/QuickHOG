#include "HOGEngineDevice.h"
#include "HOGUtils.h"
#include "HOGConvolution.h"
#include "HOGHistogram.h"
#include "HOGSVMSlider.h"
#include "HOGScale.h"
#include "HOGPadding.h"
#include "cutil.h"
#include "HOGResult.h"
#define NMS_arb 128
using namespace HOG;

int hWidth, hHeight;
int hWidthROI, hHeightROI;
int hPaddedWidth, hPaddedHeight;
int rPaddedWidth, rPaddedHeight;

int minX, minY, maxX, maxY;

int hNoHistogramBins, rNoHistogramBins;

int hPaddingSizeX, hPaddingSizeY;
int hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, hWindowSizeX, hWindowSizeY;
int hNoOfCellsX, hNoOfCellsY, hNoOfBlocksX, hNoOfBlocksY;
int rNoOfCellsX, rNoOfCellsY, rNoOfBlocksX, rNoOfBlocksY;

int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;
int hNumberOfWindowsX, hNumberOfWindowsY;
int rNumberOfWindowsX, rNumberOfWindowsY;
int *resultId;

float4 *paddedRegisteredImage;

float1 *resizedPaddedImageF1;
float4 *resizedPaddedImageF4;

float2 *colorGradientsF2;

float1 *blockHistograms;
float1 *cellHistograms;

float1 *svmScores;

bool hUseGrayscale;

uchar1* outputTest1;
uchar4* outputTest4;

float* hResult;

int *resId_h;
int *resId_d;
int *counter;

float scaleRatio;
float startScale;
float endScale;
int scaleCount;

int num_elements;

int avSizeX, avSizeY, marginX, marginY;

HOGResult *CFR_res_d;
HOGResult *CFR_res_h;

HOGResult *Ox_windows;
bool *NMS_res_d;
bool *NMS_res_h;
extern uchar4* paddedRegisteredImageU4;


__host__ void InitHOG(int width, int height,
					  int _avSizeX, int _avSizeY,
					  int _marginX, int _marginY,
					  int cellSizeX, int cellSizeY,
					  int blockSizeX, int blockSizeY,
					  int windowSizeX, int windowSizeY,
					  int noOfHistogramBins, float wtscale,
					  float svmBias, float* svmWeights, int svmWeightsCount,
					  bool useGrayscale)
{
	cudaSetDevice(cutGetMaxGflopsDeviceId() );
	int i;
	int toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0;

	hWidth = width; hHeight = height;
	avSizeX = _avSizeX; avSizeY = _avSizeY; marginX = _marginX; marginY = _marginY;

	if (avSizeX) { toaddxx = hWidth * marginX / avSizeX; toaddxy = hHeight * marginY / avSizeX; }
	if (avSizeY) { toaddyx = hWidth * marginX / avSizeY; toaddyy = hHeight * marginY / avSizeY; }

	hPaddingSizeX = max(toaddxx, toaddyx); hPaddingSizeY = max(toaddxy, toaddyy);

	hPaddedWidth = hWidth + hPaddingSizeX*2;
	hPaddedHeight = hHeight + hPaddingSizeY*2;

	hUseGrayscale = useGrayscale;

	hNoHistogramBins = noOfHistogramBins;
	hCellSizeX = cellSizeX; hCellSizeY = cellSizeY; hBlockSizeX = blockSizeX; hBlockSizeY = blockSizeY;
	hWindowSizeX = windowSizeX; hWindowSizeY = windowSizeY;

	hNoOfCellsX = hPaddedWidth / cellSizeX;
	hNoOfCellsY = hPaddedHeight / cellSizeY;

	hNoOfBlocksX = hNoOfCellsX - blockSizeX + 1;
	hNoOfBlocksY = hNoOfCellsY - blockSizeY + 1;

	hNumberOfBlockPerWindowX = (windowSizeX - cellSizeX * blockSizeX) / cellSizeX + 1;
	hNumberOfBlockPerWindowY = (windowSizeY - cellSizeY * blockSizeY) / cellSizeY + 1;
	
	

	resId_h = (int *)malloc(sizeof(int)*32);
		

	hNumberOfWindowsX = 0;
	for (i=0; i<hNumberOfBlockPerWindowX; i++) hNumberOfWindowsX += (hNoOfBlocksX-i)/hNumberOfBlockPerWindowX;

	hNumberOfWindowsY = 0;
	for (i=0; i<hNumberOfBlockPerWindowY; i++) hNumberOfWindowsY += (hNoOfBlocksY-i)/hNumberOfBlockPerWindowY;

	scaleRatio = 1.05f;
	startScale = 1.0f;
	endScale = min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight / (float) hWindowSizeY);
	scaleCount = (int)floor(logf(endScale/startScale)/logf(scaleRatio)) + 1;
	

	//printf("values: %d -- %d -- %d ",hNumberOfWindowsX , hNumberOfWindowsY , scaleCount);
	num_elements = hNumberOfWindowsX * hNumberOfWindowsY * scaleCount;
	int zero_padded, pad=1;	
	
	while(num_elements>pad){ pad = pad << 1;}	
	num_elements = pad;
	printf("\n NUM_elements: %d\n", num_elements);
	CFR_res_h = (HOGResult *)malloc( sizeof(HOGResult) * num_elements);
	NMS_res_h = (bool *)malloc( sizeof(bool) * NMS_arb);
	Ox_windows = (HOGResult *)malloc( sizeof(HOGResult) * NMS_arb);

	cutilSafeCall(cudaMalloc((void**) &paddedRegisteredImage, sizeof(float4) * hPaddedWidth * hPaddedHeight));

	if (useGrayscale)
		cutilSafeCall(cudaMalloc((void**) &resizedPaddedImageF1, sizeof(float1) * hPaddedWidth * hPaddedHeight));
	else
		cutilSafeCall(cudaMalloc((void**) &resizedPaddedImageF4, sizeof(float4) * hPaddedWidth * hPaddedHeight));

	cutilSafeCall(cudaMalloc((void**) &colorGradientsF2, sizeof(float2) * hPaddedWidth * hPaddedHeight));
	cutilSafeCall(cudaMalloc((void**) &blockHistograms, sizeof(float1) * hNoOfBlocksX * hNoOfBlocksY * cellSizeX * cellSizeY * hNoHistogramBins));
	cutilSafeCall(cudaMalloc((void**) &cellHistograms, sizeof(float1) * hNoOfCellsX * hNoOfCellsY * hNoHistogramBins));

	cutilSafeCall(cudaMalloc((void**) &svmScores, sizeof(float1) * hNumberOfWindowsX * hNumberOfWindowsY * scaleCount));

	cutilSafeCall(cudaMemset(svmScores, 0, sizeof(float1)* hNumberOfWindowsX * hNumberOfWindowsY * scaleCount));
	
	cutilSafeCall(cudaMalloc((void**) &CFR_res_d, sizeof(HOGResult) * num_elements));
	cutilSafeCall(cudaMalloc((void**) &NMS_res_d, sizeof(bool) * NMS_arb));
	
	cutilSafeCall(cudaMemset(CFR_res_d, 0, sizeof(HOGResult) * num_elements));
	cutilSafeCall(cudaMemset(NMS_res_d, 1, sizeof(bool) * NMS_arb));
	
	InitConvolution(hPaddedWidth, hPaddedHeight, useGrayscale);
	InitHistograms(cellSizeX, cellSizeY, blockSizeX, blockSizeY, noOfHistogramBins, wtscale);
	InitSVM(svmBias, svmWeights, svmWeightsCount);
	InitScale(hPaddedWidth, hPaddedHeight);
	InitPadding(hPaddedWidth, hPaddedHeight);
	


	rPaddedWidth = hPaddedWidth;
	rPaddedHeight = hPaddedHeight;

	if (useGrayscale)
		cutilSafeCall(cudaMalloc((void**) &outputTest1, sizeof(uchar1) * hPaddedWidth * hPaddedHeight));
	else
		cutilSafeCall(cudaMalloc((void**) &outputTest4, sizeof(uchar4) * hPaddedWidth * hPaddedHeight));

	cutilSafeCall(cudaMallocHost((void**)&hResult, sizeof(float1) * hNumberOfWindowsX * hNumberOfWindowsY * scaleCount));
}



__host__ void InitCFR()
{
	dim3 grid_arb = dim3(1,hNumberOfWindowsY,scaleCount);
	dim3 block_arb = dim3(hNumberOfWindowsX,1,1);
	
	CFR_kernel<<<grid_arb, block_arb>>>(CFR_res_d, svmScores, hNumberOfWindowsX, hNumberOfWindowsY, hWindowSizeX, hWindowSizeY, startScale, hPaddedWidth, hPaddedHeight, hCellSizeX, hCellSizeY, hPaddingSizeX, hPaddingSizeY, minX, minY, resultId, resId_d, counter,scaleRatio);

	cudaDeviceSynchronize();
	int j, k, zero_padded;

/*
	num_elements = hNumberOfWindowsX*hNumberOfWindowsY*scaleCount;
	if(num_elements%128 != 0)
	{
		zero_padded = 128 - (num_elements % 128);
		num_elements = num_elements + zero_padded;
	}
*/	
//	printf("CFR last score: %f", CFR_res_d[num_elements-1].score);
	printf("\nnum elements:  %d\n", num_elements);  	
	/* Major step */
  	for (k = 2; k <=num_elements; k *= 2 ) {
    	/* Minor step */
    		for (j = k/2; j>0; j/=2) {
      			bitonic_sort_step<<<(num_elements/128), 128>>>(CFR_res_d, j, k);
																								
    		}
  	}


	InitNMS();
}


__global__ void bitonic_sort_step(HOGResult *CFR_res_d, int j, int k)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
  	ixj = i^j;
  
  	/* The threads with the lowest ids sort the array. */
  	if ((ixj)>i) 
	{
    		if ((i&k)==0) 
		{
      			/* Sort ascending */
      			if (CFR_res_d[i].score < CFR_res_d[ixj].score) 
      			{
			//printf("\n dev_values[%d](%f) > dev_values[%d].score(%f) :::  ",i,CFR_res_d	[i].score,ixj ,CFR_res_d[ixj].score );
        			/* exchange(i,ixj); */
        			HOGResult temp = CFR_res_d[i];
       				CFR_res_d[i] = CFR_res_d[ixj];
        			CFR_res_d[ixj] = temp;
      			}
    		}

    		if ((i&k)!=0) 
		{
      			/* Sort descending */
      			if (CFR_res_d[i].score > CFR_res_d[ixj].score) 
			{
        			/* exchange(i,ixj); */
        			HOGResult temp = CFR_res_d[i];
        			CFR_res_d[i] = CFR_res_d[ixj];
        			CFR_res_d[ixj] = temp;
      			}
    		}
  	}

}

__device__ int getGlobalIdx_3D_3D()
{
	int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x 
			 + gridDim.x * gridDim.y * blockIdx.z; 
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			  + (threadIdx.z * (blockDim.x * blockDim.y))
			  + (threadIdx.y * blockDim.x)
			  + threadIdx.x;
	return threadId;
}

__device__ int getLocalIdx_2D_2D()
{
	
	int threadId = threadIdx.x + threadIdx.y*blockDim.x; 
	return threadId;
}

__device__ int func(int a, int b)
{
	return (a % int(b) != 0) ? int(a / b + 1) : int(a / b);
}



__global__ void CFR_kernel(HOGResult *CFR_res,float1 *svmScores, int rNumberOfWindowsX, int rNumberOfWindowsY, int rWindowSizeX , int rWindowSizeY , int startScale, int rPaddedWidth, int rPaddedHeight, int rCellSizeX, int rCellSizeY, int rPaddingSizeX, int rPaddingSizeY, int minX, int minY, int* resultId, int *resId_d, int *counter, float rscaleRatio)
{	
    	int Id = getGlobalIdx_3D_3D();
	__shared__ int total_count;
	//extern __shared__ int write_test[];
	
	if(threadIdx.x == 0) total_count =0;
	
	int currentWidth, currentHeight, leftoverX, leftoverY;
	int i = blockIdx.z;
	int k = threadIdx.x;
	int j = blockIdx.y;
	
	float currentScale = startScale*powf(rscaleRatio,i);	
	
	
	float1* scaleoffset = svmScores + i* rNumberOfWindowsX * rNumberOfWindowsY;
	float1 score = scaleoffset[k + j*rNumberOfWindowsX];
	
	__syncthreads();

	if(score.x>0.0f)
	{	
		
		currentWidth = func(rPaddedWidth, currentScale);
		currentHeight = func(rPaddedHeight, currentScale);

		rNumberOfWindowsX = (currentWidth - rWindowSizeX) / rCellSizeX + 1;
		rNumberOfWindowsY = (currentHeight - rWindowSizeY) / rCellSizeY + 1;

		leftoverX = (currentWidth - rWindowSizeX - rCellSizeX * (rNumberOfWindowsX - 1)) / 2;
		leftoverY = (currentHeight - rWindowSizeY - rCellSizeY * (rNumberOfWindowsY - 1)) / 2;

		CFR_res[Id].origX = k * rCellSizeX + leftoverX;
		CFR_res[Id].origY = j * rCellSizeY + leftoverY;

		CFR_res[Id].width = (int)floorf((float)rWindowSizeX * currentScale);
		CFR_res[Id].height = (int)floorf((float)rWindowSizeY * currentScale);

		CFR_res[Id].x = (int)ceilf(currentScale * (CFR_res[Id].origX + rWindowSizeX / 2) - (float) rWindowSizeX * currentScale / 2) - rPaddingSizeX + minX;
					
		CFR_res[Id].y = (int)ceilf(currentScale * (CFR_res[Id].origY + rWindowSizeY / 2) - (float) rWindowSizeY * currentScale / 2) - rPaddingSizeY + minY;

		CFR_res[Id].scale = currentScale;
		CFR_res[Id].score = score.x;
		
	
		atomicAdd(&total_count,1);
		__syncthreads();
		//printf("Id:: %d -- %f ",Id,  CFR_res[Id].score);
		//atomicAdd(total_count,1);
		
		//printf("resultID: %f", resultId);
	}
		
	__syncthreads();
}	

__host__ void InitNMS()
{
	
	NMS_GPU<<<dim3(1,NMS_arb,1),NMS_arb>>>(CFR_res_d,NMS_res_d);	
}

__device__
float IOU_calc(HOGResult b1, HOGResult b2)
{
	float ai = (float)(b1.width + 1)*(b1.height + 1);
	float aj = (float)(b2.width + 1)*(b2.height + 1);
	float x_inter, x2_inter, y_inter, y2_inter;

	x_inter = max(b1.x,b2.x);
	y_inter = max(b1.y,b2.y);
	
	x2_inter = min((b1.x + b1.width),(b2.x + b2.width));
	y2_inter = min((b1.y + b1.height),(b2.y + b2.height));
	
	float w = (float)max((float)0, x2_inter - x_inter);  
	float h = (float)max((float)0, y2_inter - y_inter);  
	
	float inter = ((w*h)/(ai + aj - w*h));
	return inter;
}
	


__global__
void NMS_GPU( HOGResult *d, bool *NMS_res)
{
	int abs_y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int abs_x = (blockIdx.x * blockDim.x) + threadIdx.x;	 
    	float theta = 0.75;

    	if(d[abs_y].score >0 )
	{
		if(IOU_calc(d[abs_y],d[abs_x]) > theta)
		{
  			if(d[abs_x].score < d[abs_y].score)
    			{		
			
	 			NMS_res[abs_x] = false; 
	 		}
			
			else if(d[abs_x].score > d[abs_y].score)
			{
				NMS_res[abs_y] = false; 
			}
    		}
		else
		{
			//printf("\nthe x: %d and y: %d which escaped",abs_x, abs_y );
		}
  	}
	else NMS_res[abs_y] = false;
}

__host__ void CloseHOG()
{
	cutilSafeCall(cudaFree(paddedRegisteredImage));

	if (hUseGrayscale)
		cutilSafeCall(cudaFree(resizedPaddedImageF1));
	else
		cutilSafeCall(cudaFree(resizedPaddedImageF4));

	cutilSafeCall(cudaFree(colorGradientsF2));
	cutilSafeCall(cudaFree(blockHistograms));
	cutilSafeCall(cudaFree(cellHistograms));

	cutilSafeCall(cudaFree(svmScores));

	cutilSafeCall(cudaFree(CFR_res_d));
	cutilSafeCall(cudaFree(resultId));
	CloseConvolution();
	CloseHistogram();
	CloseSVM();
	CloseScale();
	ClosePadding();

	if (hUseGrayscale)
		cutilSafeCall(cudaFree(outputTest1));
	else
		cutilSafeCall(cudaFree(outputTest4));

	cutilSafeCall(cudaFreeHost(hResult));

	

	cudaThreadExit();
}

__host__ void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, float minScale, float maxScale)
{
	int i;
	minX = minx; minY = miny; maxX = maxx; maxY = maxy;
	PadHostImage((uchar4*)hostImage, paddedRegisteredImage, minX, minY, maxX, maxY);

	rPaddedWidth = hPaddedWidth; rPaddedHeight = hPaddedHeight;
	scaleRatio = 1.05f;
	startScale = (minScale < 0.0f) ? 1.0f : minScale;
	endScale = (maxScale < 0.0f) ? min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight / (float) hWindowSizeY) : maxScale;
	scaleCount = (int)floor(logf(endScale/startScale)/logf(scaleRatio)) + 1;
	
	float currentScale = startScale;

	ResetSVMScores(svmScores);

	for (i=0; i<scaleCount; i++)
	{
		DownscaleImage(0, scaleCount, i, currentScale, hUseGrayscale, paddedRegisteredImage, resizedPaddedImageF1, resizedPaddedImageF4);

		SetConvolutionSize(rPaddedWidth, rPaddedHeight);

		if(hUseGrayscale) ComputeColorGradients1to2(resizedPaddedImageF1, colorGradientsF2);
		else ComputeColorGradients4to2(resizedPaddedImageF4, colorGradientsF2);

		ComputeBlockHistogramsWithGauss(colorGradientsF2, blockHistograms, hNoHistogramBins,
			hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, hWindowSizeX, hWindowSizeY,  rPaddedWidth, rPaddedHeight);

		NormalizeBlockHistograms(blockHistograms, hNoHistogramBins, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, rPaddedWidth, rPaddedHeight);

		LinearSVMEvaluation(svmScores, blockHistograms, hNoHistogramBins, hWindowSizeX, hWindowSizeY, hCellSizeX, hCellSizeY,
			hBlockSizeX, hBlockSizeY, rNoOfBlocksX, rNoOfBlocksY, i, rPaddedWidth, rPaddedHeight);

		currentScale *= scaleRatio;
		
	}
	InitCFR();
}



__host__ float* EndHOGProcessing()
{
	cudaThreadSynchronize();
	cutilSafeCall(cudaMemcpy(hResult, svmScores,sizeof(float1)* hNumberOfWindowsX * hNumberOfWindowsY * scaleCount, cudaMemcpyDeviceToHost));

	
	return hResult;
}
__host__ bool* End_Oxsight_HOG()
{
	cutilSafeCall(cudaMemcpy(CFR_res_h, CFR_res_d,sizeof(HOGResult) * num_elements, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(NMS_res_h, NMS_res_d,sizeof(bool) * NMS_arb, cudaMemcpyDeviceToHost));

	int counter =0;
	for(int i = 0;counter !=NMS_arb, i<NMS_arb ; i++)
	{
		if(NMS_res_h[i]==true)
		{
			counter++;
		}	
		printf("\n CFR [%d] = score:%f x:%d y:%d",i,CFR_res_h[i].score,CFR_res_h[i].x, CFR_res_h[i].y);	
		Ox_windows[i] = CFR_res_h[i];	
	}
	
	return NMS_res_h;
}

__host__ HOGResult* getOxwindows()
{
	return Ox_windows;
}


__host__ void GetProcessedImage(unsigned char* hostImage, int imageType)
{
		switch (imageType)
		{
		case 0:
			Float4ToUchar4(resizedPaddedImageF4, outputTest4, rPaddedWidth, rPaddedHeight);
			break;
		case 1:
			Float2ToUchar4(colorGradientsF2, outputTest4, rPaddedWidth, rPaddedHeight, 0);
			break;
		case 2:
			Float2ToUchar4(colorGradientsF2, outputTest4, rPaddedWidth, rPaddedHeight, 1);
			break;
		case 3:
			cutilSafeCall(cudaMemcpy(hostImage, paddedRegisteredImageU4, sizeof(uchar4) * hPaddedWidth * hPaddedHeight, cudaMemcpyDeviceToHost));
			return;
		case 4:
			cutilSafeCall(cudaMemcpy2D(((uchar4*)hostImage) + minX + minY * hWidth, hWidth * sizeof(uchar4), 
				paddedRegisteredImageU4 + hPaddingSizeX + hPaddingSizeY * hPaddedWidth, hPaddedWidth * sizeof(uchar4),
				hWidthROI * sizeof(uchar4), hHeightROI, cudaMemcpyDeviceToHost));
			return;
		}
		cutilSafeCall(cudaMemcpy2D(hostImage, hPaddedWidth * sizeof(uchar4), outputTest4, rPaddedWidth * sizeof(uchar4),
			rPaddedWidth * sizeof(uchar4), rPaddedHeight, cudaMemcpyDeviceToHost));

	//cutilSafeCall(cudaMemcpy(hostImage, paddedRegisteredImage, sizeof(uchar4) * hPaddedWidth * hPaddedHeight, cudaMemcpyDeviceToHost));
}

__host__ void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
							   int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
							   int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
							   int *cNumberOfWindowsX, int *cNumberOfWindowsY,
							   int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY)
{
	*cStartScale = startScale;
	*cEndScale = endScale;
	*cScaleRatio = scaleRatio;
	*cScaleCount = scaleCount;
	*cPaddingSizeX = hPaddingSizeX;
	*cPaddingSizeY = hPaddingSizeY;
	*cPaddedWidth = hPaddedWidth;
	*cPaddedHeight = hPaddedHeight;
	*cNoOfCellsX = hNoOfCellsX;
	*cNoOfCellsY = hNoOfCellsY;
	*cNoOfBlocksX = hNoOfBlocksX;
	*cNoOfBlocksY = hNoOfBlocksY;
	*cNumberOfWindowsX = hNumberOfWindowsX;
	*cNumberOfWindowsY = hNumberOfWindowsY;
	*cNumberOfBlockPerWindowX = hNumberOfBlockPerWindowX;
	*cNumberOfBlockPerWindowY = hNumberOfBlockPerWindowY;
}

cudaArray *imageArray2 = 0;
texture<float4, 2, cudaReadModeElementType> tex2;
cudaChannelFormatDesc channelDescDownscale2;

__global__ void resizeFastBicubic3(float4 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale)
{
	int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	int i = __umul24(y, width) + x;

	float u = x*scale;
	float v = y*scale;

	if (x < width && y < height)
	{
		float4 cF;

		if (scale == 1.0f)
			cF = paddedRegisteredImage[x + y * width];
		else
			cF = tex2D(tex2, u, v);

		outputFloat[i] = cF;
	}
}

__host__ void DownscaleImage2(float scale, float4* paddedRegisteredImage,
							  float4* resizedPaddedImageF4, int width, int height,
							  int &rPaddedWidth, int &rPaddedHeight)
{
	dim3 hThreadSize, hBlockSize;

	hThreadSize = dim3(THREAD_SIZE_W, THREAD_SIZE_H);

	rPaddedWidth = iDivUpF(width, scale);
	rPaddedHeight = iDivUpF(height, scale);

	hBlockSize = dim3(iDivUp(rPaddedWidth, hThreadSize.x), iDivUp(rPaddedHeight, hThreadSize.y));

	cutilSafeCall(cudaMemcpyToArray(imageArray2, 0, 0, paddedRegisteredImage, sizeof(float4) * width * height, cudaMemcpyDeviceToDevice));
	cutilSafeCall(cudaBindTextureToArray(tex2, imageArray2, channelDescDownscale2));

	cutilSafeCall(cudaMemset(resizedPaddedImageF4, 0, width * height * sizeof(float4)));
	resizeFastBicubic3<<<hBlockSize, hThreadSize>>>((float4*)resizedPaddedImageF4, (float4*)paddedRegisteredImage, rPaddedWidth, rPaddedHeight, scale);

	cutilSafeCall(cudaUnbindTexture(tex2));
}

__host__ float3* CUDAImageRescale(float3* src, int width, int height, int &rWidth, int &rHeight, float scale)
{
	int i, j, offsetC, offsetL;

	float4* srcH; float4* srcD;
	float4* dstD; float4* dstH;
	float3 val3; float4 val4;

	channelDescDownscale2 = cudaCreateChannelDesc<float4>();
	tex2.filterMode = cudaFilterModeLinear; tex2.normalized = false;

	cudaMalloc((void**)&srcD, sizeof(float4) * width * height);
	cudaMalloc((void**)&dstD, sizeof(float4) * width * height);
	cudaMallocHost((void**)&srcH, sizeof(float4) * width * height);
	cudaMallocHost((void**)&dstH, sizeof(float4) * width * height);
	cutilSafeCall(cudaMallocArray(&imageArray2, &channelDescDownscale2, width, height) );

	for (i=0; i<width; i++)
	{
		for (j=0; j<height; j++)
		{
			offsetC = j + i * height;
			offsetL = j * width + i;

			val3 = src[offsetC];

			srcH[offsetL].x = val3.x;
			srcH[offsetL].y = val3.y;
			srcH[offsetL].z = val3.z;
		}
	}
	cudaMemcpy(srcD, srcH, sizeof(float4) * width * height, cudaMemcpyHostToDevice);

	DownscaleImage2(scale, srcD, dstD, width, height, rWidth, rHeight);

	cudaMemcpy(dstH, dstD, sizeof(float4) * rWidth * rHeight, cudaMemcpyDeviceToHost);

	float3* dst = (float3*) malloc (rWidth * rHeight * sizeof(float3));
	for (i=0; i<rWidth; i++)
	{
		for (j=0; j<rHeight; j++)
		{
			offsetC = j + i * rHeight;
			offsetL = j * rWidth + i;

			val4 = dstH[offsetL];

			dst[offsetC].x = val4.x;
			dst[offsetC].y = val4.y;
			dst[offsetC].z = val4.z;
		}
	}

	cutilSafeCall(cudaFreeArray(imageArray2));
	cudaFree(srcD);
	cudaFree(dstD);
	cudaFreeHost(srcH);
	cudaFreeHost(dstH);

	return dst;
}
