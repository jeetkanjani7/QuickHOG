#include "HOGHistogram.h"
#include "HOGUtils.h"
#include "cutil.h"

__device__ __constant__ float cenBound[3], halfBin[3], bandWidth[3], oneHalf = 0.5f;
__device__ __constant__ int tvbin[3];

texture<float, 1, cudaReadModeElementType> texGauss;
cudaArray* gaussArray;
cudaChannelFormatDesc channelDescGauss;

extern __shared__ float allShared[];

extern int rNoHistogramBins, rNoOfCellsX, rNoOfCellsY, rNoOfBlocksX, rNoOfBlocksY, rNumberOfWindowsX, rNumberOfWindowsY;

// wt scale == scale for weighting function span
__host__ void InitHistograms(int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY, int noHistogramBins, float wtscale)
{
	int i, j;

	float var2x = cellSizeX * blockSizeX / (2 * wtscale);
	float var2y = cellSizeY * blockSizeY / (2 * wtscale);
	var2x *= var2x * 2; var2y *= var2y * 2;

	float centerX = cellSizeX * blockSizeX / 2.0f;
	float centerY = cellSizeY * blockSizeY / 2.0f;

	float* weights = (float*)malloc(cellSizeX * blockSizeX * cellSizeY * blockSizeY * sizeof(float));

	for (i=0; i<cellSizeX * blockSizeX; i++)
	{
		for (j=0; j<cellSizeY * blockSizeY; j++)
		{
			float tx = i - centerX;
			float ty = j - centerY;

			tx *= tx / var2x;
			ty *= ty / var2y;

			weights[i + j * cellSizeX * blockSizeX] = exp(-(tx + ty));
		}
	}

	channelDescGauss = cudaCreateChannelDesc<float>();

	cutilSafeCall(cudaMallocArray(&gaussArray, &channelDescGauss, cellSizeX * blockSizeX * cellSizeY * blockSizeY, 1) );
	cutilSafeCall(cudaMemcpyToArray(gaussArray, 0, 0, weights, sizeof(float) * cellSizeX * blockSizeX * cellSizeY * blockSizeY, cudaMemcpyHostToDevice));

	int h_tvbin[3];
	float h_cenBound[3], h_halfBin[3], h_bandWidth[3];
	h_cenBound[0] = cellSizeX * blockSizeX / 2.0f;
	h_cenBound[1] = cellSizeY * blockSizeY / 2.0f;
	h_cenBound[2] = 180 / 2.0f; //TODO -> can be 360

	h_halfBin[0] = blockSizeX / 2.0f;
	h_halfBin[1] = blockSizeY / 2.0f;
	h_halfBin[2] = noHistogramBins / 2.0f;

	h_bandWidth[0] = (float) cellSizeX; h_bandWidth[0] = 1.0f / h_bandWidth[0];
	h_bandWidth[1] = (float) cellSizeY; h_bandWidth[1] = 1.0f / h_bandWidth[1];
	h_bandWidth[2] = 180.0f / (float) noHistogramBins; h_bandWidth[2] = 1.0f / h_bandWidth[2]; //TODO -> can be 360

	h_tvbin[0] = blockSizeX; h_tvbin[1] = blockSizeY; h_tvbin[2] = noHistogramBins;

	cutilSafeCall(cudaMemcpyToSymbol(cenBound, h_cenBound, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpyToSymbol(halfBin, h_halfBin, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpyToSymbol(bandWidth, h_bandWidth, 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpyToSymbol(tvbin, h_tvbin, 3 * sizeof(int), 0, cudaMemcpyHostToDevice));
}

__host__ void CloseHistogram()
{
}

__global__ void computeBlockHistogramsWithGauss(float2* inputImage, float1* blockHistograms, int noHistogramBins,
												int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
												int leftoverX, int leftoverY, int width, int height)
{
	int i;
	float2 localValue;
	float* shLocalHistograms = (float*)allShared;

	int cellIdx = threadIdx.y;
	int cellIdy = threadIdx.z;
	int columnId = threadIdx.x;

	int smemReadPos = __mul24(cellIdx, noHistogramBins) + __mul24(cellIdy, blockSizeX) * noHistogramBins;
	int gmemWritePos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, gridDim.x) * __mul24(blockDim.y, noHistogramBins) +
		__mul24(blockIdx.x, noHistogramBins) * blockDim.y + __mul24(blockIdx.y, gridDim.x) * __mul24(blockDim.y, noHistogramBins) * blockDim.z;

	int gmemReadStride = width;

	int gmemReadPos = leftoverX + __mul24(leftoverY, gmemReadStride) +
		(__mul24(blockIdx.x, cellSizeX) + __mul24(blockIdx.y, cellSizeY) * gmemReadStride)
		+ (columnId + __mul24(cellIdx, cellSizeX) + __mul24(cellIdy, cellSizeY) * gmemReadStride);

	int histogramSize = __mul24(noHistogramBins, blockSizeX) * blockSizeY;
	int smemLocalHistogramPos = (columnId + __mul24(cellIdx, cellSizeX)) * histogramSize + __mul24(cellIdy, histogramSize) * __mul24(blockSizeX, cellSizeX);

	int cmemReadPos = columnId + __mul24(cellIdx, cellSizeX) + __mul24(cellIdy, cellSizeY) * __mul24(cellSizeX, blockSizeX);

	float atx, aty;
	float pIx, pIy, pIz;

	int fIx, fIy, fIz;
	int cIx, cIy, cIz;
	float dx, dy, dz;
	float cx, cy, cz;

	bool lowervalidx, lowervalidy;
	bool uppervalidx, uppervalidy;
	bool canWrite;

	int offset;

	for (i=0; i<histogramSize; i++) shLocalHistograms[smemLocalHistogramPos + i] = 0;

#ifdef UNROLL_LOOPS
	int halfSizeYm1 = cellSizeY / 2 - 1;
#endif

	//if (blockIdx.x == 5 && blockIdx.y == 4)
	//{
	//	int asasa;
	//	asasa = 0;
	//	asasa++;
	//}

	for (i=0; i<cellSizeY; i++)
	{
		localValue = inputImage[gmemReadPos + i * gmemReadStride];
		localValue.x *= tex1D(texGauss, cmemReadPos + i * cellSizeX * blockSizeX);

		atx = cellIdx * cellSizeX + columnId + 0.5;
		aty = cellIdy * cellSizeY + i + 0.5;

		pIx = halfBin[0] - oneHalf + (atx - cenBound[0]) * bandWidth[0];
		pIy = halfBin[1] - oneHalf + (aty - cenBound[1]) * bandWidth[1];
		pIz = halfBin[2] - oneHalf + (localValue.y - cenBound[2]) * bandWidth[2];

		fIx = floorf(pIx); fIy = floorf(pIy); fIz = floorf(pIz);
		cIx = fIx + 1; cIy = fIy + 1; cIz = fIz + 1; //eq ceilf(pI.)

		dx = pIx - fIx; dy = pIy - fIy; dz = pIz - fIz;
		cx = 1 - dx; cy = 1 - dy; cz = 1 - dz;

		cIz %= tvbin[2];
		fIz %= tvbin[2];
		if (fIz < 0) fIz += tvbin[2];
		if (cIz < 0) cIz += tvbin[2];

#ifdef UNROLL_LOOPS
		if ((i & halfSizeYm1) == 0)
#endif
		{
			uppervalidx = !(cIx >= tvbin[0] - oneHalf || cIx < -oneHalf);
			uppervalidy = !(cIy >= tvbin[1] - oneHalf || cIy < -oneHalf);
			lowervalidx = !(fIx < -oneHalf || fIx >= tvbin[0] - oneHalf);
			lowervalidy = !(fIy < -oneHalf || fIy >= tvbin[1] - oneHalf);
		}

		canWrite = (lowervalidx) && (lowervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (fIx + fIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * cx * cy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * cx * cy * dz;
		}

		canWrite = (lowervalidx) && (uppervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (fIx + cIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * cx * dy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * cx * dy * dz;
		}

		canWrite = (uppervalidx) && (lowervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (cIx + fIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * dx * cy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * dx * cy * dz;
		}

		canWrite = (uppervalidx) && (uppervalidy);
		if (canWrite)
		{
			offset = smemLocalHistogramPos + (cIx + cIy * blockSizeY) * noHistogramBins;
			shLocalHistograms[offset + fIz] += localValue.x * dx * dy * cz;
			shLocalHistograms[offset + cIz] += localValue.x * dx * dy * dz;
		}
	}

	__syncthreads();

	//TODO -> aligned block size * cell size
	int smemTargetHistogramPos;
	for(unsigned int s = blockSizeY >> 1; s>0; s>>=1)
	{
		if (cellIdy < s && (cellIdy + s) < blockSizeY)
		{
			smemTargetHistogramPos = (columnId + __mul24(cellIdx, cellSizeX)) * histogramSize + __mul24((cellIdy + s), histogramSize) * __mul24(blockSizeX, cellSizeX);

	for (i=0; i<histogramSize; i++)
				shLocalHistograms[smemLocalHistogramPos + i] += shLocalHistograms[smemTargetHistogramPos + i];

		}

		__syncthreads();
	}

	for(unsigned int s = blockSizeX >> 1; s>0; s>>=1)
	{
		if (cellIdx < s && (cellIdx + s) < blockSizeX)
		{
			smemTargetHistogramPos = (columnId + __mul24((cellIdx + s), cellSizeX)) * histogramSize + __mul24(cellIdy, histogramSize) * __mul24(blockSizeX, cellSizeX);

			for (i=0; i<histogramSize; i++)
				shLocalHistograms[smemLocalHistogramPos + i] += shLocalHistograms[smemTargetHistogramPos + i];

		}

		__syncthreads();
	}

	for(unsigned int s = cellSizeX >> 1; s>0; s>>=1)
	{
		if (columnId < s && (columnId + s) < cellSizeX)
		{
			smemTargetHistogramPos = (columnId + s + __mul24(cellIdx, cellSizeX)) * histogramSize + __mul24(cellIdy, histogramSize) * __mul24(blockSizeX, cellSizeX);

			for (i=0; i<histogramSize; i++)
				shLocalHistograms[smemLocalHistogramPos + i] += shLocalHistograms[smemTargetHistogramPos + i];

		}

		__syncthreads();
	}

	if (columnId == 0)
	{
		//write result to gmem
		for (i=0; i<noHistogramBins; i++)
			blockHistograms[gmemWritePos + i].x = shLocalHistograms[smemReadPos + i];
	}

	if (blockIdx.x == 10 && blockIdx.y == 8)
	{
		int asasa;
		asasa = 0;
		asasa++;
	}
}

__host__ void ComputeBlockHistogramsWithGauss(float2* inputImage, float1* blockHistograms, int noHistogramBins,
											  int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
											  int windowSizeX, int windowSizeY,
											  int width, int height)
{
	int leftoverX;
	int leftoverY;

	dim3 hThreadSize, hBlockSize;

	rNoOfCellsX = width / cellSizeX;
	rNoOfCellsY = height / cellSizeY;

	rNoOfBlocksX = rNoOfCellsX - blockSizeX + 1;
	rNoOfBlocksY = rNoOfCellsY - blockSizeY + 1;

	rNumberOfWindowsX = (width-windowSizeX)/cellSizeX + 1;
	rNumberOfWindowsY = (height-windowSizeY)/cellSizeY + 1;

	leftoverX = (width - windowSizeX - cellSizeX * (rNumberOfWindowsX - 1))/2;
	leftoverY = (height - windowSizeY - cellSizeY * (rNumberOfWindowsY - 1))/2;

	hThreadSize = dim3(cellSizeX, blockSizeX, blockSizeY);
	hBlockSize = dim3(rNoOfBlocksX, rNoOfBlocksY);

	cutilSafeCall(cudaBindTextureToArray(texGauss, gaussArray, channelDescGauss));

	computeBlockHistogramsWithGauss<<<hBlockSize, hThreadSize, noHistogramBins * blockSizeX * blockSizeY * cellSizeX * blockSizeY * blockSizeX * sizeof(float) >>>
		(inputImage, blockHistograms, noHistogramBins, cellSizeX, cellSizeY, blockSizeX, blockSizeY, leftoverX, leftoverY, width, height);

	cutilSafeCall(cudaUnbindTexture(texGauss));
}

__host__ void NormalizeBlockHistograms(float1* blockHistograms, int noHistogramBins,
									   int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
									   int width, int height)
{
	dim3 hThreadSize, hBlockSize;

	rNoOfCellsX = width / cellSizeX;
	rNoOfCellsY = height / cellSizeY;

	rNoOfBlocksX = rNoOfCellsX - blockSizeX + 1;
	rNoOfBlocksY = rNoOfCellsY - blockSizeY + 1;

	hThreadSize = dim3(noHistogramBins, blockSizeX, blockSizeY);
	hBlockSize = dim3(rNoOfBlocksX, rNoOfBlocksY);

	int alignedBlockDimX = iClosestPowerOfTwo(noHistogramBins);
	int alignedBlockDimY = iClosestPowerOfTwo(blockSizeX);
	int alignedBlockDimZ = iClosestPowerOfTwo(blockSizeY);

	normalizeBlockHistograms<<<hBlockSize, hThreadSize, noHistogramBins * blockSizeX * blockSizeY * sizeof(float)>>>
		(blockHistograms, noHistogramBins,
		rNoOfBlocksX, rNoOfBlocksY, blockSizeX, blockSizeY,
		alignedBlockDimX, alignedBlockDimY, alignedBlockDimZ,
		noHistogramBins * rNoOfCellsX, rNoOfCellsY);

}

__global__ void normalizeBlockHistograms(float1 *blockHistograms, int noHistogramBins,
										 int rNoOfHOGBlocksX, int rNoOfHOGBlocksY,
										 int blockSizeX, int blockSizeY,
										 int alignedBlockDimX, int alignedBlockDimY, int alignedBlockDimZ,
										 int width, int height)
{
	int smemLocalHistogramPos, smemTargetHistogramPos, gmemPosBlock, gmemWritePosBlock;

	float* shLocalHistogram = (float*)allShared;

	float localValue, norm1, norm2; float eps2 = 0.01f;

	smemLocalHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;
	gmemPosBlock = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, gridDim.x) * __mul24(blockDim.y, blockDim.x) +
		threadIdx.x + __mul24(blockIdx.x, noHistogramBins) * blockDim.y + __mul24(blockIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) * blockDim.z;
	gmemWritePosBlock = __mul24(threadIdx.z, noHistogramBins) + __mul24(threadIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) +
		threadIdx.x + __mul24(blockIdx.x, noHistogramBins) * blockDim.y + __mul24(blockIdx.y, gridDim.x) * __mul24(blockDim.y, blockDim.x) * blockDim.z;

	localValue = blockHistograms[gmemPosBlock].x;
	shLocalHistogram[smemLocalHistogramPos] = localValue * localValue;

	if (blockIdx.x == 10 && blockIdx.y == 8)
	{
		int asasa;
		asasa = 0;
		asasa++;
	}

	__syncthreads();

	for(unsigned int s = alignedBlockDimZ >> 1; s>0; s>>=1)
	{
		if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z)
		{
			smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24((threadIdx.z + s), blockDim.x) * blockDim.y + threadIdx.x;
			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		__syncthreads();

	}

	for (unsigned int s = alignedBlockDimY >> 1; s>0; s>>=1)
	{
		if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y)
		{
			smemTargetHistogramPos = __mul24((threadIdx.y + s), noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;
			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		__syncthreads();

	}

	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		{
			smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + (threadIdx.x + s);
			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		__syncthreads();
	}

	//if (blockIdx.x == 5 && blockIdx.y == 4)
	//{
	//	int asasa;
	//	asasa = 0;
	//	asasa++;
	//}

	norm1 = sqrtf(shLocalHistogram[0]) + __mul24(noHistogramBins, blockSizeX) * blockSizeY;
	localValue /= norm1;

	localValue = fminf(0.2f, localValue); //why 0.2 ??

	__syncthreads();

	shLocalHistogram[smemLocalHistogramPos] = localValue * localValue;

	__syncthreads();

	for(unsigned int s = alignedBlockDimZ >> 1; s>0; s>>=1)
	{
		if (threadIdx.z < s && (threadIdx.z + s) < blockDim.z)
		{
			smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24((threadIdx.z + s), blockDim.x) * blockDim.y + threadIdx.x;
			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		__syncthreads();

	}

	for (unsigned int s = alignedBlockDimY >> 1; s>0; s>>=1)
	{
		if (threadIdx.y < s && (threadIdx.y + s) < blockDim.y)
		{
			smemTargetHistogramPos = __mul24((threadIdx.y + s), noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + threadIdx.x;
			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		__syncthreads();

	}

	for(unsigned int s = alignedBlockDimX >> 1; s>0; s>>=1)
	{
		if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x)
		{
			smemTargetHistogramPos = __mul24(threadIdx.y, noHistogramBins) + __mul24(threadIdx.z, blockDim.x) * blockDim.y + (threadIdx.x + s);
			shLocalHistogram[smemLocalHistogramPos] += shLocalHistogram[smemTargetHistogramPos];
		}

		__syncthreads();
	}

	norm2 = sqrtf(shLocalHistogram[0]) + eps2;
	localValue /= norm2;

	blockHistograms[gmemWritePosBlock].x = localValue;

	if (blockIdx.x == 10 && blockIdx.y == 8)
	{
		int asasa;
		asasa = 0;
		asasa++;
	}
}
