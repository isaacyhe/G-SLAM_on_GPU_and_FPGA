#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "MatrixOps.h"
#include "inv.h"

__global__ void kernel2(float *result, unsigned int dim)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int m = index / dim;
	unsigned int n = index % dim;

	if(m == n)
	{
		result[m * dim + n] = 1.0;
	}
	else
	{
		result[m * dim + n] = 0.0;
	}
}

__global__ void kernel_ratio(float *temp, float *ratio, unsigned int i, unsigned int dim)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int j = index / dim;
        unsigned int k = index % dim;

	if(i != j && k ==0)
        {
              ratio[j] = temp[j * dim + i] / temp[i * dim + i];
        }
}

__global__ void kernel(float *result, float *temp, float *ratio, unsigned int i, unsigned int dim)
{
	if(temp[i * dim + i] == 0.0)
    	{
		for(unsigned int m = 0; m < dim * dim; m ++)
        	{
			result[m] = 0.0;
        	}

        	return;
	}

	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int j = index / dim;
	unsigned int k = index % dim;

    	if(i != j)
    	{
		temp[j * dim + k] = temp[j * dim + k] - ratio[j] * temp[i * dim + k];
        	result[j * dim + k] = result[j * dim + k] - ratio[j] * result[i * dim + k];
    	}
}

__global__ void kernel3(float *result, float *temp, unsigned int dim)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int j = index / dim;

	result[index] = result[index] / temp[j * dim +j];
}


void inv_cuda(float *result, float *mat, unsigned int dim) 
{
	float *temp_d, *ratio_d;
	cudaMalloc((void **)&temp_d, sizeof(float) * dim * dim);
	cudaMalloc((void **)&ratio_d, sizeof(float) * dim);

	cudaMemcpy(temp_d, mat, sizeof(float) * dim * dim, cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("cudamemcpy CUDA ERROR: %s\n", cudaGetErrorString(err));
	}

	int threadsPerBlock;
	int blocksPerGrid;
	if(dim <= 300)
	{
		threadsPerBlock = dim;
		blocksPerGrid = dim;
	}
	else
	{
		threadsPerBlock = 300;
		blocksPerGrid = (dim * dim + threadsPerBlock - 1) / threadsPerBlock;
	}

	kernel2<<< blocksPerGrid, threadsPerBlock >>>(result, dim);

	cudaError_t err1 = cudaGetLastError();
	if(err1 != cudaSuccess)
	{
		printf("kernel2 CUDA ERROR: %s\n", cudaGetErrorString(err1));
	}

	for(unsigned int i = 0; i < dim; i ++)
	{
		kernel_ratio<<< blocksPerGrid, threadsPerBlock >>>(temp_d, ratio_d, i, dim);
		kernel<<< blocksPerGrid, threadsPerBlock >>>(result, temp_d, ratio_d, i, dim);	
	}

	cudaError_t err2 = cudaGetLastError();
	if(err2 != cudaSuccess)
	{
		printf("for CUDA ERROR: %s\n", cudaGetErrorString(err2));
	}
 
	kernel3<<< blocksPerGrid, threadsPerBlock >>>(result, temp_d, dim);

	cudaDeviceSynchronize();

	cudaFree(ratio_d);
	cudaFree(temp_d);

}

__global__ void kernel_dot(float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x)
{
        float a_row[300];
        float b_column[300];
        
        unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

        for(unsigned int j = 0; j < a_x; j ++)
        {
                a_row[j] = a [index / b_x * a_x + j];
                b_column[j] = b[j * b_x + index % b_x];
                result[index] += a_row[j] * b_column[j];
        }
}

void dot_cuda(float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x, unsigned int dim ) 
{
        assert( a_x == b_y );

        kernel_dot<<< dim, 1 >>>(result, a, a_y, a_x, b, b_y, b_x);

	cudaDeviceSynchronize();
}
