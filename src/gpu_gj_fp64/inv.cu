#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "MatrixOps.h"
#include "inv.h"

/*__global__ void kernel1(double *temp)
{
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        
        temp[index] = 0.0;
}*/

/**
 * @brief CUDA Kernel: Initialize identity matrix (Step 1 of Gauss-Jordan)
 *
 * Parallelizes the initialization of the result matrix as an identity matrix.
 * Each thread handles one element of the dim × dim matrix.
 *
 * Thread mapping:
 * - Global thread index: blockDim.x * blockIdx.x + threadIdx.x
 * - Row m = index / dim
 * - Column n = index % dim
 *
 * Total threads launched: dim × dim
 * Parallelism: O(dim²) independent operations executed simultaneously
 *
 * @param result Output: Identity matrix initialized (1.0 on diagonal, 0.0 elsewhere)
 * @param temp Working copy of input matrix (not used in this kernel)
 * @param dim Matrix dimension
 */
__global__ void kernel2(double *result, double *temp, unsigned int dim)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int m = index / dim;  // Row index
	unsigned int n = index % dim;  // Column index

	if(m == n)
	{
		result[m * dim + n] = 1.0;  // Diagonal elements
	}
	else
	{
		result[m * dim + n] = 0.0;  // Off-diagonal elements
	}
}

/**
 * @brief CUDA Kernel: Compute elimination ratios (Step 2a of Gauss-Jordan)
 *
 * For pivot row i, computes the ratio needed to eliminate column i in all other rows.
 * This kernel calculates: ratio[j] = temp[j,i] / temp[i,i] for all j != i
 *
 * Only threads where k==0 compute ratios (one thread per row).
 * This ratio is used in the subsequent kernel to perform row operations.
 *
 * Thread mapping:
 * - Row j = index / dim
 * - Column k = index % dim
 * - Only threads with k==0 and j!=i perform computation
 *
 * @param temp Working matrix (current state during elimination)
 * @param ratio Output: Elimination ratios for each row (length: dim)
 * @param i Current pivot row index (sequential outer loop in host code)
 * @param dim Matrix dimension
 */
__global__ void kernel_ratio(double *temp, double *ratio, unsigned int i, unsigned int dim)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int j = index / dim;  // Row index
        unsigned int k = index % dim;  // Column index

	// Compute ratio for row j (only once per row, when k==0)
	if(i != j && k ==0)
        {
              ratio[j] = temp[j * dim + i] / temp[i * dim + i];
        }
}

/**
 * @brief CUDA Kernel: Perform row elimination operations (Step 2b of Gauss-Jordan)
 *
 * This is the core Gauss-Jordan elimination kernel!
 * For pivot row i, eliminates column i in all other rows by subtracting
 * ratio[j] * row_i from row_j.
 *
 * Operations performed (for all j != i, all k):
 *   temp[j,k] = temp[j,k] - ratio[j] * temp[i,k]
 *   result[j,k] = result[j,k] - ratio[j] * result[i,k]
 *
 * Thread mapping:
 * - Row j = index / dim
 * - Column k = index % dim
 * - Each thread handles one matrix element
 *
 * Total threads launched: dim × dim
 * Parallelism: O(dim²) operations per pivot iteration
 *
 * Note: The pivot diagonal element temp[i,i] must be non-zero.
 * If zero, the matrix is singular and cannot be inverted.
 *
 * @param result Inverse matrix being computed (updated in-place)
 * @param temp Working matrix (updated in-place)
 * @param ratio Pre-computed elimination ratios from kernel_ratio
 * @param i Current pivot row index
 * @param dim Matrix dimension
 */
__global__ void kernel(double *result, double *temp, double *ratio, unsigned int i, unsigned int dim)
{
	// Check for singular matrix (zero pivot)
	if(temp[i * dim + i] == 0.0)
        {
        	for(unsigned int m = 0; m < dim * dim; m ++)
                {
                	result[m] = 0.0;  // Return zero matrix (singular)
                }

                return;
        }

        unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

        unsigned int j = index / dim;  // Row index
        unsigned int k = index % dim;  // Column index

	// Earlier implementation used shared memory for ratios (commented out)
	// Current version uses global memory for simplicity
//	__shared__ double ratio[300];

//        if(i != j && k ==0)
//        {
//		ratio[j] = temp[j * dim + i] / temp[i * dim + i];
//        }
//        __syncthreads();

        // Eliminate column i in row j (skip pivot row i)
        if(i != j)
        {
		//printf("here is ok\n");
                temp[j * dim + k] = temp[j * dim + k] - ratio[j] * temp[i * dim + k];
                result[j * dim + k] = result[j * dim + k] - ratio[j] * result[i * dim + k];
        }
}

/*__global__ void kernel3(double *result, double *temp, int i, unsigned int dim)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	result[i * dim + index] = result[i * dim + index] / temp[i * dim + i];
}*/

/**
 * @brief CUDA Kernel: Final normalization (Step 3 of Gauss-Jordan)
 *
 * After elimination, the temp matrix has been transformed to a diagonal matrix.
 * This kernel divides each row by its diagonal element to produce the identity
 * matrix in temp, and the corresponding inverse in result.
 *
 * Operation: result[j,k] = result[j,k] / temp[j,j]
 *
 * Thread mapping:
 * - Global index maps to all dim² elements
 * - Row j = index / dim
 * - Each thread normalizes one element
 *
 * Total threads launched: dim × dim
 * Parallelism: O(dim²) independent division operations
 *
 * @param result Inverse matrix (normalized to final result)
 * @param temp Diagonal matrix after elimination
 * @param dim Matrix dimension
 */
__global__ void kernel3(double *result, double *temp, unsigned int dim)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int j = index / dim;  // Row index for diagonal element

	// Divide each element by its row's diagonal value
	result[index] = result[index] / temp[j * dim + j];
}


/**
 * @brief CUDA-accelerated matrix inversion using Gauss-Jordan elimination
 *
 * This function implements the parallelized version of the matrix inversion
 * algorithm that was identified as the primary bottleneck in G-SLAM.
 *
 * Algorithm Overview (matching CPU version in MatrixOps.c):
 * 1. Initialize result as identity matrix (kernel2) - PARALLEL
 * 2. For each pivot row i (sequential loop on CPU):
 *    a. Compute elimination ratios (kernel_ratio) - PARALLEL
 *    b. Perform row operations (kernel) - PARALLEL
 * 3. Normalize by diagonal elements (kernel3) - PARALLEL
 *
 * Memory layout:
 * - result_d: Pre-allocated GPU memory for output (caller manages)
 * - temp_d: GPU working copy of input matrix
 * - ratio_d: GPU array for elimination ratios
 *
 * Grid/Block configuration:
 * - If dim ≤ 300: blocksPerGrid=dim, threadsPerBlock=dim (dim² total threads)
 * - If dim > 300: threadsPerBlock=300, blocksPerGrid=⌈(dim²)/300⌉
 *
 * Performance note:
 * The sequential loop (lines 185-189) launches dim kernel pairs.
 * This is the unavoidable sequential dependency in Gauss-Jordan.
 * However, each kernel launch parallelizes O(dim²) operations.
 *
 * @param result_d Pre-allocated GPU memory for inverse matrix (dim × dim)
 * @param mat Input matrix on CPU (dim × dim)
 * @param dim Matrix dimension
 */
void inv_cuda( double *result_d, double *mat, unsigned int dim )
{
	//double *temp = (double *)malloc ( sizeof( double ) * dim * dim ); // dim * dim

	double *temp_d, *ratio_d;

        // Allocate GPU memory for working matrix and ratio array
        cudaMalloc((double **)&temp_d, sizeof(double) * dim * dim);
        cudaMalloc((double **)&ratio_d, sizeof(double) * dim);

        // Copy input matrix to GPU
        cudaMemcpy(temp_d, mat, sizeof(double) * dim * dim, cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess)
        {
                printf("CUDA ERROR1: %s\n", cudaGetErrorString(err));
        }

	/*for ( unsigned int i = 0; i < dim * dim; i ++ ) {
		temp[ i ] = 0.0;
		//printf( "result[ %d ]: %f\n", i, temp[ i ] );
	}*/
	//kernel1<<< 300, 300 >>>(temp_d);
	
	/*for ( unsigned int m = 0; m < dim; m ++ ) {
		for ( unsigned int n = 0; n < dim; n ++ ) {
			temp[ m * dim + n ] = mat[ m * dim + n ];

			if( m == n ) {
				result[ m * dim + n ] = 1.0;
			}

			else {
				result[ m * dim + n ] = 0.0;
			}
		}
	}*/

	// Configure grid and block dimensions for kernel launches
	int threadsPerBlock;
	int blocksPerGrid;
	if(dim <= 300)
	{
		// Small matrices: Use dim blocks × dim threads (square grid)
		blocksPerGrid = dim;
		threadsPerBlock = dim;
	}
	else
	{
		// Large matrices: Use fixed block size and calculate grid size
		threadsPerBlock = 300;
		blocksPerGrid = (dim * dim + threadsPerBlock - 1) / threadsPerBlock;
	}

	// STEP 1: Initialize result as identity matrix (PARALLEL)
	kernel2<<< blocksPerGrid, threadsPerBlock >>>(result_d, temp_d, dim);
	
	cudaError_t err1 = cudaGetLastError();
	if(err1 != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(err1));
	}

	/*for(unsigned int i = 0; i < dim * dim; i++)
	{
       		unsigned int j = i % dim;
       		unsigned int h = i / dim;
       

           	//for(unsigned int m = 0; m < dim * dim; m++)
           	//{
             	//    result[m] = 0.0;
           	//}
		if(temp[h * dim + h] == 0.0)
		{
            		kernel1<<< 900, 100 >>>(result_d, temp_d, dim);
		
			return;
		}

       		if(j != h)
       		{
            
            		kernel2<<< 3, 100 >>>(result_d, temp_d, j, h, ratio_d, dim);

            	/*for(unsigned int k = 0; k < dim; k ++)
            	{
                	temp[j * dim + k] = temp[j * dim + k] - ratio * temp[h * dim + k];
                	printf("temp[%d]: %lf\n", j * dim + k, temp[j * dim + k]);
                	result[j * dim + k] = result[j * dim + k] - ratio * result[h * dim + k];
            	}

        	}
   	}*/
	// STEP 2: Gauss-Jordan elimination (SEQUENTIAL OUTER LOOP, PARALLEL INNER OPS)
	// This loop MUST be sequential due to data dependencies between iterations
	// However, each iteration parallelizes O(dim²) operations on the GPU
	for(unsigned int i = 0; i < dim; i ++)
	{
		// Compute elimination ratios for pivot row i (PARALLEL)
		kernel_ratio<<< blocksPerGrid, threadsPerBlock >>>(temp_d, ratio_d, i, dim);

		// Perform row operations to eliminate column i (PARALLEL)
		kernel<<< blocksPerGrid, threadsPerBlock >>>(result_d, temp_d, ratio_d, i, dim);
	}
	//printf("the third for loop\n");

	// STEP 3: Final normalization by diagonal elements (PARALLEL)
	kernel3<<< blocksPerGrid, threadsPerBlock >>>(result_d, temp_d, dim);

	cudaError_t err2 = cudaGetLastError();
	if (err2 != cudaSuccess)
	{
		printf("CUDA Error: %s\n", cudaGetErrorString(err2));
	}

	//cudaMemcpy(result, result_d, sizeof(double) * dim *dim, cudaMemcpyDeviceToHost);

	/*for(unsigned int i = 0; i < dim * dim; i++)
        {
                unsigned int j = i / dim;
                result[i] = result[i] / temp[j * dim + j];
                //printf("result[%d]: %lf\n", i, result[i]);
        }*/

	// Ensure all kernels complete before returning
	cudaDeviceSynchronize();

	// Clean up temporary GPU memory
	cudaFree(ratio_d);
	cudaFree(temp_d);
        //cudaFree(result_d);  // Managed by caller (result stays on GPU)

}

