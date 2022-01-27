// parallel HelloWorld using GPUs
// Simple starting example for CUDA program : this only works on arch 2 or higher
// Cong Xiao and Senlei Wang, Modified on Sep 2018

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N_THRDS    4 // Nr of threads in a block (blockDim)
#define N_BLKS    4 // Nr of blocks in a kernel (gridDim)

void checkCudaError(const char *error)
{
   if (cudaGetLastError() != cudaSuccess)
   {
      fprintf (stderr, "Cuda : %s\n",error);
      exit(EXIT_FAILURE);
   }
}

void checkCardVersion()
{
   cudaDeviceProp prop;
   
   cudaGetDeviceProperties(&prop, 0);
   checkCudaError("cudaGetDeviceProperties failed");
   
   fprintf(stderr,"This GPU has major architecture %d, minor %d \n",prop.major,prop.minor);
   if(prop.major < 2)
   {
      fprintf(stderr,"Need compute capability 2 or higher.\n");
      exit(1);
   }
}
   
__global__ void HelloworldOnGPU(void)
{
   int myid = (blockIdx.x * blockDim.x) + threadIdx.x;
   // Each thread simply  prints it's own string :
   printf( "Hello World, I am thread %d in block with index %d, my thread index is %d \n",
	   myid, blockIdx.x, threadIdx.x);
}

int main(void)
{
   checkCardVersion();
 
   HelloworldOnGPU <<< N_BLKS, N_THRDS >>> ();
   cudaDeviceSynchronize(); // without using synchronization, output won't be shown

   return 0;
}
