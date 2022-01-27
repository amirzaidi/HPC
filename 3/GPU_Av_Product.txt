/*****************************************************************************
This function finds the product of Matrix A and vector V
*****************************************************************************/

// ****************************************************************************************************************************************************/
// parallelization method for the Matrix-vector multiplication as follows: 

// each thread handle a multiplication of each row of Matrix A and vector V;

// The share memory is limited for a block, instead of reading an entire row of matrix A or vector V from global memory to share memory, 
// a square submatrix of A is shared by a block, the size of square submatrix is BLOCK_SIZE*BLOCK_SIZE; Thus, a for-loop is used to
// handle a multiplication of each row of Matrix A and vector V step by step. In eacg step, two subvectors with size BLOCK_SIZE is multiplied.
//*****************************************************************************************************************************************************/


__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int aBegin = N * BLOCK_SIZE * bx;

    int aEnd   = aBegin + N - 1;
    int step  = BLOCK_SIZE;

    int bBegin = 0;//BLOCK_SIZE * bx;
    int bIndex=0;
    int aIndex =0;
    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += step, b += step)
    {

        __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];

        __shared__ float bs[BLOCK_SIZE];
        

        for (int aa = 0; aa < BLOCK_SIZE;aa+= 1)
        {
            aIndex = a+tx+aa*N;
            if( aIndex < N*N)
        	    As[tx+aa*BLOCK_SIZE] = g_MatA[aIndex];
		        else
        	    As[tx+aa*BLOCK_SIZE] = 0;
        }

        bIndex = b+tx;
   	    if(bIndex<N)   
		      bs[tx] = g_VecV[bIndex];
	      else
		      bs[tx] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[k+tx*BLOCK_SIZE] * bs[k];
        }//}
        __syncthreads();
    }

    g_VecW[ BLOCK_SIZE * bx + tx] = Csub;
}

