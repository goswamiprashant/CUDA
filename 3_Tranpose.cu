%%writefile MatTrans.cu
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void MatTrans(int *a, int *b ,int rows ,int cols)
{

  int i = blockIdx.y*blockDim.y + threadIdx.y;   // to handle rows (corresp to inp)
  int j = blockIdx.x*blockDim.x + threadIdx.x;   // to handle cols (corresp to inp)

 if(i<rows && j<cols)
 {
  b[j*rows + i] = a[i*cols + j];
 }

}

int main(){

// Host var initialization


int rows =4,cols=8; // 16 ,32
int N = rows*cols;
int h_A[N] ,h_B[N]  ;
//host var initialization
 for(int i =0;i<rows*cols;i++)
  {
    h_A[i] = i;
  }

for(int i=0;i<rows;i++){
  for(int j=0;j<cols;j++)
     printf("%d,",h_A[i*cols+j]);
  printf("\n");
}
// device var initialization
int *d_A , *d_B;

// Memory allocation on device
cudaMalloc((void**)&d_A,rows*cols*sizeof(int));
cudaMalloc((void**)&d_B,rows*cols*sizeof(int));

// Moving data from CPU memor to GPU memory
cudaMemcpy(d_A,h_A,rows*cols*sizeof(int),cudaMemcpyHostToDevice);

// Tesla T4 consists : 1 SM --> 64 SPs , Total SMs --> 40 leads to Total CUDA Cores = 64*40 = 2560
// Total number of threads utilized = 512  ,SM utilized = 2^9 / 2^6 = 2^3

dim3 ThreadsPerBlock(4,4);
dim3 BlocksperGrid(2,2);

MatTrans<<<BlocksperGrid,ThreadsPerBlock>>>(d_A,d_B,rows,cols);

// Synchronize and check errors

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return -1;
  }


cudaMemcpy(h_B,d_B,rows*cols*sizeof(int),cudaMemcpyDeviceToHost);

for(int i=0;i<cols;i++){
  for(int j=0;j<rows;j++)
     printf("%d,",h_B[i*rows+j]);
  printf("\n");
}

cudaFree(d_A);
cudaFree(d_B);

}