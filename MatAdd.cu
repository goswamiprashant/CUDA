#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void Matadd(int *a,int *b ,int *c,int N)
{
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N && j<N )
  {
     //  c[i][j] = a[i][j]+b[i][j];  this does not work bcoz we are using 1 dim array
         c[(i*N)+j] = a[(i*N)+j] +b[i*N+j];

  }
}
// Matrix Multiplication Kernel
int main()
{

 // host vars
  int N =30;
  int h_A[N][N],h_B[N][N],h_C[N][N];

// host var initialization
  for(int j =0;j<N;j++)
     for(int k=0;k<N;k++)
            {
              h_A[j][k]=j+k;
              h_B[j][k]=j*k;
            }
printf("Prior:");
for(int j =0;j<N;j++)
    for(int k=0;k<N;k++)
         printf("\n %d + %d = %d",h_A[j][k],h_B[j][k],h_C[j][k]);

// device vars
  int *d_A,*d_B,*d_C;

 // Mem allocation on device
   cudaMalloc((void**)&d_A,N*N*sizeof(int));
   cudaMalloc((void**)&d_B,N*N*sizeof(int));
   cudaMalloc((void**)&d_C,N*N*sizeof(int));

// copying data from host to device
  cudaMemcpy(d_A,h_A,N*N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,h_B,N*N*sizeof(int),cudaMemcpyHostToDevice);


// Calling Kernel

dim3 threadsPerBlock(N,N);
dim3 blocksPerGrid(1,1);
Matadd<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);

// Synchronize and check errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return -1;
  }

cudaMemcpy(h_C,d_C,N*N*sizeof(int),cudaMemcpyDeviceToHost);

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

printf("\nAfter: \n");
for(int j =0;j<N;j++){
    for(int k=0;k<N;k++)
       {
        printf(" %d + %d = %d ,",h_A[j][k],h_B[j][k],h_C[j][k]);
       }
        printf("\n");
}


return 0;

}

