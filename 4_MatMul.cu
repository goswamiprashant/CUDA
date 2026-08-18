#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void Matmul(int *a,int *b ,int *c,int N)
{
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N && j<N )
  {
    int temp =0;
    for(int k =0;k<N;k++)
     {
        temp += a[(i*N)+k]*b[(k*N)+j];
     }
      c[i*N+j]= temp;
  }
}
// Matrix Multiplication Kernel
int main()
{

 // host vars
  int N =3;
  int h_A[N][N],h_B[N][N],h_C[N][N];

// host var initialization
  for(int j =0;j<N;j++)
     for(int k=0;k<N;k++)
            {
              h_A[j][k]=j+k;
              h_B[j][k]=j*k;
            }
printf("Printing Matrix A:\n");
for(int j =0;j<N;j++)
{
   for(int k=0;k<N;k++)
         printf("%d,",h_A[j][k]);
   printf("\n");
}

printf("Printing Matrix B:\n");
for(int j =0;j<N;j++)
{
   for(int k=0;k<N;k++)
         printf("%d,",h_B[j][k]);
   printf("\n");
}

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
Matmul<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);

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
        printf(" %d ,",h_C[j][k]);
       }
        printf("\n");
}


return 0;

}

