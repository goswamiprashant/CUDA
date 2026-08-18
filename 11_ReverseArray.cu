%%writefile RevArray.cu
#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda.h>

__global__ void RevArray(int *d_A, int n)
{
 int x = blockDim.x*blockIdx.x + threadIdx.x;
 int k =  ceil(n/2);
 if(x<k)
 {
    int temp =d_A[x] ;
    d_A[x] = d_A[n-1-x]; 
    d_A[n-1-x] =temp;
 }
    
}

int main()
{
 int n=256;
 int h_A[256],h_B[256];
 for(int k=0;k<n;k++)
     h_A[k] = k*k;

 // device vars
int *d_A;

cudaMalloc((void**)&d_A,n*sizeof(int));

cudaMemcpy(d_A,h_A,n*sizeof(int),cudaMemcpyHostToDevice);

RevArray<<<1,ceil(n/2)>>>(d_A,n);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return -1;
  }

cudaMemcpy(h_B,d_A,n*sizeof(int),cudaMemcpyDeviceToHost);
cudaFree(d_A);

for(int k=0;k<n;k++){
    printf("%d ",h_B[k]);
}

}