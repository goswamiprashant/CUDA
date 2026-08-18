%%writefile Conv1D.cu
#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda.h>

__global__ void Conv1D(int *d_A, int *d_B,int *d_k,int n, int k)
{
    int x =blockDim.x*blockIdx.x+ threadIdx.x;
    int o = n-k+1;
    if(x<o)
    {
        d_B[x] = d_A[x]*d_k[0] + d_A[x+1]*d_k[1] + d_A[x+2]*d_k[2]; 
    }
}

int main()
{

int n=18,k=3,o=n-k+1;
int h_A[n],h_B[o],h_k[3]={1,0,-1};



for(int i=0;i<n;i++)
    h_A[i]=i;

// device vars
int *d_A,*d_k,*d_B;

cudaMalloc((void**)&d_A,n*sizeof(int));
cudaMalloc((void**)&d_B,o*sizeof(int));
cudaMalloc((void**)&d_k,k*sizeof(int));

cudaMemcpy(d_A,h_A,n*sizeof(int),cudaMemcpyHostToDevice);
cudaMemcpy(d_k,h_k,k*sizeof(int),cudaMemcpyHostToDevice);


Conv1D<<<1,16>>>(d_A,d_B,d_k,n,k);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return -1;
  }

cudaMemcpy(h_B,d_B,o*sizeof(int),cudaMemcpyDeviceToHost);
for(int i=0;i<o;i++)
   printf("%d ",h_B[i]); 

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_k);

}