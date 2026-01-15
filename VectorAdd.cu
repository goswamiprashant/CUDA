#include <stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void add(int *a , int *b ,int *c ,int n)
{
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{

     // declaring variables for host
     int n=16;
     int h_A[n],h_B[n],h_C[n];


     // declaring variables for device

     int *d_A,*d_B,*d_C ;

  // initializing arrays with data

    for(int i=0;i<n;i++)
    {
        h_A[i] = i*2;
        h_B[i] = i*3;
    }



    // allocating memory for variables on device(GPU)

    cudaMalloc((void**)&d_A,sizeof(int)*n);
    cudaMalloc((void**)&d_B,sizeof(int)*n);
    cudaMalloc((void**)&d_C,sizeof(int)*n);

   // Copying data from host to device
     cudaMemcpy(d_A,h_A,sizeof(int)*n,cudaMemcpyHostToDevice);
     cudaMemcpy(d_B,h_B,sizeof(int)*n,cudaMemcpyHostToDevice);

  // Calling  Kernel

    add<<<1,n>>>(d_A,d_B,d_C,n) ;  // blocks per grid :n/8 , threads per block : 8 , total no of threads = n/8 *n  = n

   // Copying back the output recieved from device to host
    cudaMemcpy(h_C,d_C,sizeof(int)*n,cudaMemcpyDeviceToHost);

   // free memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);



     //Checking output
     printf("Sum :");
     for(int i =0;i<n;i++)
     {
         printf("\n%d + %d = %d",h_A[i],h_B[i],h_C[i]);
     }


}
