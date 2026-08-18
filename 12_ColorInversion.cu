%%writefile ColorInversion.cu
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>



__global__ void ColorInv(int *a , int *b ,int height , int width)
{
        int x = blockDim.x*blockIdx.x + threadIdx.x;
        int y = blockDim.y*blockIdx.y + threadIdx.y;

        if(x<width && y<height)
        {
         b[y*width+x] =255-a[y*width+x];
         b[y*width+x+1] =255-a[y*width+x+1];
         b[y*width+x+2] =255-a[y*width+2];
         b[y*width+x+3] =a[y*width+x+3];
        }
}

int main()
{
int width = 4 , height =8;
int size = width*height;
int image[height*width*4],h_B[height*width*4];
int min = 0;
int max = 255;
for(int i =0;i<size;i++)
{

int num = (rand() % (max - min + 1)) + min;

if((i+1)%4==0)
   {
    image[i] = num/255;
   }
  else{
     image[i] = num;
  }
}

// allocation of vars on device 

int *d_A ,*d_B;

cudaMalloc((void**)&d_A,size*sizeof(int));
cudaMalloc((void**)&d_B,size*sizeof(int));

cudaMemcpy(d_A,image,size*sizeof(int),cudaMemcpyHostToDevice);

dim3 ThreadsPerBlock(4,4);
dim3 BlocksPerGrid(1,2);

ColorInv<<<BlocksPerGrid,ThreadsPerBlock>>>(d_A,d_B,height,width);
// Synchronize and check errors

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      return -1;
  }

cudaMemcpy(h_B,d_B,size*sizeof(int),cudaMemcpyDeviceToHost);

for(int i=0;i<height;i++){
  for(int j=0;j<width;j++)
     printf("%d,",h_B[i*width+j]);
  printf("\n");
}

cudaFree(d_A);
cudaFree(d_B);


}