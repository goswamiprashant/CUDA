%%writefile ImageBatchNorm.cu
#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda.h>
const int batch =64,channel=3,height=128,width=128;
float h_A[batch][height][width][channel]; 
int N = batch*channel*height*width;

__global__ void ImageBatchNorm(float *d_C, int batch,int height,int width,int channel,float mean,float std)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j =  blockIdx.y * blockDim.y + threadIdx.y;
    int k =  blockIdx.z * blockDim.z + threadIdx.z;

    for(int l=0;l<batch;l++)
        {
                if (i <height && j<width && k < channel)
                { 
            
            int index = l*(height*width*channel)+ k*(height*width)+(j*height)+i;
            d_C[index] =  (d_C[index]-mean) /std;
            
                }
        }
        
}

int main()
{
    for(int i=0;i<64;i++)
    {  for(int j=0;j<128;j++)
        {  for(int k=0;k<128;k++)
            {
                for(int l=0;l<3;l++)
                    {
                    h_A[i][j][k][l]=2.0*l;
                    }
            }
      
            }    }

    float mean = 2.0,std =.8165;
    float *d_C;
    cudaMalloc((void**)&d_C,N*sizeof(float));


    cudaMemcpy(d_C,h_A,N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8,8,8);
    dim3 blocksPerGrid( 
(width + threadsPerBlock.x - 1) / threadsPerBlock.x,
(height + threadsPerBlock.y - 1) / threadsPerBlock.y,
(channel + threadsPerBlock.z - 1) / threadsPerBlock.z
);

    ImageBatchNorm<<<dim3(16,16,1),threadsPerBlock>>>(d_C,batch,height,width,channel,mean,std);

    cudaMemcpy(h_A,d_C,N*sizeof(float),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }


   for(int i=0;i<64;i++)
    {  for(int j=0;j<128;j++)
        {  for(int k=0;k<128;k++)
            {
                for(int l=0;l<3;l++)
                    {
                     printf("%f",h_A[i][j][k][l]);
                     printf("\n");     

                    }
           }
          
      
        }   
     }

cudaFree(d_C);

    }

