%%writefile Conv2D.cu
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void Conv2D(float *d_in ,float *d_fltr, float *d_out,int o_in,int p_in, int m,int n,int k_out,int l_out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
  //(y*p_in+x),(y*p_in+x)+1,(y*n+x)*p_in,((y*n+x)*p_in)+1
   if(x<l_out && y<k_out)
   {
     int index = y*p_in + x;
     float temp =0.0;
      for(int i=0;i<n;i++)      
      {        
        for(int j=0;j<m;j++)
        {
            temp= temp+(d_in[index+j+(i*p_in)]*d_fltr[i*n+j]);
        }
        }
        d_out[y*l_out + x] =temp;
   }
    
}

int main()
{
    float h_in[5][5],h_out[4][4],h_fltr[2][2];
   int m =2;
   int n=2;
   int o_in=5,p_in=5;
   int k_out=4,l_out=4;
 // Initialization
    for(int i=0;i<5;i++)
     for(int j=0;j<5;j++)
       h_in[i][j] =i*5+j;
    
     for(int i=0;i<2;i++)
       for(int j=0;j<2;j++)
          h_fltr[i][j] =1;


 // device vars
   float *d_in,*d_out,*d_fltr;

   cudaMalloc((void**)&d_in,25*sizeof(float));
   cudaMalloc((void**)&d_out,16*sizeof(float));
   cudaMalloc((void**)&d_fltr,4*sizeof(float));

   cudaMemcpy(d_in,h_in,25*sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(d_fltr,h_fltr,4*sizeof(float),cudaMemcpyHostToDevice);

   dim3 threadsPerBlock(4,4);
   dim3 blocksPerGrid(1);
   Conv2D<<<blocksPerGrid,threadsPerBlock>>>(d_in,d_fltr,d_out,o_in,p_in,m,n,k_out,l_out);
   cudaMemcpy(h_out,d_out,sizeof(float)*16,cudaMemcpyDeviceToHost);

 for(int i=0;i<4;i++)
 {
     for(int j=0;j<4;j++)
     {
        printf("%f",h_out[i][j]);
        printf("  "); 
     }
     printf("\n"); 
 }


   cudaFree(d_in);
   cudaFree(d_out);
   cudaFree(d_fltr);
    
}