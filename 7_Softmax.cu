%%writefile softmax.cu
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<math.h>

__global__ void softmax(float* d_in,float *d_out ,int m,int n)
{
  
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;

    // getting max value 
  
  if(x<n && y<m)
   {


    float max_val=d_in[y*n];     // (row,col) in mat and (x,y) in co-ordinate system    
  
    for(int i=0;i<n;i++)
    {
        if (max_val < d_in[(y*n) + i])
        {
            max_val = d_in[y*n + i];
        }
        
    }

  //Compute exponentials and sum exponentials 
    float sum_exp=0;
    for(int i=0;i<n;i++)
    {   
      d_in[(y*n) + i] = exp(d_in[(y*n) + i]-max_val); 
      sum_exp+= d_in[(y*n) + i]; 
    }

    // Normalize
    for(int i=0;i<n;i++)
    {
        d_in[(y*n) + i] = d_in[(y*n) + i] / sum_exp;
    } 
  }
}


int main()
{
 
 int m =3,n=3;
 float h_A[3][3], h_B[3][3];

 for(int i=0;i<3;i++)
 {
     for(int j=0;j<=3;j++)
     {
        h_A[i][j] =(i*3)+j+1;
     }
 }




float *d_in,*d_out;

cudaMalloc((void**)&d_in,sizeof(float)*9);
cudaMalloc((void**)&d_out,sizeof(float)*9);

cudaMemcpy(d_in,h_A,sizeof(float)*9,cudaMemcpyHostToDevice);

dim3 threadsPerBlock(3,3);
dim3 blocksPerGrid(1);
softmax<<<blocksPerGrid,threadsPerBlock>>>(d_in,d_out,m,n);

cudaMemcpy(h_B,d_in,sizeof(float)*9,cudaMemcpyDeviceToHost);

 for(int i=0;i<3;i++)
 {
     for(int j=0;j<3;j++)
     {
        printf("%f",h_B[i][j]);
        printf("  "); 
     }
     printf("\n"); 
 }


cudaFree(d_in);
cudaFree(d_out);
}