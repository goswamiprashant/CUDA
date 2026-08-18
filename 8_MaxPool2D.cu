%%writefile MaxPool2D.cu
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void MaxPool2D(float *d_in ,float *d_out,int o_in,int p_in, int m,int n,int k_out,int l_out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
  //(y*p_in+x),(y*p_in+x)+1,(y*n+x)*p_in,((y*n+x)*p_in)+1
   
   if(x<l_out && y<k_out)
   {
     int x_upd = x*n,y_upd = y*m; 
     int index = y_upd*p_in + x_upd;
     float max_val =0.0;
      for(int i=0;i<n;i++)      
      {        
        for(int j=0;j<m;j++)
        {
          float temp = d_in[index+j+(i*p_in)];
            if(max_val<temp) 
                 max_val =temp;
        }
        }
        d_out[y*l_out + x] =max_val;
   }
    
}

int main()
{
    float h_in[4][4],h_out[2][2];
   int m =2;
   int n=2;
   int in_rows=4,in_cols=4;
   int out_rows=2,out_cols=2;
   
   
  
 // Initialization
    for(int i=0;i<4;i++)
     for(int j=0;j<4;j++)
       h_in[i][j] =i*4+j+1;
    
     

 // device vars
   float *d_in,*d_out;

   cudaMalloc((void**)&d_in,16*sizeof(float));
   cudaMalloc((void**)&d_out,4*sizeof(float));

   cudaMemcpy(d_in,h_in,16*sizeof(float),cudaMemcpyHostToDevice);

   dim3 threadsPerBlock(2,2);
   dim3 blocksPerGrid(1);
   MaxPool2D<<<blocksPerGrid,threadsPerBlock>>>(d_in,d_out,in_rows,in_cols,m,n,out_rows,out_cols);
   cudaMemcpy(h_out,d_out,sizeof(float)*4,cudaMemcpyDeviceToHost);

 for(int i=0;i<2;i++)
 {
     for(int j=0;j<2;j++)
     {
        printf("%f",h_out[i][j]);
        printf("  "); 
     }
     printf("\n"); 
 }


   cudaFree(d_in);
   cudaFree(d_out);
    
}