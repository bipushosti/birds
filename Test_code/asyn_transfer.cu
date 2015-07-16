

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N		(1024*1024)
#define FULL_SIZE	(N*20)
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void kernel(int *a,int *b,int *c);


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}




int main() 
{

	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,whichDevice));


	//cudaGetDevice(&whichDevice);
	//cudaGetDeviceProperties(&prop,whichDevice);
	if(!prop.deviceOverlap){
		printf("Device does not handle overlaps so streams are not possible\n");
		return 0;
	}


	cudaStream_t stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream1));	
	//cudaStreamCreate(&stream1);
	int *h_a,*h_b,*h_c;
	int *d_a,*d_b,*d_c;


	HANDLE_ERROR(cudaMalloc((void**)&d_a,N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b,N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c,N*sizeof(int)));
/*
	cudaMalloc((void**)&d_a,N*sizeof(int));
	cudaMalloc((void**)&d_b,N*sizeof(int));
	cudaMalloc((void**)&d_c,N*sizeof(int));
*/	
	HANDLE_ERROR(cudaHostAlloc((void**)&h_a,N*sizeof(int),cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_b,N*sizeof(int),cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_c,N*sizeof(int),cudaHostAllocDefault));
/*
	cudaHostAlloc((void**)&h_a,N*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_b,N*sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_c,N*sizeof(int),cudaHostAllocDefault);
*/

	for(int i = 0;i<FULL_SIZE;i++){
		h_a[i] = rand();
		h_b[i] = rand();
	}


	for(int i=0;i<FULL_SIZE;i+=N){
		//KERN_COMPLETE = 0;
		HANDLE_ERROR(cudaMemcpyAsync(d_a,h_a+i,N*sizeof(int),cudaMemcpyHostToDevice,stream1));
		HANDLE_ERROR(cudaMemcpyAsync(d_b,h_b+i,N*sizeof(int),cudaMemcpyHostToDevice,stream1));
		
		kernel<<<N/256,256,0,stream1>>>(d_a,d_b,d_c);
		
		HANDLE_ERROR(cudaMemcpyAsync(h_c+i,d_c,N*sizeof(int),cudaMemcpyDeviceToHost,stream1));
		
	}

	HANDLE_ERROR(cudaFreeHost(h_a));
	HANDLE_ERROR(cudaFreeHost(h_b));
	HANDLE_ERROR(cudaFreeHost(h_c));
	HANDLE_ERROR(cudaFree(d_a));
	HANDLE_ERROR(cudaFree(d_b));
	HANDLE_ERROR(cudaFree(d_c));
	
	HANDLE_ERROR(cudaStreamDestroy(stream1));
	return 0;
}


__global__ void kernel(int *a,int *b,int *c){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < N){
		int idx1 = (idx +1) %256;
		int idx2 = (idx +2) %256;

		float as = (a[idx]+a[idx1]+a[idx2])/3.0f;
		float bs = (b[idx]+b[idx1]+b[idx2])/3.0f;
		c[idx] = (as +bs)/2;
		//KERN_COMPLETE = 1;
	}
}


