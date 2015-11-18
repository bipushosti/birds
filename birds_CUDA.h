
#ifndef BIRDS_CUDA_H
#define BIRDS_CUDA_H

typedef struct
{
	//Host-side input data
	long int NumDays;	
	size_t size;

	float *h_udata,*h_vdata,*h_u10data,*h_v10data;
	float *h_precipData,*h_pressureData,*h_dir_u,*h_dir_v,*h_dirData,*h_lwData;

	//Device buffers
	float *d_udata,*d_vdata,*d_u10data,*d_v10data;
	float *d_precipData,*d_pressureData,*d_dir_u,*d_dir_v,*d_dirData,*d_lwData;

	//Stream for asynchronous command execution
	cudaStream_t stream;

} Stream_struct;

#endif
