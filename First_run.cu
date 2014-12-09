
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PI 3.14159

#define XSIZE		55
#define YSIZE		95
#define LINESIZE	8*55+53

#define TIMESTEPS	20
//#define DESIRED_ROW	
//#define DESIRED_COL
#define STARTING_ROW	90
#define STARTING_COL	25
//What is the time limit? How long will the birds keep flying/migrating before they 
//just give up?

//Assuming min tail wind speed = 1km/hr
//Assuming best tail wind speed = 40km/hr
//Assuming Max tail wind speed = 80km/hr
//Assuming Max head wind speed = 30km/hr


//--------------------------------------------------------------------------------------------------------------------------

__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle);

//--------------------------------------------------------------------------------------------------------------------------

//Kernel to get angle and magnitude from u and v matrices
__global__ void get_resultant(float * u, float* v,float* resultantMatrix,float* resultantAngle)
{
	float angle;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + y * XSIZE; 

	if(x < 55) {
		resultantMatrix[index] = hypotf(u[index],v[index]);
		angle = asin(u[index]/resultantMatrix[index]) * 180/PI;
		if(angle < 0){
			angle = (360 - angle);
		}
		if(angle > 360) {
			angle = angle - 360;
		}
		resultantAngle[index] = angle;
		printf("%f,%f\n",resultantMatrix[index],angle);
	}	

}

/*
//Computation Kernel
__global__ void bird_thread(float* resultantMatrix,float* resultantAngle,int* coords_row,int* coords_col)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;	

	int pos_row,pos_col;
	pos_row = 90;
	pos_col = 25;
	
	float tempAngle = 0;
	int i;

	for(i = 0;i<TIMESTEPS;i++) {
		tempAngle = resultantAngle[pos_row * XSIZE + pos_col];
		if((tempAngle >= 0) && (tempAngle < 45)) {
			pos_row -= 1;
		} 
		else if((tempAngle >= 45) && (tempAngle < 90)) {
			pos_row -= 1;
			pos_col += 1;
		}
		else if((tempAngle >= 90) && (tempAngle < 135)) {
			pos_col += 1;
		}
		else if((tempAngle >= 135) && (tempAngle < 180)) {
			pos_row += 1;
			pos_col += 1;
		}
		else if((tempAngle >= 180) && (tempAngle < 225)) {
			pos_row += 1;
		}
		else if((tempAngle >= 225) && (tempAngle < 270)) {
			pos_row += 1;
			pos_col -= 1;
		}
		else if((tempAngle >= 270) && (tempAngle < 315)) {
			pos_col -= 1;
		}
		else if((tempAngle >= 315) && (tempAngle < 360)) {
			pos_row -= 1;
			pos_col -= 1;
		}
		
		coords_row[TIMESTEPS] = pos_row;
		coords_col[TIMESTEPS] = pos_col;
		printf("%d,%d\n",pos_row,pos_col);		
	
	}

	
	

}
*/


//--------------------------------------------------------------------------------------------------------------------------
int main()
{

	size_t limit;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 500 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
	cudaDeviceGetLimit(&limit,cudaLimitPrintfFifoSize);

	float udata[YSIZE * XSIZE];
	float vdata[YSIZE * XSIZE];

	FILE *udataTxt;
	FILE *vdataTxt;

	udataTxt = fopen("udata.txt","r");
	if(udataTxt == NULL) {
		perror("Cannot open udataTxt file\n");
		return -1;
	}

	vdataTxt =fopen("vdata.txt","r");
	if(vdataTxt == NULL) {
		perror("Cannot open vdataTxt file\n");
		return -1;
	}

	char line[LINESIZE];
	memset(line,'\0',sizeof(line));

	char tempVal[8];
	memset(tempVal,'\0',sizeof(tempVal));

	char* startPtr,*endPtr;

	int i,j;
	float Value;
	
	i=0;
	j=0;
	

	while(fgets(line,LINESIZE,vdataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<XSIZE;i++){

			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));

			if(i != (XSIZE - 1)) {

				endPtr = strchr(startPtr,' ');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				vdata[j * XSIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (XSIZE - 1)){

				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				vdata[j * XSIZE + i] = Value;
			}
			
			
		}
		j++;
	}	

	memset(line,'\0',sizeof(line));
	memset(tempVal,'\0',sizeof(tempVal));

	i=0;
	j=0;

	while(fgets(line,LINESIZE,udataTxt)!=NULL){
		startPtr = line;
		for(i=0;i<XSIZE;i++){

			Value = 0;
			memset(tempVal,'\0',sizeof(tempVal));

			if(i != (XSIZE - 1)) {

				endPtr = strchr(startPtr,' ');
				strncpy(tempVal,startPtr,endPtr-startPtr);
				Value = atof(tempVal);
				udata[j * XSIZE + i] = Value;
				endPtr = endPtr + 1;
				startPtr = endPtr;
			}
			else if(i == (XSIZE - 1)){

				strcpy(tempVal,startPtr);
				Value = atof(tempVal);
				udata[j * XSIZE + i] = Value;
			}
		//printf("%f\n",udata[j * XSIZE + i]);
		}
		j++;
	}	


	float resultantMatrix[XSIZE * YSIZE];
	float resultantAngle[XSIZE * YSIZE];
	
	resultantMatrix[YSIZE * XSIZE -1] = 834.0;	

	float *udataPtr,*vdataPtr,*resultantMatrixPtr,*resultantAnglePtr;

	cudaMalloc((void**)&udataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMemcpy(udataPtr,udata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&vdataPtr,XSIZE * YSIZE * sizeof(float));
	cudaMemcpy(vdataPtr,vdata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&resultantMatrixPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,XSIZE * YSIZE * sizeof(float));
	
	

	
	cudaMemcpy(vdataPtr,vdata,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 gridSize(1,95,1);
	dim3 blockSize(64,1,1);
	printf("Hello2\n");
	get_resultant<<<gridSize,blockSize>>>(udataPtr,vdataPtr,resultantMatrixPtr,resultantAnglePtr);
	printf("Hello3\n");
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
  	{
		printf("CUDA Error: %s\n", cudaGetErrorString(error));

    	// we can't recover from the error -- exit the program
    	return 0;
  	}

	cudaMemcpy(resultantMatrix,resultantMatrixPtr,YSIZE * XSIZE * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(resultantAngle,resultantAnglePtr,YSIZE * XSIZE * sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(udataPtr);
	cudaFree(vdataPtr);
	cudaFree(resultantMatrixPtr);
	cudaFree(resultantAnglePtr);
	

	//for(i=0;i< XSIZE * YSIZE;i++) {
	//	printf("%f\n",resultantAngle[i]);
	//}
//--------------------------------------------------------------------------------------------------------------------------
	/*int coords_row[TIMESTEPS];
	int coords_col[TIMESTEPS];
	int *coords_rowPtr,*coords_colPtr;

	cudaMalloc((void**)&coords_rowPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&coords_colPtr,TIMESTEPS * sizeof(int));
	cudaMalloc((void**)&resultantMatrixPtr,XSIZE * YSIZE * sizeof(float));
	cudaMalloc((void**)&resultantAnglePtr,XSIZE * YSIZE * sizeof(float));

	cudaMemcpy(resultantMatrixPtr,resultantMatrix,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(resultantAnglePtr,resultantAngle,XSIZE * YSIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 gridSize2(1,0,0);
	dim3 blockSize2(1,0,0);

	bird_thread<<<gridSize2,blockSize2>>>(resultantMatrix,resultantAngle,coords_row,coords_col);
	
	cudaMemcpy(coords_row,coords_rowPtr,TIMESTEPS * sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(coords_row,coords_colPtr,TIMESTEPS * sizeof(int),cudaMemcpyDeviceToHost);

	cudaFree(coords_rowPtr);
	cudaFree(coords_colPtr);
	cudaFree(resultantMatrixPtr);
	cudaFree(resultantAnglePtr);

	//printf("%d,%d\n",coords_row[0],coords_col[0]);
	*/
	return 0;

}





